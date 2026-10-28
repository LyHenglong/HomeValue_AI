"""Inference: validated price estimates with conformal ranges (model v2: optional ZIP, hybrid).

    from predict import load_artifacts, supported_locations, zip_options, predict_price
    art = load_artifacts()
    predict_price(art, "Texas", "Houston", bed=3, bath=2, house_size=1800, acre_lot=0.2, interval=0.8)
    predict_price(art, "Texas", "Houston", 3, 2, 1800, zip_code="77002")        # optional ZIP (string)

Every call returns a dict with `status` ("ok", "warning" or "rejected"), the estimate and range (None when rejected),
`model_path` ("no_zip", "zip_model" or "zip_fallback"), a `path_note` and human-readable `messages`.
Nothing is predicted for unsupported or unknown locations, invalid inputs, or ZIPs that do not belong to the city.
Without a ZIP the v1 model and v1 calibrated ranges are used unchanged. If models/zip_model.joblib or
models/zip_city_map.csv is absent, ZIP input is disabled and the module behaves exactly like v1 (rollback).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc  # noqa: E402  (needed to unpickle the saved pipeline)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DISCLAIMER = ("Estimate of the recorded price of sold-status records in the dataset (about Oct 2021 - May 2022). "
              "It is not an appraisal or a guaranteed market value.")


@dataclass
class Artifacts:
    bundle: dict                          # v1 model: used for every no-ZIP request and as the ZIP fallback
    support: pd.DataFrame
    zip_bundle: dict | None = None        # None -> ZIP input disabled
    zip_map: pd.DataFrame | None = None   # verified (state, city, zip_code) combinations from the cleaned data
    metadata: dict | None = None

    @property
    def zip_enabled(self) -> bool:
        return self.zip_bundle is not None and self.zip_map is not None


def load_artifacts(models_dir: Path = MODELS) -> Artifacts:
    d = Path(models_dir)
    bundle = joblib.load(d / "house_price_model.joblib")
    support = pd.read_csv(d / "location_support.csv", keep_default_na=False)
    zip_bundle = zip_map = None
    if (d / "zip_model.joblib").exists() and (d / "zip_city_map.csv").exists():
        zip_bundle = joblib.load(d / "zip_model.joblib")
        # dtype=str keeps leading zeros (00601 must not become 601)
        zip_map = pd.read_csv(d / "zip_city_map.csv", dtype={"zip_code": str, "state": str, "city": str}, keep_default_na=False)
    meta_path = d / "model_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    return Artifacts(bundle, support, zip_bundle, zip_map, metadata)


def zip_options(art: Artifacts, state: str, city: str) -> pd.DataFrame:
    """ZIPs recorded for this exact (state, city) in the dataset, with the route each would take."""
    if not art.zip_enabled:
        return pd.DataFrame(columns=["zip_code", "zip_train_rows", "route"])
    m = art.zip_map
    return m[(m.state == state) & (m.city == city)][["zip_code", "zip_train_rows", "route"]].sort_values("zip_code").reset_index(drop=True)


def supported_locations(art: Artifacts, include_limited: bool = True) -> pd.DataFrame:
    """Locations for the UI dropdowns, read from the validated support table (never hardcoded)."""
    keep = ["supported", "limited"] if include_limited else ["supported"]
    return art.support[art.support.status.isin(keep)][["state", "city", "status"]].reset_index(drop=True)


def _num(value, name, errors):
    """Return a finite float or None (recording an error); booleans and strings are not accepted."""
    if value is None or isinstance(value, (bool, str)):
        errors.append(f"{name} is missing or not a number.")
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} is not a number.")
        return None
    if not math.isfinite(v):
        errors.append(f"{name} must be a finite number.")
        return None
    return v


def _reject(messages, **extra):
    return {"status": "rejected", "estimate": None, "low": None, "high": None, "interval_level": None,
            "messages": messages, "disclaimer": DISCLAIMER, **extra}


def _check_zip(art: Artifacts, zip_code, state: str, city: str, errors: list) -> str | None:
    """Return the validated 5-character ZIP, None if no ZIP was given, or record an error."""
    if zip_code is None or (isinstance(zip_code, str) and not zip_code.strip()):
        return None
    if not art.zip_enabled:
        errors.append("ZIP-level estimates are not available in this installation; leave the ZIP code blank.")
        return None
    if not isinstance(zip_code, str):
        errors.append("Enter the ZIP code as text with all five digits (for example '02134'), so leading zeros are kept.")
        return None
    z = mc.normalize_zip(zip_code)
    if z is None:
        errors.append("ZIP code must have exactly five digits (for example 02134).")
        return None
    m = art.zip_map
    if not ((m.state == state) & (m.city == city) & (m.zip_code == z)).any():
        errors.append(f"ZIP {z} is not recorded for {city}, {state} in the dataset. Choose one of the listed ZIP codes "
                      "or leave the ZIP blank.")
        return None
    return z


def predict_price(art: Artifacts, state, city, bed, bath, house_size, acre_lot=None, interval: float = 0.8,
                  zip_code=None) -> dict:
    b = art.bundle
    errors: list[str] = []
    warnings: list[str] = []

    # ---- location ---------------------------------------------------------------------------------
    if not isinstance(state, str) or not isinstance(city, str) or not state.strip() or not city.strip():
        return _reject(["Choose a state and a city."])
    sup = art.support
    hit = sup[(sup.state.str.casefold() == state.strip().casefold()) & (sup.city.str.casefold() == city.strip().casefold())]
    if hit.empty:
        return _reject([f"'{city.strip()}, {state.strip()}' is not in the validated location list; no estimate is given."],
                       location_status="unknown")
    row = hit.iloc[0]
    state, city, loc_status = row.state, row.city, row.status
    if loc_status == "unsupported":
        return _reject([f"{city}, {state} has too little training data ({row.reason}); no estimate is given."],
                       location_status=loc_status)
    if loc_status == "limited":
        warnings.append(f"Limited coverage for {city}, {state}: {row.reason}. Treat the estimate with extra caution.")

    # ---- property inputs --------------------------------------------------------------------------
    lo_b, hi_b = mc.BOUNDS["bed"]
    v_bed = _num(bed, "Bedrooms", errors)
    if v_bed is not None and not (lo_b <= v_bed <= hi_b and v_bed == int(v_bed)):
        errors.append(f"Bedrooms must be a whole number from {lo_b} to {hi_b}.")
    lo_t, hi_t = mc.BOUNDS["bath"]
    v_bath = _num(bath, "Bathrooms", errors)
    if v_bath is not None:
        if not (lo_t <= v_bath <= hi_t):
            errors.append(f"Bathrooms must be from {lo_t} to {hi_t}.")
        elif v_bath != int(v_bath):
            warnings.append("The training data contains only whole bathroom counts; a fractional value is an extrapolation.")
    lo_s, hi_s = mc.BOUNDS["house_size"]
    v_size = _num(house_size, "House size", errors)
    if v_size is not None:
        if not (lo_s <= v_size <= hi_s):
            errors.append(f"House size must be between {lo_s:,} and {hi_s:,} square feet.")
        elif not (b["training_ranges"]["house_size_p0.5"] <= v_size <= b["training_ranges"]["house_size_p99.5"]):
            warnings.append("House size is unusual compared with the training data (outside its 0.5-99.5 percentile range); "
                            "the estimate is less reliable.")
    v_lot = np.nan
    if acre_lot is not None and not (isinstance(acre_lot, float) and math.isnan(acre_lot)):
        v = _num(acre_lot, "Lot size", errors)
        if v is not None:
            if not (0 < v <= mc.BOUNDS["acre_lot"][1]):
                errors.append(f"Lot size must be above 0 and at most {mc.BOUNDS['acre_lot'][1]:,} acres (leave it blank if unknown).")
            else:
                v_lot = v
                if v > b["training_ranges"]["acre_lot_p99.5"]:
                    warnings.append("Lot size is unusually large compared with the training data; the estimate is less reliable.")
    level_key = str(float(interval))
    if level_key not in b["conformal_quantiles"]:
        errors.append(f"interval must be one of {sorted(float(k) for k in b['conformal_quantiles'])}.")
    z = _check_zip(art, zip_code, state, city, errors)
    if errors:
        return _reject(errors, location_status=loc_status)

    # ---- prediction (same build_features + fitted pipeline as training) ------------------------------
    row_in = {"state": state, "city": city, "bed": v_bed, "bath": v_bath, "house_size": v_size, "acre_lot": v_lot}
    X = mc.build_features(pd.DataFrame([row_in]))
    p_v1 = float(mc.predict_dollars(b["model"], X)[0])
    pre = b["model"].regressor_.named_steps["pre"] if hasattr(b["model"], "regressor_") else b["model"].named_steps["pre"]
    Z_pre = pre.transform(X)
    if z is None:
        # v1 path, unchanged: v1 model, v1 spread model, v1 calibration
        est, path, note = p_v1, "no_zip", "No ZIP given: city-level estimate."
        sigma = max(float(b["scale_model"].predict(np.column_stack([Z_pre, np.log([est])]))[0]), b["scale_floor"])
        half = b["conformal_quantiles"][level_key] * sigma
    else:
        zb = art.zip_bundle
        n_zip = int(zb["train_zip_counts"].get(z, 0))
        use_zip = n_zip >= zb["min_zip_rows"]
        if use_zip:
            est = float(mc.predict_dollars(zb["zip_model"], mc.build_zip_features(pd.DataFrame([{**row_in, "zip_code": z}])))[0])
            path, note = "zip_model", f"ZIP-level estimate for {z}."
        else:
            est, path = p_v1, "zip_fallback"
            note = (f"ZIP {z} has only {n_zip} training record{'s' if n_zip != 1 else ''} (fewer than {zb['min_zip_rows']}), "
                    "so the city-level estimate is used; the range reflects the larger uncertainty for such ZIPs.")
        sigma = max(float(zb["scale_model"].predict(np.column_stack([Z_pre, np.log([est]), [float(use_zip)]]))[0]),
                    zb["scale_floor"])
        half = zb["conformal_quantiles"]["zip" if use_zip else "fallback"][level_key] * sigma
    if not (math.isfinite(est) and est > 0 and math.isfinite(half)):
        raise ValueError("non-finite prediction")
    low, high = est * math.exp(-half), est * math.exp(half)

    ppsf = est / v_size
    if not (5 <= ppsf <= 10_000):
        warnings.append(f"Implied price per square foot (${ppsf:,.0f}) is outside the range seen in cleaned data.")
    messages = warnings + [f"About {float(interval):.0%} of comparable held-out homes had a recorded price inside ranges built this way; "
                           "that is an average over the dataset, not a guarantee for this home."]
    return {"status": "warning" if warnings else "ok", "estimate": est, "low": low, "high": high,
            "interval_level": float(interval), "location_status": loc_status, "messages": messages, "disclaimer": DISCLAIMER,
            "zip_code": z, "model_path": path, "path_note": note}


if __name__ == "__main__":
    art = load_artifacts()
    top = supported_locations(art, include_limited=False).iloc[0]
    print(predict_price(art, top.state, top.city, bed=3, bath=2, house_size=1800, acre_lot=0.2))
