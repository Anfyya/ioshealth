from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


IOS_ROOT = Path(__file__).resolve().parents[1]
HEALTH_ROOT = IOS_ROOT.parent / "health"
RAIS_HOURLY = HEALTH_ROOT / "rais_anonymized" / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv"
OUTPUT = IOS_ROOT / "ioshealth" / "Resources" / "PopulationPriors.json"

FEATURE_COUNT = 8
WINDOW_SIZE = 252
STRIDE = 6
HORIZON = 24
LAG = 6


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def bucket_start(date_text: str, hour_text: str) -> datetime | None:
    try:
        day = datetime.strptime(date_text, "%Y-%m-%d")
        hour = int(float(hour_text))
    except (TypeError, ValueError):
        return None
    return day + timedelta(hours=(hour // 4) * 4)


def add_mean(bucket: dict[str, list[float]], index: int, value: float | None) -> None:
    if value is None:
        return
    bucket["sums"][index] += value
    bucket["counts"][index] += 1.0
    bucket["feature_mask"][index] = 1.0


def add_sum(bucket: dict[str, list[float]], index: int, value: float | None) -> None:
    if value is None:
        return
    bucket["sums"][index] += value
    bucket["counts"][index] += 1.0
    bucket["feature_mask"][index] = 1.0


def load_rais_hourly() -> tuple[np.ndarray, np.ndarray, int]:
    users: dict[str, dict[datetime, dict[str, list[float]]]] = defaultdict(dict)
    with RAIS_HOURLY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = row.get("id")
            start = bucket_start(row.get("date", ""), row.get("hour", ""))
            if not user_id or start is None:
                continue
            bucket = users[user_id].setdefault(
                start,
                {
                    "sums": [0.0] * FEATURE_COUNT,
                    "counts": [0.0] * FEATURE_COUNT,
                    "feature_mask": [0.0] * FEATURE_COUNT,
                },
            )
            add_mean(bucket, 0, to_float(row.get("bpm")))
            add_sum(bucket, 1, to_float(row.get("steps")))
            add_sum(bucket, 2, to_float(row.get("calories")))
            zone_2 = to_float(row.get("minutes_in_default_zone_2")) or 0.0
            zone_3 = to_float(row.get("minutes_in_default_zone_3")) or 0.0
            exercise = zone_2 + zone_3
            add_sum(bucket, 7, exercise if exercise > 0 else None)

    all_windows: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    valid_users = 0
    for buckets in users.values():
        if not buckets:
            continue
        starts = sorted(buckets)
        first, last = starts[0], starts[-1]
        total = int((last - first).total_seconds() // (4 * 3600)) + 1
        values = np.full((total, FEATURE_COUNT), np.nan, dtype=np.float64)
        mask = np.zeros(total, dtype=np.float64)
        for start, bucket in buckets.items():
            idx = int((start - first).total_seconds() // (4 * 3600))
            for feature in range(FEATURE_COUNT):
                count = bucket["counts"][feature]
                if count <= 0:
                    continue
                if feature == 0:
                    values[idx, feature] = bucket["sums"][feature] / count
                else:
                    values[idx, feature] = bucket["sums"][feature]
            if bucket["feature_mask"][0] > 0:
                mask[idx] = 1.0

        if mask.sum() < WINDOW_SIZE:
            continue
        valid_users += 1
        values = impute(values)
        valid = mask > 0
        for feature in range(FEATURE_COUNT):
            mean = float(values[valid, feature].mean()) if valid.any() else 0.0
            std = float(values[valid, feature].std()) if valid.any() else 1.0
            if not math.isfinite(std) or std < 1e-6:
                std = 1.0
            values[valid, feature] = (values[valid, feature] - mean) / std

        for start in range(0, total - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            all_windows.append(values[start:end].astype(np.float64))
            all_masks.append(mask[start:end].astype(np.float64))

    if not all_windows:
        raise RuntimeError("RAIS hourly data produced no windows")
    return np.stack(all_windows), np.stack(all_masks), valid_users


def impute(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for feature in range(out.shape[1]):
        col = out[:, feature]
        valid = np.where(np.isfinite(col))[0]
        if len(valid) == 0:
            out[:, feature] = 0.0
            continue
        for idx in range(1, len(col)):
            if not math.isfinite(out[idx, feature]):
                out[idx, feature] = out[idx - 1, feature]
        for idx in range(len(col) - 2, -1, -1):
            if not math.isfinite(out[idx, feature]):
                out[idx, feature] = out[idx + 1, feature]
        out[~np.isfinite(out[:, feature]), feature] = 0.0
    return out


def fit_prior(windows: np.ndarray, masks: np.ndarray) -> dict[str, object]:
    features = []
    start_t = max(LAG, WINDOW_SIZE - HORIZON)
    for target in range(FEATURE_COUNT):
        rows = []
        labels = []
        for window, mask in zip(windows, masks):
            for t in range(start_t, WINDOW_SIZE):
                if mask[t] <= 0:
                    continue
                prev = window[t - 1]
                seasonal = window[t - LAG]
                row = [1.0, prev[target], seasonal[target]]
                row.extend(prev[j] if j != target else 0.0 for j in range(FEATURE_COUNT))
                rows.append(row)
                labels.append(window[t, target])
        if not rows:
            weights = np.zeros(3 + FEATURE_COUNT, dtype=np.float64)
        else:
            x = np.asarray(rows, dtype=np.float64)
            y = np.asarray(labels, dtype=np.float64)
            reg = np.eye(x.shape[1]) * 1e-3
            reg[0, 0] = 0.0
            weights = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
        weights = np.where(np.isfinite(weights), weights, 0.0)
        cross = weights[3:].tolist()
        cross[target] = 0.0
        features.append(
            {
                "intercept": float(weights[0]),
                "autoreg": float(weights[1]),
                "seasonal": float(weights[2]),
                "cross": [float(v) for v in cross],
            }
        )
    return {
        "name": "RAIS71-hourly4h-8f-lifted",
        "source": "RAIS 71 participants hourly wearable data lifted to Apple Health 8-feature layout; HR, steps, calories and exercise-zone minutes are observed, missing Apple-only channels use zero-filled conservative priors",
        "featureCount": FEATURE_COUNT,
        "windowSize": WINDOW_SIZE,
        "horizon": HORIZON,
        "lag": LAG,
        "features": features,
    }


def main() -> int:
    if not RAIS_HOURLY.exists():
        raise FileNotFoundError(RAIS_HOURLY)
    windows, masks, valid_users = load_rais_hourly()
    prior = fit_prior(windows, masks)
    prior["source"] = f"{prior['source']}; usable users for 4h windows: {valid_users}/71; windows: {len(windows)}"

    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    priors = [item for item in payload["priors"] if item.get("name") != prior["name"]]
    insert_at = 0
    for idx, item in enumerate(priors):
        if item.get("name") == "PMData16-4h-8f-lifted":
            insert_at = idx + 1
            break
    priors.insert(insert_at, prior)
    payload["priors"] = priors
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {prior['name']}: windows={len(windows)}, usable_users={valid_users}/71")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
