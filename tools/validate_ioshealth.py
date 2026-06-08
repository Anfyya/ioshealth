from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "ioshealth"


EXPECTED_FILES = [
    ROOT / "ioshealth.xcodeproj" / "project.pbxproj",
    APP_DIR / "HealthAnomalyApp.swift",
    APP_DIR / "HealthKitPipeline.swift",
    APP_DIR / "Preprocessing.swift",
    APP_DIR / "ATModel.swift",
    APP_DIR / "ATEngine.swift",
    APP_DIR / "ATFusion.swift",
    APP_DIR / "ATStore.swift",
    APP_DIR / "ioshealth.entitlements",
    APP_DIR / "Resources" / "PredictionBase.json",
    APP_DIR / "Resources" / "PredictionBase.safetensors",
]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def count_char(text: str, char: str) -> int:
    return sum(1 for item in text if item == char)


def validate_files(errors: list[str]) -> None:
    for path in EXPECTED_FILES:
        if not path.exists():
            fail(f"missing file: {path.relative_to(ROOT)}", errors)
    weights = APP_DIR / "Resources" / "PredictionBase.safetensors"
    if weights.exists() and weights.stat().st_size <= 0:
        fail("PredictionBase.safetensors is empty", errors)


def validate_project(errors: list[str]) -> None:
    pbxproj = ROOT / "ioshealth.xcodeproj" / "project.pbxproj"
    if not pbxproj.exists():
        return
    text = pbxproj.read_text(encoding="utf-8")
    required = [
        "HealthKit.framework",
        "PredictionBase.safetensors",
        "PredictionBase.json",
        "ATModel.swift",
        "ATEngine.swift",
        "ATFusion.swift",
        "ATStore.swift",
        "CODE_SIGN_ENTITLEMENTS = ioshealth/ioshealth.entitlements;",
        "INFOPLIST_KEY_NSHealthShareUsageDescription",
    ]
    for marker in required:
        if marker not in text:
            fail(f"project missing marker: {marker}", errors)
    if "path = ioshealth/" in text:
        fail("project has duplicated ioshealth/ source path", errors)
    if "\ufffd" in text:
        fail("project contains replacement characters", errors)


def validate_swift_structure(errors: list[str]) -> None:
    shorthand = re.compile(r"\b(if|guard|else if) let [A-Za-z_][A-Za-z0-9_]* \{")
    for path in APP_DIR.glob("*.swift"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        braces = (count_char(text, "{"), count_char(text, "}"))
        parens = (count_char(text, "("), count_char(text, ")"))
        brackets = (count_char(text, "["), count_char(text, "]"))
        if braces[0] != braces[1]:
            fail(f"{rel}: brace mismatch {braces[0]}/{braces[1]}", errors)
        if parens[0] != parens[1]:
            fail(f"{rel}: paren mismatch {parens[0]}/{parens[1]}", errors)
        if brackets[0] != brackets[1]:
            fail(f"{rel}: bracket mismatch {brackets[0]}/{brackets[1]}", errors)
        if "`n" in text:
            fail(f"{rel}: contains literal PowerShell newline escape", errors)
        if "Array(windows[valEnd...])" in text:
            fail(f"{rel}: uses closed range for test split", errors)
        if "\ufffd" in text:
            fail(f"{rel}: contains replacement characters", errors)
        if shorthand.search(text):
            fail(f"{rel}: uses optional binding shorthand", errors)


def require_markers(path: Path, markers: list[str], errors: list[str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)
    for marker in markers:
        if marker not in text:
            fail(f"{rel}: missing marker: {marker}", errors)


def validate_pipeline(errors: list[str]) -> None:
    require_markers(
        APP_DIR / "HealthKitPipeline.swift",
        [
            "enum HealthDataRange",
            "case all, fiveYears, threeYears, sevenMonths",
            "func loadHistory(range: HealthDataRange = .all)",
            "HKQuery.predicateForSamples(withStart: start, end: end, options: options)",
            "toShare: Set<HKSampleType>()",
            "guard let objectType = objectType else { return nil }",
        ],
        errors,
    )
    require_markers(
        APP_DIR / "Preprocessing.swift",
        [
            "let bucketHours = 4",
            "let bucketHours: Int",
            "func bucketStart(at index: Int, bucketHours: Int)",
            "let lastEventDate = sorted.map",
            "let effectiveEnd = event.end > event.start ?",
            "let overlapStart = event.start > bucketStart ?",
            "event.value * (weight / duration)",
        ],
        errors,
    )
    require_markers(
        APP_DIR / "HealthAnomalyApp.swift",
        [
            "selectedRange: HealthDataRange = .all",
            "let plannedEpochs = TrainingPlanner.epochs",
            "let evaluation = dataset.train + dataset.validation + dataset.test",
            "RawAlertDetector.reports(from: events)",
            "TabView",
            "RangeSelector",
            "SignalSummaryCard",
            "PredictionBase.safetensors",
        ],
        errors,
    )
    app_text = (APP_DIR / "HealthAnomalyApp.swift").read_text(encoding="utf-8")
    if ".suffix(30)" in app_text:
        fail("HealthAnomalyApp.swift: still limits analysis display to suffix(30)", errors)
    if app_text.count("Task.detached(priority: .userInitiated)") < 2:
        fail("HealthAnomalyApp.swift: preprocessing/training are not both offloaded from MainActor", errors)


def validate_model(errors: list[str]) -> None:
    require_markers(
        APP_DIR / "ATModel.swift",
        [
            "final class ReconNet",
            "final class PredNet",
            "final class CrossAttention",
            "AnomalyAttentionLayer",
        ],
        errors,
    )
    require_markers(
        APP_DIR / "ATEngine.swift",
        [
            "struct ATScoreDetail",
            "PredictionBase.safetensors",
            "func trainRecon",
            "func reconScoreDetails",
            "func predScoreDetails",
            "func reconFeatureError(_ window: HealthWindow, stepIndex: Int? = nil)",
            "private static func details(from flat: [Float], rows: Int, cols: Int)",
        ],
        errors,
    )
    require_markers(
        APP_DIR / "ATFusion.swift",
        [
            "enum ReportSource",
            "case rawGuard, personalModel, predictionBase, fused",
            "static let watchThreshold = 85.0",
            "enum RawAlertDetector",
            "sleepDurationReports",
            "func mergeAdjacent",
        ],
        errors,
    )
    require_markers(
        APP_DIR / "ATStore.swift",
        [
            "let selectedRange: HealthDataRange?",
            "let trainingEpochs: Int?",
            "let evaluatedWindows: Int?",
            "let rawGuardCount: Int?",
        ],
        errors,
    )


def validate_prediction_base(errors: list[str]) -> None:
    path = APP_DIR / "Resources" / "PredictionBase.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"PredictionBase.json invalid JSON: {exc}", errors)
        return
    expected = {
        "model_type": "anomaly_transformer_crossattn_pred",
        "win_size": 252,
        "context_len": 228,
        "predict_horizon": 24,
        "enc_in": 8,
        "c_out": 8,
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            fail(f"PredictionBase.json: {key} expected {value!r}, got {payload.get(key)!r}", errors)
    features = payload.get("feature_order")
    if not isinstance(features, list) or len(features) != 8:
        fail("PredictionBase.json: feature_order must contain 8 features", errors)


def main() -> int:
    errors: list[str] = []
    validate_files(errors)
    validate_project(errors)
    validate_swift_structure(errors)
    validate_pipeline(errors)
    validate_model(errors)
    validate_prediction_base(errors)
    if errors:
        print("ioshealth static validation failed:")
        for item in errors:
            print(f"- {item}")
        return 1
    print("ioshealth static validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
