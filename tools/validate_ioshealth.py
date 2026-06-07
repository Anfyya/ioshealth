from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "ioshealth"


EXPECTED_FILES = [
    ROOT / "ioshealth.xcodeproj" / "project.pbxproj",
    APP_DIR / "HealthAnomalyApp.swift",
    APP_DIR / "HealthKitPipeline.swift",
    APP_DIR / "Preprocessing.swift",
    APP_DIR / "ModelsAndScoring.swift",
    APP_DIR / "Storage.swift",
    APP_DIR / "ioshealth.entitlements",
    APP_DIR / "Resources" / "PopulationPriors.json",
]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def count_char(text: str, char: str) -> int:
    return sum(1 for item in text if item == char)


def validate_files(errors: list[str]) -> None:
    for path in EXPECTED_FILES:
        if not path.exists():
            fail(f"missing file: {path.relative_to(ROOT)}", errors)


def validate_project(errors: list[str]) -> None:
    pbxproj = ROOT / "ioshealth.xcodeproj" / "project.pbxproj"
    if not pbxproj.exists():
        return
    text = pbxproj.read_text(encoding="utf-8")
    required = [
        "HealthKit.framework",
        "PopulationPriors.json",
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


def validate_swift(errors: list[str]) -> None:
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
        if "if let error {" in text or "guard let objectType else" in text or "else if let last {" in text or "else if let next {" in text:
            fail(f"{rel}: uses optional binding shorthand in Swift 5 project", errors)

    healthkit = APP_DIR / "HealthKitPipeline.swift"
    if healthkit.exists():
        text = healthkit.read_text(encoding="utf-8")
        if "toShare: Set<HKSampleType>()" not in text:
            fail("HealthKitPipeline.swift: HealthKit share type is not explicit", errors)
        if "guard let objectType = objectType else { return nil }" not in text:
            fail("HealthKitPipeline.swift: sampleType does not safely unwrap objectType", errors)
        if "case .heartRate, .hrv, .oxygen, .respiratoryRate, .sleep: return .mean" not in text:
            fail("HealthKitPipeline.swift: sleep should be duration-weighted mean per bucket", errors)

    app = APP_DIR / "HealthAnomalyApp.swift"
    if app.exists():
        text = app.read_text(encoding="utf-8")
        if text.count("Task.detached(priority: .userInitiated)") < 2:
            fail("HealthAnomalyApp.swift: preprocessing/training are not both offloaded from MainActor", errors)
        if "private struct TrainingOutput: Sendable" not in text:
            fail("HealthAnomalyApp.swift: missing Sendable training output", errors)

    preprocessing = APP_DIR / "Preprocessing.swift"
    if preprocessing.exists():
        text = preprocessing.read_text(encoding="utf-8")
        required = [
            "let lastEventDate = sorted.map",
            "let effectiveEnd = event.end > event.start ?",
            "let overlapStart = event.start > bucketStart ?",
            "let overlapEnd = event.end < bucketEnd ?",
            "event.value * (weight / duration)",
        ]
        for marker in required:
            if marker not in text:
                fail(f"Preprocessing.swift: missing bucket overlap marker: {marker}", errors)

    scoring = APP_DIR / "ModelsAndScoring.swift"
    if scoring.exists():
        text = scoring.read_text(encoding="utf-8")
        if "let midpoint = Double(lower) + Double(upper - lower) * 0.5" not in text:
            fail("ModelsAndScoring.swift: percentile rank is not tie-aware", errors)


def validate_priors(errors: list[str]) -> None:
    path = APP_DIR / "Resources" / "PopulationPriors.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"PopulationPriors.json invalid JSON: {exc}", errors)
        return
    priors = payload.get("priors")
    if not isinstance(priors, list) or not priors:
        fail("PopulationPriors.json has no priors", errors)
        return
    seen_8f = False
    for prior in priors:
        name = prior.get("name", "<unnamed>")
        feature_count = prior.get("featureCount")
        features = prior.get("features")
        for field in ("windowSize", "horizon", "lag"):
            if not isinstance(prior.get(field), int) or prior[field] <= 0:
                fail(f"{name}: invalid {field}", errors)
        if not isinstance(feature_count, int) or feature_count <= 0:
            fail(f"{name}: invalid featureCount", errors)
            continue
        if feature_count == 8:
            seen_8f = True
        if not isinstance(features, list) or len(features) != feature_count:
            fail(f"{name}: feature length does not match featureCount", errors)
            continue
        for idx, feature in enumerate(features):
            cross = feature.get("cross")
            if not isinstance(cross, list) or len(cross) != feature_count:
                fail(f"{name}: feature {idx} cross length mismatch", errors)
            for field in ("intercept", "autoreg", "seasonal"):
                if not isinstance(feature.get(field), (int, float)):
                    fail(f"{name}: feature {idx} invalid {field}", errors)
    if not seen_8f:
        fail("PopulationPriors.json lacks 8-feature prior for HealthKit pipeline", errors)


def main() -> int:
    errors: list[str] = []
    validate_files(errors)
    validate_project(errors)
    validate_swift(errors)
    validate_priors(errors)
    if errors:
        print("ioshealth static validation failed:")
        for item in errors:
            print(f"- {item}")
        return 1
    print("ioshealth static validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
