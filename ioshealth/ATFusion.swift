import Foundation

// MARK: - Fusion of model scores and raw guardrail events
//
// The app uses two time-series Transformer signals:
//   - personal reconstruction: trained on the user's own history
//   - prediction base: bundled cross-attention model trained before install
// Raw guardrail alerts run before bucketing, so short events such as a 5 minute
// heart-rate spike are not hidden by the 4 hour model window.

let featureDisplayNames = [
    "心率", "步数", "活动消耗", "心率变异性", "血氧", "呼吸频率", "睡眠", "运动时长",
]

enum RiskBand: String, Codable, Hashable, Sendable {
    case normal, watch, elevated, critical

    var severity: Int {
        switch self {
        case .normal: return 0
        case .watch: return 1
        case .elevated: return 2
        case .critical: return 3
        }
    }
}

enum ReportSource: String, Codable, Hashable, Sendable {
    case rawGuard, personalModel, predictionBase, fused

    var title: String {
        switch self {
        case .rawGuard: return "原始数据哨兵"
        case .personalModel: return "个人模型"
        case .predictionBase: return "通用基座"
        case .fused: return "双模型一致"
        }
    }
}

struct HealthReport: Identifiable, Codable, Hashable, Sendable {
    var id = UUID()
    var startedAt: Date
    var endedAt: Date
    var reconPercentile: Double
    var predPercentile: Double
    var finalPercentile: Double
    var band: RiskBand
    var source: ReportSource
    var topFeature: String
    var headline: String
    var detail: String
    var evidence: [String]
    var featureErrors: [Double]

    var durationText: String {
        let seconds = max(60, Int(endedAt.timeIntervalSince(startedAt)))
        let minutes = max(1, seconds / 60)
        if minutes < 60 { return "\(minutes) 分钟" }
        let hours = minutes / 60
        let remainMinutes = minutes % 60
        if hours < 24 {
            return remainMinutes == 0 ? "\(hours) 小时" : "\(hours) 小时 \(remainMinutes) 分钟"
        }
        let days = hours / 24
        let remainHours = hours % 24
        return remainHours == 0 ? "\(days) 天" : "\(days) 天 \(remainHours) 小时"
    }
}

struct Calibrator: Codable, Hashable, Sendable {
    let sorted: [Double]
    init(_ scores: [Double]) { sorted = scores.sorted() }

    func percentile(_ value: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        var lower = 0
        var upper = sorted.count
        while lower < upper {
            let mid = (lower + upper) / 2
            if sorted[mid] <= value { lower = mid + 1 } else { upper = mid }
        }
        return 100.0 * Double(lower) / Double(sorted.count)
    }
}

enum Fusion {
    static let watchThreshold = 85.0
    static let elevatedThreshold = 93.0
    static let criticalThreshold = 98.0

    static func band(forFinal final: Double) -> RiskBand {
        if final >= criticalThreshold { return .critical }
        if final >= elevatedThreshold { return .elevated }
        if final >= watchThreshold { return .watch }
        return .normal
    }

    static func modelReport(
        startedAt: Date,
        endedAt: Date,
        reconScore: Double,
        predScore: Double,
        reconCal: Calibrator,
        predCal: Calibrator,
        featureErrors: [Double]
    ) -> HealthReport {
        let recon = reconCal.percentile(reconScore)
        let pred = predCal.percentile(predScore)
        let final = max(recon, pred)
        let band = band(forFinal: final)
        let topIndex = featureErrors.indices.max(by: { featureErrors[$0] < featureErrors[$1] })
        let topName = topIndex != nil && topIndex! < featureDisplayNames.count ? featureDisplayNames[topIndex!] : "身体信号"
        let source = dominantSource(recon: recon, pred: pred, band: band)
        let phrase = phrasing(band: band, source: source, topName: topName)
        let evidence = [
            "个人模型历史分位 \(formatPercent(recon))",
            "通用基座历史分位 \(formatPercent(pred))",
            "预警线: 观察 \(Int(watchThreshold))%, 异常 \(Int(elevatedThreshold))%, 高危 \(Int(criticalThreshold))%",
            "模型时间粒度: 4 小时桶",
        ]
        return HealthReport(
            startedAt: startedAt,
            endedAt: endedAt,
            reconPercentile: recon,
            predPercentile: pred,
            finalPercentile: final,
            band: band,
            source: source,
            topFeature: band == .normal ? "" : topName,
            headline: phrase.headline,
            detail: phrase.detail,
            evidence: evidence,
            featureErrors: featureErrors)
    }

    static func rawReport(
        startedAt: Date,
        endedAt: Date,
        band: RiskBand,
        feature: String,
        valueText: String,
        headline: String,
        detail: String
    ) -> HealthReport {
        HealthReport(
            startedAt: startedAt,
            endedAt: max(endedAt, startedAt.addingTimeInterval(60)),
            reconPercentile: 0,
            predPercentile: 0,
            finalPercentile: band == .critical ? 100 : (band == .elevated ? 96 : 88),
            band: band,
            source: .rawGuard,
            topFeature: feature,
            headline: headline,
            detail: detail,
            evidence: [
                valueText,
                "原始样本时间粒度: 保留 HealthKit 样本的开始和结束时间",
                "这条规则先于 4 小时模型桶运行",
            ],
            featureErrors: Array(repeating: 0, count: featureDisplayNames.count))
    }

    static func mergeAdjacent(_ reports: [HealthReport]) -> [HealthReport] {
        let sortedReports = reports.sorted {
            if $0.startedAt == $1.startedAt { return $0.band.severity > $1.band.severity }
            return $0.startedAt < $1.startedAt
        }
        var merged: [HealthReport] = []
        for report in sortedReports {
            guard var last = merged.last else {
                merged.append(report)
                continue
            }
            let gap = report.startedAt.timeIntervalSince(last.endedAt)
            let canMerge = gap <= 4 * 3600
                && report.source == last.source
                && report.band == last.band
                && report.topFeature == last.topFeature
            if canMerge {
                last.endedAt = max(last.endedAt, report.endedAt)
                last.finalPercentile = max(last.finalPercentile, report.finalPercentile)
                last.reconPercentile = max(last.reconPercentile, report.reconPercentile)
                last.predPercentile = max(last.predPercentile, report.predPercentile)
                last.evidence = appendUnique(last.evidence, report.evidence)
                merged[merged.count - 1] = last
            } else {
                merged.append(report)
            }
        }
        return merged
    }

    private static func dominantSource(recon: Double, pred: Double, band: RiskBand) -> ReportSource {
        guard band != .normal else { return .fused }
        if recon >= watchThreshold && pred >= watchThreshold { return .fused }
        return recon >= pred ? .personalModel : .predictionBase
    }

    private static func phrasing(band: RiskBand, source: ReportSource, topName: String) -> (headline: String, detail: String) {
        if band == .normal {
            return ("未达到异常线", "这段时间没有超过当前预警线,所以不会作为异常段展示。")
        }
        let level: String
        switch band {
        case .critical: level = "很明显"
        case .elevated: level = "明显"
        case .watch: level = "值得观察"
        case .normal: level = ""
        }
        switch source {
        case .personalModel:
            return (
                "\(topName) 和其它指标的组合关系\(level)偏离平时",
                "个人模型在这段时间的重建误差升高,意思是这些指标放在一起看不像你训练集里的常见状态。"
            )
        case .predictionBase:
            return (
                "\(topName) 的走势被通用基座判为\(level)偏离",
                "通用预测基座用前面的节律预测后续变化,这段的实际走势和预测差距更大。它补的是个人模型容易漏掉的单指标突变和渐变。"
            )
        case .fused:
            return (
                "两个模型都觉得这段\(level)不寻常",
                "个人模型和通用基座都给出较高分位,说明它既不像你自己的常见节律,也不太符合多人基座学到的走势。"
            )
        case .rawGuard:
            return ("原始样本触发预警", "这条报告来自原始 HealthKit 样本,不依赖模型窗口。")
        }
    }

    private static func formatPercent(_ value: Double) -> String {
        "\(Int(value.rounded()))%"
    }

    private static func appendUnique(_ left: [String], _ right: [String]) -> [String] {
        var result = left
        for item in right where !result.contains(item) {
            result.append(item)
        }
        return result
    }
}

enum RawAlertDetector {
    static func reports(from events: [RawHealthEvent]) -> [HealthReport] {
        var reports: [HealthReport] = []
        for event in events {
            if let report = classify(event) {
                reports.append(report)
            }
        }
        reports.append(contentsOf: sleepDurationReports(from: events))
        return Fusion.mergeAdjacent(reports)
    }

    private static func classify(_ event: RawHealthEvent) -> HealthReport? {
        switch event.feature {
        case .heartRate:
            if event.value >= 190 {
                return raw(event, band: .critical, feature: "心率", value: "\(Int(event.value.rounded())) 次/分", headline: "心率出现高危峰值")
            }
            if event.value >= 160 {
                return raw(event, band: .elevated, feature: "心率", value: "\(Int(event.value.rounded())) 次/分", headline: "心率明显高于常规范围")
            }
        case .oxygen:
            if event.value <= 0.90 {
                return raw(event, band: .critical, feature: "血氧", value: String(format: "%.0f%%", event.value * 100), headline: "血氧出现高危低值")
            }
            if event.value <= 0.94 {
                return raw(event, band: .elevated, feature: "血氧", value: String(format: "%.0f%%", event.value * 100), headline: "血氧低于常规范围")
            }
        case .respiratoryRate:
            if event.value >= 36 {
                return raw(event, band: .critical, feature: "呼吸频率", value: "\(Int(event.value.rounded())) 次/分", headline: "呼吸频率出现高危峰值")
            }
            if event.value >= 28 {
                return raw(event, band: .elevated, feature: "呼吸频率", value: "\(Int(event.value.rounded())) 次/分", headline: "呼吸频率明显升高")
            }
        case .hrv:
            if event.value > 0 && event.value <= 10 {
                return raw(event, band: .watch, feature: "心率变异性", value: "\(Int(event.value.rounded())) ms", headline: "HRV 明显偏低")
            }
        default:
            return nil
        }
        return nil
    }

    private static func raw(_ event: RawHealthEvent, band: RiskBand, feature: String, value: String, headline: String) -> HealthReport {
        let end = event.end > event.start ? event.end : event.start.addingTimeInterval(60)
        return Fusion.rawReport(
            startedAt: event.start,
            endedAt: end,
            band: band,
            feature: feature,
            valueText: "\(feature): \(value)",
            headline: headline,
            detail: "\(feature) 的原始样本达到预警条件。短时间峰值不会再被 4 小时平均值掩盖。")
    }

    private static func sleepDurationReports(from events: [RawHealthEvent]) -> [HealthReport] {
        let calendar = Calendar.current
        let sleeps = events.filter { $0.feature == .sleep && $0.value > 0 && $0.end > $0.start }
        guard !sleeps.isEmpty else { return [] }
        var totals: [Date: TimeInterval] = [:]
        var starts: [Date: Date] = [:]
        var ends: [Date: Date] = [:]
        for event in sleeps {
            let key = calendar.startOfDay(for: event.end)
            totals[key, default: 0] += event.end.timeIntervalSince(event.start)
            starts[key] = min(starts[key] ?? event.start, event.start)
            ends[key] = max(ends[key] ?? event.end, event.end)
        }
        var reports: [HealthReport] = []
        for key in totals.keys.sorted() {
            guard let total = totals[key], total > 0, let start = starts[key], let end = ends[key] else { continue }
            let hours = total / 3600
            if hours < 3.0 {
                reports.append(Fusion.rawReport(
                    startedAt: start,
                    endedAt: end,
                    band: .elevated,
                    feature: "睡眠",
                    valueText: String(format: "当日睡眠 %.1f 小时", hours),
                    headline: "睡眠时长明显不足",
                    detail: "这一天记录到的睡眠时长很短,会作为作息异常段展示。"))
            } else if hours < 4.5 {
                reports.append(Fusion.rawReport(
                    startedAt: start,
                    endedAt: end,
                    band: .watch,
                    feature: "睡眠",
                    valueText: String(format: "当日睡眠 %.1f 小时", hours),
                    headline: "睡眠时长偏少",
                    detail: "这一天记录到的睡眠时长低于常规恢复需要,建议结合体感和近期作息看。"))
            }
        }
        return reports
    }
}
