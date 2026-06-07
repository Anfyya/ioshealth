import Foundation

// MARK: - Fusion of the two model paths into a plain-language report
//
// Two complementary signals (see the project benchmarks):
//   - reconstruction (personal): correlation-break / joint / burst anomalies
//   - prediction (multi-person base): single-variable / gradual / trend deviations
// Each raw score is turned into a personal percentile (vs the user's own training
// distribution), fused (max), thresholded into a risk band, and explained.

/// Plain display names for the 8 features, in HealthFeature order.
let featureDisplayNames = [
    "心率", "步数", "活动消耗", "心率变异性", "血氧", "呼吸频率", "睡眠", "运动时长",
]

enum RiskBand: String, Codable, Hashable, Sendable {
    case normal, watch, elevated, critical
}

/// One analyzed 4-hour window, ready for display.
struct HealthReport: Identifiable, Codable, Hashable, Sendable {
    var id = UUID()
    let startedAt: Date
    let reconPercentile: Double   // personal "pattern" signal
    let predPercentile: Double    // trend signal
    let finalPercentile: Double
    let band: RiskBand
    let topFeature: String        // plain name of the most-off metric (or "")
    let headline: String          // plain one-liner
    let detail: String            // plain explanation
    let featureErrors: [Double]   // per-feature reconstruction error (len 8)
}

/// Maps a raw score to a personal percentile against the training distribution.
struct Calibrator: Codable, Hashable, Sendable {
    let sorted: [Double]
    init(_ scores: [Double]) { sorted = scores.sorted() }

    /// Percentile (0–100): fraction of training scores <= value.
    func percentile(_ v: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        var lo = 0, hi = sorted.count
        while lo < hi {
            let m = (lo + hi) / 2
            if sorted[m] <= v { lo = m + 1 } else { hi = m }
        }
        return 100.0 * Double(lo) / Double(sorted.count)
    }
}

enum Fusion {
    static func band(forFinal final: Double) -> RiskBand {
        if final >= 99.5 { return .critical }
        if final >= 99 { return .elevated }
        if final >= 97 { return .watch }
        return .normal
    }

    static func report(
        startedAt: Date,
        reconScore: Double, predScore: Double,
        reconCal: Calibrator, predCal: Calibrator,
        featureErrors: [Double]
    ) -> HealthReport {
        let r = reconCal.percentile(reconScore)
        let p = predCal.percentile(predScore)
        let final = max(r, p)
        let band = band(forFinal: final)

        let topIdx = featureErrors.indices.max(by: { featureErrors[$0] < featureErrors[$1] })
        let topName = (band != .normal && topIdx != nil && topIdx! < featureDisplayNames.count)
            ? featureDisplayNames[topIdx!] : ""

        let (headline, detail) = phrasing(band: band, reconDominant: r >= p, topName: topName)

        return HealthReport(
            startedAt: startedAt,
            reconPercentile: r, predPercentile: p, finalPercentile: final,
            band: band, topFeature: topName, headline: headline, detail: detail,
            featureErrors: featureErrors)
    }

    private static func phrasing(band: RiskBand, reconDominant: Bool, topName: String) -> (String, String) {
        if band == .normal {
            return ("一切看起来正常",
                    "最近这段时间,你的各项身体信号都在平时的范围内,没有发现值得注意的异常。")
        }
        let level = band == .critical ? "明显" : (band == .elevated ? "比较明显" : "轻微")
        if reconDominant {
            let who = topName.isEmpty ? "某些指标" : "「\(topName)」"
            return ("\(who)和其它指标的配合\(level)反常",
                    "你的\(who)和平时与其它身体指标之间的联动关系不太一样了。单看某一项也许并不极端,但几项组合在一起\(level)偏离了你自己的规律,建议留意。")
        } else {
            let who = topName.isEmpty ? "身体信号" : "「\(topName)」"
            return ("\(who)的变化趋势\(level)反常",
                    "你的\(who)最近的走势\(level)偏离了你以往的规律。如果持续出现,建议关注近期的作息、压力或身体状态。")
        }
    }
}
