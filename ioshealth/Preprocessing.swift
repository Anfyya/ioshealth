import Foundation

struct HealthWindow: Codable, Sendable, Identifiable {
    var id = UUID()
    let start: Date
    let values: [[Double]]
    let mask: [Double]
    let featureMask: [[Double]]

    func bucketStart(at index: Int, bucketHours: Int) -> Date {
        let safeIndex = max(0, min(index, max(0, values.count - 1)))
        return Calendar.current.date(byAdding: .hour, value: safeIndex * bucketHours, to: start) ?? start
    }
}

struct NormalizationStats: Codable, Sendable { let mean: [Double]; let std: [Double] }
struct DatasetSummary: Codable, Sendable {
    let firstDate: Date; let lastDate: Date; let bucketCount: Int
    let trainWindows: Int; let validationWindows: Int; let testWindows: Int; let validRatio: Double
    let bucketHours: Int
}
struct PreparedHealthDataset: Codable, Sendable {
    let train: [HealthWindow]; let validation: [HealthWindow]; let test: [HealthWindow]
    let stats: NormalizationStats; let summary: DatasetSummary
}

final class HealthPreprocessor {
    let bucketHours = 4
    let windowSize = 252
    let stride = 6
    let missingThresholdBuckets = 2

    func prepare(events: [RawHealthEvent]) throws -> PreparedHealthDataset {
        guard !events.isEmpty else { throw AppError.noHealthData }
        let buckets = makeBuckets(events: events)
        guard buckets.count >= windowSize + stride else { throw AppError.notEnoughHistory }
        let rowMask = buildRowMask(buckets: buckets)
        let filled = impute(values: buckets.map { $0.values })
        let trainBucketEnd = max(windowSize, Int(Double(filled.count) * 0.70))
        let stats = computeStats(values: filled, rowMask: rowMask, upperBound: trainBucketEnd)
        let normalized = filled.map { row in row.enumerated().map { ($0.element - stats.mean[$0.offset]) / stats.std[$0.offset] } }
        let windows = makeWindows(values: normalized, rowMask: rowMask, featureMask: buckets.map { $0.featureMask }, starts: buckets.map { $0.start })
        guard windows.count >= 3 else { throw AppError.notEnoughHistory }
        let trainEnd = Int(Double(windows.count) * 0.70)
        let valEnd = Int(Double(windows.count) * 0.85)
        let summary = DatasetSummary(firstDate: buckets.first!.start, lastDate: buckets.last!.start, bucketCount: buckets.count, trainWindows: trainEnd, validationWindows: max(0, valEnd - trainEnd), testWindows: max(0, windows.count - valEnd), validRatio: rowMask.reduce(0.0,+) / Double(rowMask.count), bucketHours: bucketHours)
        return PreparedHealthDataset(train: Array(windows[..<trainEnd]), validation: Array(windows[trainEnd..<valEnd]), test: Array(windows[valEnd..<windows.count]), stats: stats, summary: summary)
    }

    private struct Bucket { var start: Date; var sums: [Double]; var counts: [Double]; var maxes: [Double]; var featureMask: [Double]
        var values: [Double] { HealthFeature.allCases.map { feature in
            let i = feature.rawValue
            switch feature.aggregation {
            case .mean: return counts[i] > 0 ? sums[i] / counts[i] : .nan
            case .sum: return counts[i] > 0 ? sums[i] : .nan
            case .max: return counts[i] > 0 ? maxes[i] : .nan
            }
        } }
    }

    private func makeBuckets(events: [RawHealthEvent]) -> [Bucket] {
        let calendar = Calendar.current
        let sorted = events.sorted { $0.start < $1.start }
        let first = floorDate(sorted.first!.start, calendar: calendar)
        let lastEventDate = sorted.map { $0.end > $0.start ? $0.end : $0.start }.max() ?? sorted.last!.start
        let last = floorDate(lastEventDate, calendar: calendar)
        let bucketSeconds = Double(bucketHours * 3600)
        let count = max(1, Int(last.timeIntervalSince(first) / Double(bucketHours * 3600)) + 1)
        var buckets = (0..<count).map { idx -> Bucket in
            let start = calendar.date(byAdding: .hour, value: idx * bucketHours, to: first) ?? first
            return Bucket(start: start, sums: Array(repeating: 0, count: 8), counts: Array(repeating: 0, count: 8), maxes: Array(repeating: -Double.infinity, count: 8), featureMask: Array(repeating: 0, count: 8))
        }
        for event in sorted {
            let f = event.feature.rawValue
            let duration = max(1.0, event.end.timeIntervalSince(event.start))
            let effectiveEnd = event.end > event.start ? event.end.addingTimeInterval(-0.001) : event.start
            let startIndex = Int(floorDate(event.start, calendar: calendar).timeIntervalSince(first) / bucketSeconds)
            let endIndex = Int(floorDate(effectiveEnd, calendar: calendar).timeIntervalSince(first) / bucketSeconds)
            let lower = max(0, startIndex)
            let upper = min(buckets.count - 1, endIndex)
            guard lower <= upper else { continue }

            for idx in lower...upper {
                let bucketStart = buckets[idx].start
                let bucketEnd = calendar.date(byAdding: .hour, value: bucketHours, to: bucketStart) ?? bucketStart.addingTimeInterval(bucketSeconds)
                let overlapStart = event.start > bucketStart ? event.start : bucketStart
                let overlapEnd = event.end < bucketEnd ? event.end : bucketEnd
                let overlap = max(0.0, overlapEnd.timeIntervalSince(overlapStart))
                guard overlap > 0 || event.end <= event.start else { continue }
                let weight = event.end > event.start ? overlap : 1.0
                switch event.feature.aggregation {
                case .mean:
                    buckets[idx].sums[f] += event.value * weight
                    buckets[idx].counts[f] += weight
                case .sum:
                    buckets[idx].sums[f] += event.value * (weight / duration)
                    buckets[idx].counts[f] += 1
                case .max:
                    buckets[idx].maxes[f] = max(buckets[idx].maxes[f], event.value)
                    buckets[idx].counts[f] += 1
                }
                buckets[idx].featureMask[f] = 1
            }
        }
        return buckets
    }

    private func floorDate(_ date: Date, calendar: Calendar) -> Date {
        let c = calendar.dateComponents([.year,.month,.day,.hour], from: date)
        let hour = ((c.hour ?? 0) / bucketHours) * bucketHours
        return calendar.date(from: DateComponents(year: c.year, month: c.month, day: c.day, hour: hour)) ?? date
    }

    private func buildRowMask(buckets: [Bucket]) -> [Double] {
        var mask = buckets.map { $0.featureMask[HealthFeature.heartRate.rawValue] > 0 ? 1.0 : ($0.featureMask.reduce(0,+) > 0 ? 1.0 : 0.0) }
        var i = 0
        while i < mask.count {
            if mask[i] > 0 { i += 1; continue }
            let start = i
            while i < mask.count && mask[i] == 0 { i += 1 }
            if i - start < missingThresholdBuckets { for j in start..<i { mask[j] = 1 } }
        }
        return mask
    }

    private func impute(values: [[Double]]) -> [[Double]] {
        var out = values; guard let first = out.first else { return out }
        for f in 0..<first.count {
            var last: Double?
            for i in out.indices { if out[i][f].isFinite { last = out[i][f] } else if let last = last { out[i][f] = last } }
            var next: Double?
            for i in out.indices.reversed() { if out[i][f].isFinite { next = out[i][f] } else if let next = next { out[i][f] = next } }
            for i in out.indices where !out[i][f].isFinite { out[i][f] = 0 }
        }
        return out
    }

    private func computeStats(values: [[Double]], rowMask: [Double], upperBound: Int) -> NormalizationStats {
        let features = values[0].count; let end = min(values.count, upperBound)
        var mean = Array(repeating: 0.0, count: features); var count = Array(repeating: 0.0, count: features)
        for i in 0..<end where rowMask[i] > 0 { for f in 0..<features { mean[f] += values[i][f]; count[f] += 1 } }
        for f in 0..<features { mean[f] /= max(1.0, count[f]) }
        var variance = Array(repeating: 0.0, count: features)
        for i in 0..<end where rowMask[i] > 0 { for f in 0..<features { let d = values[i][f] - mean[f]; variance[f] += d*d } }
        let std = variance.enumerated().map { item in max(1e-6, sqrt(item.element / max(1.0, count[item.offset]))) }
        return NormalizationStats(mean: mean, std: std)
    }

    private func makeWindows(values: [[Double]], rowMask: [Double], featureMask: [[Double]], starts: [Date]) -> [HealthWindow] {
        var windows: [HealthWindow] = []; var start = 0
        while start + windowSize <= values.count {
            let end = start + windowSize
            windows.append(HealthWindow(start: starts[start], values: Array(values[start..<end]), mask: Array(rowMask[start..<end]), featureMask: Array(featureMask[start..<end])))
            start += stride
        }
        return windows
    }
}

enum SyntheticHealthHistory {
    static func makeEvents(days: Int) -> [RawHealthEvent] {
        var events: [RawHealthEvent] = []; let calendar = Calendar.current
        let start = calendar.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        for block in 0..<(days * 6) {
            guard let date = calendar.date(byAdding: .hour, value: block * 4, to: start) else { continue }
            let phase = Double(block % 6) / 6 * 2 * Double.pi; let week = sin(Double(block) / 42 * 2 * Double.pi)
            let activity = max(0, 0.7 * sin(phase) + 0.25 * week + 0.35); let sleep = (block % 6) <= 1 ? 1.0 : 0.0
            let values: [(HealthFeature, Double)] = [(.heartRate, 68 + 18*activity - 8*sleep + Double.random(in: -4...4)), (.steps, max(0, 2200*activity + Double.random(in: -120...120))), (.activeEnergy, max(0, 110*activity + Double.random(in: -8...8))), (.hrv, 45 - 8*activity + 10*sleep + Double.random(in: -4...4)), (.oxygen, 0.97 + Double.random(in: -0.01...0.01)), (.respiratoryRate, 14 + 2*activity + Double.random(in: -0.8...0.8)), (.sleep, sleep), (.exercise, max(0, 25*activity + Double.random(in: -3...3)))]
            let end = calendar.date(byAdding: .hour, value: 4, to: date) ?? date
            for item in values { events.append(RawHealthEvent(feature: item.0, start: date, end: end, value: item.1)) }
        }
        if let spikeStart = calendar.date(byAdding: .day, value: -10, to: Date()),
           let spikeEnd = calendar.date(byAdding: .minute, value: 5, to: spikeStart) {
            events.append(RawHealthEvent(feature: .heartRate, start: spikeStart, end: spikeEnd, value: 205))
        }
        return events
    }
}


