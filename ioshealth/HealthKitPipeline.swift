import Foundation
import HealthKit

enum HealthFeature: Int, Codable, CaseIterable, Identifiable, Sendable {
    case heartRate, steps, activeEnergy, hrv, oxygen, respiratoryRate, sleep, exercise
    var id: Int { rawValue }
    var aggregation: AggregationMethod {
        switch self {
        case .heartRate, .hrv, .oxygen, .respiratoryRate, .sleep: return .mean
        case .steps, .activeEnergy, .exercise: return .sum
        }
    }
}

enum AggregationMethod: String, Codable, Sendable { case mean, sum, max }

struct RawHealthEvent: Codable, Sendable {
    let feature: HealthFeature
    let start: Date
    let end: Date
    let value: Double
}

enum AppError: LocalizedError {
    case healthKitUnavailable, authorizationDenied, invalidDateRange, noHealthData, notEnoughHistory, populationPriorMissing
    var errorDescription: String? {
        switch self {
        case .healthKitUnavailable: return "此设备不支持 HealthKit"
        case .authorizationDenied: return "未获得 Apple 健康读取权限"
        case .invalidDateRange: return "历史日期范围无效"
        case .noHealthData: return "没有读取到可用健康数据"
        case .notEnoughHistory: return "历史数据不足，无法训练 42 天窗口模型"
        case .populationPriorMissing: return "缺少内置多人预测模型资产"
        }
    }
}

final class HealthKitDataSource {
    private let store = HKHealthStore()

    func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else { throw AppError.healthKitUnavailable }
        let readTypes = Set(HealthFeature.allCases.compactMap { $0.objectType })
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            store.requestAuthorization(toShare: Set<HKSampleType>(), read: readTypes) { success, error in
                if let error = error { continuation.resume(throwing: error) }
                else if success { continuation.resume() }
                else { continuation.resume(throwing: AppError.authorizationDenied) }
            }
        }
    }

    func loadHistory(daysBack: Int) async throws -> [RawHealthEvent] {
        let end = Date()
        guard let start = Calendar.current.date(byAdding: .day, value: -daysBack, to: end) else { throw AppError.invalidDateRange }
        var events: [RawHealthEvent] = []
        for feature in HealthFeature.allCases {
            events.append(contentsOf: try await query(feature: feature, start: start, end: end))
        }
        return events.sorted { $0.start < $1.start }
    }

    private func query(feature: HealthFeature, start: Date, end: Date) async throws -> [RawHealthEvent] {
        guard let sampleType = feature.sampleType else { return [] }
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate])
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[RawHealthEvent], Error>) in
            let query = HKSampleQuery(sampleType: sampleType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sort]) { _, samples, error in
                if let error = error { continuation.resume(throwing: error); return }
                let events = (samples ?? []).compactMap { sample -> RawHealthEvent? in
                    if let quantity = sample as? HKQuantitySample, let unit = feature.unit {
                        return RawHealthEvent(feature: feature, start: quantity.startDate, end: quantity.endDate, value: quantity.quantity.doubleValue(for: unit))
                    }
                    if let category = sample as? HKCategorySample, feature == .sleep {
                        let asleepValues: Set<Int> = [
                            HKCategoryValueSleepAnalysis.asleep.rawValue,
                            HKCategoryValueSleepAnalysis.asleepCore.rawValue,
                            HKCategoryValueSleepAnalysis.asleepDeep.rawValue,
                            HKCategoryValueSleepAnalysis.asleepREM.rawValue,
                            HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue
                        ]
                        let value = asleepValues.contains(category.value) ? 1.0 : 0.0
                        return RawHealthEvent(feature: feature, start: category.startDate, end: category.endDate, value: value)
                    }
                    return nil
                }
                continuation.resume(returning: events)
            }
            store.execute(query)
        }
    }
}

private extension HealthFeature {
    var objectType: HKObjectType? {
        switch self {
        case .heartRate: return HKObjectType.quantityType(forIdentifier: .heartRate)
        case .steps: return HKObjectType.quantityType(forIdentifier: .stepCount)
        case .activeEnergy: return HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)
        case .hrv: return HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)
        case .oxygen: return HKObjectType.quantityType(forIdentifier: .oxygenSaturation)
        case .respiratoryRate: return HKObjectType.quantityType(forIdentifier: .respiratoryRate)
        case .sleep: return HKObjectType.categoryType(forIdentifier: .sleepAnalysis)
        case .exercise: return HKObjectType.quantityType(forIdentifier: .appleExerciseTime)
        }
    }
    var sampleType: HKSampleType? {
        guard let objectType = objectType else { return nil }
        return objectType as? HKSampleType
    }
    var unit: HKUnit? {
        switch self {
        case .heartRate: return HKUnit.count().unitDivided(by: .minute())
        case .steps: return HKUnit.count()
        case .activeEnergy: return HKUnit.kilocalorie()
        case .hrv: return HKUnit.secondUnit(with: .milli)
        case .oxygen: return HKUnit.percent()
        case .respiratoryRate: return HKUnit.count().unitDivided(by: .minute())
        case .sleep: return nil
        case .exercise: return HKUnit.minute()
        }
    }
}



