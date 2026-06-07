import Foundation

enum MathUtils {
    static func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a,b).reduce(0) { $0 + $1.0 * $1.1 } }
    static func percentile(_ values: [Double], _ p: Double) -> Double { let s = values.sorted(); guard !s.isEmpty else { return 0 }; return s[Int(round(max(0.0, min(100.0, p))/100 * Double(s.count-1)))] }
    static func rank(sorted: [Double], value: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let lower = bound(sorted: sorted, value: value, includeEqual: false)
        let upper = bound(sorted: sorted, value: value, includeEqual: true)
        let midpoint = Double(lower) + Double(upper - lower) * 0.5
        return 100 * midpoint / Double(sorted.count)
    }
    private static func bound(sorted: [Double], value: Double, includeEqual: Bool) -> Int {
        var low = 0; var high = sorted.count
        while low < high {
            let middle = (low + high) / 2
            let before = includeEqual ? sorted[middle] <= value : sorted[middle] < value
            if before { low = middle + 1 } else { high = middle }
        }
        return low
    }
}

struct PersonalReconstructionModel: Codable, Sendable {
    let nFeatures: Int
    var weights: [[Double]]
    init(nFeatures: Int) { self.nFeatures = nFeatures; self.weights = Array(repeating: Array(repeating: 0, count: 3+nFeatures), count: nFeatures); for f in 0..<nFeatures { weights[f][1] = 0.35; weights[f][2] = 0.35 } }
    mutating func train(windows: [HealthWindow], epochs: Int = 10, lr: Double = 0.008, l2: Double = 0.0001) {
        for _ in 0..<epochs { for w in windows { guard w.values.count > 2 else { continue }; for t in 1..<(w.values.count-1) where w.mask[t] > 0 { for f in 0..<nFeatures { let input = input(w.values, t, f); let err = MathUtils.dot(weights[f], input) - w.values[t][f]; for i in weights[f].indices { weights[f][i] -= lr * (err * input[i] + l2 * weights[f][i]) } } } } }
    }
    func stepScores(window: HealthWindow) -> [Double] {
        var scores = Array(repeating: 0.0, count: window.values.count); guard window.values.count > 2 else { return scores }
        for t in 1..<(window.values.count-1) where window.mask[t] > 0 { var sum = 0.0; for f in 0..<nFeatures { let pred = MathUtils.dot(weights[f], input(window.values, t, f)); let e = pred - window.values[t][f]; sum += e*e }; scores[t] = sum / Double(nFeatures) }
        return scores
    }
    private func input(_ values: [[Double]], _ t: Int, _ feature: Int) -> [Double] { var x = Array(repeating: 0.0, count: 3+nFeatures); x[0]=1; x[1]=values[t-1][feature]; x[2]=values[t+1][feature]; for f in 0..<nFeatures { x[3+f] = f == feature ? 0 : values[t][f] }; return x }
}

struct PopulationPriorFile: Codable, Sendable { let priors: [PopulationPrior] }
struct PopulationPrior: Codable, Sendable { let name: String; let source: String; let featureCount: Int; let windowSize: Int; let horizon: Int; let lag: Int; var features: [FeaturePredictor] }
struct FeaturePredictor: Codable, Sendable { var intercept: Double; var autoreg: Double; var seasonal: Double; var cross: [Double] }

struct PopulationPriorStore {
    static func loadBundled() throws -> [PopulationPrior] { guard let url = Bundle.main.url(forResource: "PopulationPriors", withExtension: "json") else { throw AppError.populationPriorMissing }; let data = try Data(contentsOf: url); return try JSONDecoder().decode(PopulationPriorFile.self, from: data).priors }
    static func compatiblePriors(featureCount: Int, windowSize: Int, horizon: Int, lag: Int) throws -> [PopulationPrior] {
        let priors = try loadBundled()
        let compatible = priors.filter { $0.featureCount == featureCount && $0.windowSize == windowSize && $0.horizon == horizon && $0.lag == lag }
        if !compatible.isEmpty { return compatible }
        if let exact = priors.first(where: { $0.featureCount == featureCount }) { return [exact] }
        if let first = priors.first { return [first] }
        throw AppError.populationPriorMissing
    }
    static func bestPrior(featureCount: Int) throws -> PopulationPrior { try compatiblePriors(featureCount: featureCount, windowSize: 252, horizon: 24, lag: 6)[0] }
}

struct PopulationPredictionModel: Codable, Sendable { let prior: PopulationPrior; func stepScores(window: HealthWindow) -> [Double] { predictionScores(values: window.values, mask: window.mask, prior: prior) } }
struct PopulationPredictionEnsemble: Codable, Sendable {
    let priors: [PopulationPrior]
    func stepScores(window: HealthWindow) -> [Double] {
        var combined = Array(repeating: 0.0, count: window.values.count)
        for prior in priors {
            let scores = predictionScores(values: window.values, mask: window.mask, prior: prior)
            for i in scores.indices { combined[i] = max(combined[i], scores[i]) }
        }
        return combined
    }
}
struct PersonalPredictionModel: Codable, Sendable {
    var prior: PopulationPrior
    mutating func fineTune(windows: [HealthWindow], epochs: Int = 6, lr: Double = 0.004, l2: Double = 0.0001) {
        for _ in 0..<epochs { for w in windows { let start = max(prior.lag, w.values.count - prior.horizon); for t in start..<w.values.count where w.mask[t] > 0 { let prev = w.values[max(0,t-1)]; let seas = w.values[max(0,t-prior.lag)]; for f in 0..<prior.featureCount { var p = prior.features[f]; let err = predictFeature(values: w.values, t: t, feature: f, predictor: p, lag: prior.lag) - w.values[t][f]; p.intercept -= lr * err; p.autoreg -= lr * (err * prev[f] + l2*p.autoreg); p.seasonal -= lr * (err * seas[f] + l2*p.seasonal); for j in p.cross.indices where j != f { p.cross[j] -= lr * (err * prev[j] + l2*p.cross[j]) }; prior.features[f] = p } } } }
    }
    func stepScores(window: HealthWindow) -> [Double] { predictionScores(values: window.values, mask: window.mask, prior: prior) }
}

func predictionScores(values: [[Double]], mask: [Double], prior: PopulationPrior) -> [Double] {
    var scores = Array(repeating: 0.0, count: values.count); let start = max(prior.lag, values.count - prior.horizon)
    for t in start..<values.count where mask[t] > 0 { var sum = 0.0; for f in 0..<prior.featureCount { let e = predictFeature(values: values, t: t, feature: f, predictor: prior.features[f], lag: prior.lag) - values[t][f]; sum += e*e }; scores[t] = sum / Double(prior.featureCount) }
    return scores
}
func predictFeature(values: [[Double]], t: Int, feature: Int, predictor: FeaturePredictor, lag: Int) -> Double { let prev = values[max(0,t-1)]; let seas = values[max(0,t-lag)]; var y = predictor.intercept + predictor.autoreg * prev[feature] + predictor.seasonal * seas[feature]; for j in 0..<min(prev.count, predictor.cross.count) where j != feature { y += predictor.cross[j] * prev[j] }; return y }

struct PercentileCalibrator: Codable, Sendable { let sortedScores: [Double]; init(scores: [Double]) { sortedScores = scores.sorted() }; func percentile(score: Double) -> Double { MathUtils.rank(sorted: sortedScores, value: score) } }
struct ScoreSet: Codable, Sendable { let reconstruction: Double; let personalPrediction: Double; let populationPrediction: Double }
enum RiskLevel: String, Codable, Sendable { case normal, watch, elevated, critical }
struct AnomalyReport: Codable, Sendable, Identifiable { var id = UUID(); let startedAt: Date; let reconstructionPercentile: Double; let personalPredictionPercentile: Double; let populationPredictionPercentile: Double; let finalPercentile: Double; let level: RiskLevel; let reason: String }
struct FusionEngine: Codable, Sendable {
    let reconstruction: PercentileCalibrator; let personalPrediction: PercentileCalibrator; let populationPrediction: PercentileCalibrator
    func report(window: HealthWindow, scores: ScoreSet) -> AnomalyReport { let r = reconstruction.percentile(score: scores.reconstruction); let p = personalPrediction.percentile(score: scores.personalPrediction); let g = populationPrediction.percentile(score: scores.populationPrediction); let final = max(r, max(p, g * 0.7)); let level: RiskLevel = final >= 99.5 ? .critical : (final >= 99 ? .elevated : (final >= 97 ? .watch : .normal)); let reason = r >= 99 && p >= 99 ? "个人模式和预测趋势同时异常" : (r >= 99 ? "个人指标联动关系偏离历史模式" : (p >= 99 ? "未来趋势与个人历史预测不一致" : (g >= 99 ? "偏离多人先验模型" : "未达到强异常阈值"))); return AnomalyReport(startedAt: window.start, reconstructionPercentile: r, personalPredictionPercentile: p, populationPredictionPercentile: g, finalPercentile: final, level: level, reason: reason) }
}

struct UserModelBundle: Codable, Sendable { let createdAt: Date; let stats: NormalizationStats; let reconstruction: PersonalReconstructionModel; let populationPrior: PopulationPrior; let populationPriors: [PopulationPrior]; let personalPrediction: PersonalPredictionModel; let fusion: FusionEngine }

final class ModelTrainer {
    func train(dataset: PreparedHealthDataset) async throws -> UserModelBundle {
        guard let first = dataset.train.first, let firstRow = first.values.first else { throw AppError.notEnoughHistory }
        var recon = PersonalReconstructionModel(nFeatures: firstRow.count); recon.train(windows: dataset.train)
        let priors = try PopulationPriorStore.compatiblePriors(featureCount: firstRow.count, windowSize: first.values.count, horizon: 24, lag: 6)
        let prior = priors[0]
        var personal = PersonalPredictionModel(prior: prior); personal.fineTune(windows: dataset.train)
        let population = PopulationPredictionEnsemble(priors: priors)
        let reconScores = dataset.train.map { maxScore(recon.stepScores(window: $0)) }
        let personalScores = dataset.train.map { maxScore(personal.stepScores(window: $0)) }
        let populationScores = dataset.train.map { maxScore(population.stepScores(window: $0)) }
        let fusion = FusionEngine(reconstruction: PercentileCalibrator(scores: reconScores), personalPrediction: PercentileCalibrator(scores: personalScores), populationPrediction: PercentileCalibrator(scores: populationScores))
        return UserModelBundle(createdAt: Date(), stats: dataset.stats, reconstruction: recon, populationPrior: prior, populationPriors: priors, personalPrediction: personal, fusion: fusion)
    }
    func evaluateRecent(dataset: PreparedHealthDataset, bundle: UserModelBundle, limit: Int) -> [AnomalyReport] { let priors = bundle.populationPriors.isEmpty ? [bundle.populationPrior] : bundle.populationPriors; let population = PopulationPredictionEnsemble(priors: priors); return Array((dataset.validation + dataset.test).suffix(limit)).map { w in let scores = ScoreSet(reconstruction: maxScore(bundle.reconstruction.stepScores(window: w)), personalPrediction: maxScore(bundle.personalPrediction.stepScores(window: w)), populationPrediction: maxScore(population.stepScores(window: w))); return bundle.fusion.report(window: w, scores: scores) } }
}

func maxScore(_ scores: [Double]) -> Double { scores.max() ?? 0 }

