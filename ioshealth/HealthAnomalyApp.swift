import SwiftUI
import HealthKit

@main
struct HealthAnomalyApp: App {
    @StateObject private var state = AppState()
    var body: some Scene {
        WindowGroup { ContentView().environmentObject(state) }
    }
}

@MainActor
final class AppState: ObservableObject {
    enum Phase: Equatable { case idle, importing, preprocessing, training, ready, failed(String) }
    @Published var phase: Phase = .idle
    @Published var status = "等待导入 Apple 健康历史数据"
    @Published var summary: DatasetSummary?
    @Published var bundle: UserModelBundle?
    @Published var reports: [AnomalyReport] = []

    private let health = HealthKitDataSource()
    private let storage = LocalModelStore()

    init() {
        if let saved = try? storage.load() {
            bundle = saved.bundle
            summary = saved.summary
            reports = saved.reports
            phase = .ready
            status = "已加载本地个性化模型"
        }
    }

    func importAndTrain(days: Int = 730) {
        Task {
            do {
                phase = .importing
                status = "请求 HealthKit 权限并读取历史数据"
                try await health.requestAuthorization()
                let events = try await health.loadHistory(daysBack: days)
                try await train(events: events, save: true)
            } catch {
                phase = .failed(error.localizedDescription)
                status = "失败：\(error.localizedDescription)"
            }
        }
    }

    func runDemo() {
        Task {
            do {
                status = "生成演示历史数据"
                let events = SyntheticHealthHistory.makeEvents(days: 240)
                try await train(events: events, save: false)
            } catch {
                phase = .failed(error.localizedDescription)
                status = "失败：\(error.localizedDescription)"
            }
        }
    }

    private func train(events: [RawHealthEvent], save: Bool) async throws {
        phase = .preprocessing
        status = "聚合 4 小时窗口、缺失处理、训练集标准化"
        let dataset = try await Task.detached(priority: .userInitiated) {
            try HealthPreprocessor().prepare(events: events)
        }.value
        summary = dataset.summary
        phase = .training
        status = "训练个人重建模型，微调多人预测模型"
        let output = try await Task.detached(priority: .userInitiated) {
            let trainer = ModelTrainer()
            let trained = try await trainer.train(dataset: dataset)
            let recentReports = trainer.evaluateRecent(dataset: dataset, bundle: trained, limit: 30)
            if save { try LocalModelStore().save(bundle: trained, summary: dataset.summary, reports: recentReports) }
            return TrainingOutput(bundle: trained, summary: dataset.summary, reports: recentReports)
        }.value
        bundle = output.bundle
        summary = output.summary
        reports = output.reports
        phase = .ready
        status = "模型已就绪"
    }
}

private struct TrainingOutput: Sendable {
    let bundle: UserModelBundle
    let summary: DatasetSummary
    let reports: [AnomalyReport]
}

struct ContentView: View {
    @EnvironmentObject private var state: AppState
    var body: some View {
        NavigationStack {
            Group {
                if case .ready = state.phase { DashboardView() } else { OnboardingView() }
            }
            .navigationTitle("Health Anomaly")
        }
    }
}

struct OnboardingView: View {
    @EnvironmentObject private var state: AppState
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("个人健康模式检测").font(.largeTitle.bold())
                Text("首次启动读取 Apple 健康历史数据，在本机训练个性化模型。多人预测先验随 App 内置，用户数据不上传服务器。")
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 10) {
                    Label("个人重建模型：学习你的指标联动关系", systemImage: "waveform.path.ecg")
                    Label("个人预测模型：从多人先验微调，补充渐变和趋势异常", systemImage: "point.3.connected.trianglepath.dotted")
                    Label("分位数融合：不同模型统一成风险等级", systemImage: "gauge.with.dots.needle.50percent")
                }
                Button { state.importAndTrain() } label: {
                    Label("导入 Apple 健康并训练", systemImage: "heart.text.square.fill").frame(maxWidth: .infinity)
                }.buttonStyle(.borderedProminent)
                Button { state.runDemo() } label: {
                    Label("无健康数据时运行演示", systemImage: "play.circle").frame(maxWidth: .infinity)
                }.buttonStyle(.bordered)
                Text(state.status).font(.footnote).foregroundStyle(.secondary)
                if case .failed(let message) = state.phase {
                    Text(message).font(.footnote).foregroundStyle(.red)
                }
            }.padding()
        }
    }
}

struct DashboardView: View {
    @EnvironmentObject private var state: AppState
    var body: some View {
        List {
            Section("模型状态") { ModelStatusView(summary: state.summary, bundle: state.bundle) }
            Section("最近窗口") {
                if state.reports.isEmpty { Text("暂无检测结果").foregroundStyle(.secondary) }
                ForEach(state.reports) { report in AnomalyReportRow(report: report) }
            }
        }
        .refreshable { state.importAndTrain() }
    }
}

struct ModelStatusView: View {
    let summary: DatasetSummary?
    let bundle: UserModelBundle?
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let summary = summary {
                LabeledContent("历史范围", value: "\(summary.firstDate.formatted(date: .abbreviated, time: .omitted)) - \(summary.lastDate.formatted(date: .abbreviated, time: .omitted))")
                LabeledContent("4h 数据块", value: "\(summary.bucketCount)")
                LabeledContent("训练窗口", value: "\(summary.trainWindows)")
                LabeledContent("有效率", value: "\(Int(summary.validRatio * 100))%")
            }
            if let bundle = bundle {
                LabeledContent("个人重建", value: "\(bundle.reconstruction.nFeatures) 维")
                LabeledContent("预测先验", value: bundle.populationPrior.name)
                LabeledContent("模型创建", value: bundle.createdAt.formatted(date: .abbreviated, time: .shortened))
            }
        }
    }
}

struct AnomalyReportRow: View {
    let report: AnomalyReport
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(report.startedAt, style: .date)
                Text(report.startedAt, style: .time)
                Spacer()
                Text(report.level.title).font(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 4).background(report.level.color.opacity(0.16)).clipShape(Capsule())
            }
            Text(report.reason).font(.subheadline)
            HStack(spacing: 12) {
                Text("重建 P\(Int(report.reconstructionPercentile))")
                Text("预测 P\(Int(report.personalPredictionPercentile))")
                Text("通用 P\(Int(report.populationPredictionPercentile))")
            }.font(.caption).foregroundStyle(.secondary)
        }.padding(.vertical, 4)
    }
}

private extension RiskLevel {
    var title: String {
        switch self { case .normal: return "正常"; case .watch: return "观察"; case .elevated: return "异常"; case .critical: return "强异常" }
    }
    var color: Color {
        switch self { case .normal: return .green; case .watch: return .yellow; case .elevated: return .orange; case .critical: return .red }
    }
}
