import SwiftUI
import HealthKit

// MARK: - App entry

@main
struct HealthAnomalyApp: App {
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(state)
        }
    }
}

// MARK: - App state / pipeline

@MainActor
final class AppState: ObservableObject {
    enum Phase: Equatable { case idle, importing, preprocessing, training, ready, failed(String) }

    @Published var phase: Phase = .idle
    @Published var status = "等待读取你的 Apple 健康数据"
    @Published var summary: DatasetSummary?
    @Published var reports: [HealthReport] = []
    @Published var lastAnalyzed: Date?
    @Published var selectedRange: HealthDataRange = .all
    @Published var trainingEpochs: Int?
    @Published var evaluatedWindows: Int?
    @Published var rawGuardCount: Int?

    private let health = HealthKitDataSource()

    private struct AnalysisResult: Sendable {
        let summary: DatasetSummary
        let reports: [HealthReport]
        let trainingEpochs: Int
        let evaluatedWindows: Int
        let rawGuardCount: Int
    }

    init() {
        if let saved = ATStore.load() {
            summary = saved.summary
            reports = saved.reports
            lastAnalyzed = saved.createdAt
            selectedRange = saved.selectedRange ?? .all
            trainingEpochs = saved.trainingEpochs
            evaluatedWindows = saved.evaluatedWindows
            rawGuardCount = saved.rawGuardCount
            phase = .ready
            status = "已载入上次的分析结果"
        }
    }

    var flaggedCount: Int { reports.count }
    var latest: HealthReport? { reports.sorted { $0.startedAt < $1.startedAt }.last }
    var isBusy: Bool {
        switch phase {
        case .importing, .preprocessing, .training: return true
        default: return false
        }
    }

    func importAndAnalyze(range: HealthDataRange? = nil) {
        let requestedRange = range ?? selectedRange
        selectedRange = requestedRange
        Task {
            do {
                phase = .importing
                status = "正在请求权限并读取\(requestedRange.title)健康数据..."
                try await health.requestAuthorization()
                let events = try await health.loadHistory(range: requestedRange)
                try await analyze(events: events, save: true, range: requestedRange)
            } catch {
                phase = .failed(error.localizedDescription)
                status = "出错了: \(error.localizedDescription)"
            }
        }
    }

    func runDemo() {
        Task {
            do {
                selectedRange = .sevenMonths
                status = "正在生成带异常的示例数据..."
                let events = SyntheticHealthHistory.makeEvents(days: 240)
                try await analyze(events: events, save: false, range: .sevenMonths)
            } catch {
                phase = .failed(error.localizedDescription)
                status = "出错了: \(error.localizedDescription)"
            }
        }
    }

    private func analyze(events: [RawHealthEvent], save: Bool, range: HealthDataRange) async throws {
        phase = .preprocessing
        status = "正在整理数据: 模型用 4 小时桶, 原始哨兵保留样本级时间..."
        let dataset = try await Task.detached(priority: .userInitiated) {
            try HealthPreprocessor().prepare(events: events)
        }.value
        summary = dataset.summary

        phase = .training
        let plannedEpochs = TrainingPlanner.epochs(for: dataset.train.count)
        status = "正在端侧训练个人模型 \(plannedEpochs) 轮, 并加载通用基座..."
        let result = try await Task.detached(priority: .userInitiated) { () -> AnalysisResult in
            let engine = try ATEngine()
            engine.trainRecon(dataset.train, epochs: plannedEpochs, lr: 3e-4, batchSize: 16)

            let reconCal = Calibrator(engine.reconScores(dataset.train))
            let predCal = Calibrator(engine.predScores(dataset.train))
            let evaluation = dataset.train + dataset.validation + dataset.test
            let reconDetails = engine.reconScoreDetails(evaluation)
            let predDetails = engine.predScoreDetails(evaluation)
            var modelReports: [HealthReport] = []

            for (index, window) in evaluation.enumerated() {
                let recon = index < reconDetails.count ? reconDetails[index] : ATScoreDetail(score: 0, stepIndex: 0)
                let pred = index < predDetails.count ? predDetails[index] : ATScoreDetail(score: 0, stepIndex: 0)
                let reconPercentile = reconCal.percentile(recon.score)
                let predPercentile = predCal.percentile(pred.score)
                let useReconStep = reconPercentile >= predPercentile
                let modelStep = useReconStep ? recon.stepIndex : engine.cfg.winSize - engine.cfg.horizon + pred.stepIndex
                let clampedStep = max(0, min(modelStep, engine.cfg.winSize - 1))
                let start = window.bucketStart(at: clampedStep, bucketHours: dataset.summary.bucketHours)
                let end = Calendar.current.date(byAdding: .hour, value: dataset.summary.bucketHours, to: start) ?? start.addingTimeInterval(Double(dataset.summary.bucketHours * 3600))
                let featureErrors = engine.reconFeatureError(window, stepIndex: clampedStep)
                let report = Fusion.modelReport(
                    startedAt: start,
                    endedAt: end,
                    reconScore: recon.score,
                    predScore: pred.score,
                    reconCal: reconCal,
                    predCal: predCal,
                    featureErrors: featureErrors)
                if report.band != .normal {
                    modelReports.append(report)
                }
            }

            let rawReports = RawAlertDetector.reports(from: events)
            let merged = Fusion.mergeAdjacent(rawReports + modelReports)
            return AnalysisResult(
                summary: dataset.summary,
                reports: merged.sorted { $0.startedAt < $1.startedAt },
                trainingEpochs: plannedEpochs,
                evaluatedWindows: evaluation.count,
                rawGuardCount: rawReports.count)
        }.value

        summary = result.summary
        reports = result.reports
        trainingEpochs = result.trainingEpochs
        evaluatedWindows = result.evaluatedWindows
        rawGuardCount = result.rawGuardCount
        lastAnalyzed = Date()
        if save {
            ATStore.save(PersistedState(
                createdAt: Date(),
                summary: result.summary,
                reports: result.reports,
                selectedRange: range,
                trainingEpochs: result.trainingEpochs,
                evaluatedWindows: result.evaluatedWindows,
                rawGuardCount: result.rawGuardCount))
        }
        phase = .ready
        status = result.reports.isEmpty
            ? "分析完成: 没有达到预警线的异常时间段"
            : "分析完成: 找到 \(result.reports.count) 段需要留意的时间"
    }
}

enum TrainingPlanner {
    static func epochs(for trainWindows: Int) -> Int {
        if trainWindows >= 1400 { return 64 }
        if trainWindows >= 800 { return 48 }
        if trainWindows >= 320 { return 36 }
        return 24
    }
}

// MARK: - Design tokens

enum HG {
    static let accent = Color(red: 1.0, green: 0.216, blue: 0.373)
    static let blue = Color(red: 0.039, green: 0.518, blue: 1.0)
    static let purple = Color(red: 0.749, green: 0.353, blue: 0.949)
    static let watchThreshold = Fusion.watchThreshold
}

extension RiskBand {
    var title: String {
        switch self {
        case .normal: return "未达到预警线"
        case .watch: return "观察"
        case .elevated: return "异常"
        case .critical: return "高危"
        }
    }

    var tint: Color {
        switch self {
        case .normal: return .green
        case .watch: return .yellow
        case .elevated: return .orange
        case .critical: return .red
        }
    }

    var symbol: String {
        switch self {
        case .normal: return "checkmark.seal.fill"
        case .watch: return "eye.fill"
        case .elevated: return "exclamationmark.triangle.fill"
        case .critical: return "exclamationmark.octagon.fill"
        }
    }
}

extension ReportSource {
    var symbol: String {
        switch self {
        case .rawGuard: return "waveform.path.ecg.rectangle"
        case .personalModel: return "person.crop.circle.badge.exclamationmark"
        case .predictionBase: return "point.3.connected.trianglepath.dotted"
        case .fused: return "sparkles"
        }
    }

    var tint: Color {
        switch self {
        case .rawGuard: return HG.accent
        case .personalModel: return HG.purple
        case .predictionBase: return HG.blue
        case .fused: return .orange
        }
    }
}

// MARK: - Root

struct RootView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        Group {
            if state.summary != nil || !state.reports.isEmpty || state.phase == .ready {
                MainTabView()
            } else {
                OnboardingView()
            }
        }
        .background(AppBackground().ignoresSafeArea())
    }
}

struct AppBackground: View {
    var body: some View {
        LinearGradient(
            colors: [Color(.systemBackground), Color(.secondarySystemBackground)],
            startPoint: .top,
            endPoint: .bottom)
    }
}

// MARK: - Onboarding

struct OnboardingView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(LinearGradient(colors: [Color(red: 1, green: 0.357, blue: 0.498), HG.accent],
                                         startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 74, height: 74)
                    .overlay(Image(systemName: "heart.text.square.fill").font(.system(size: 34)).foregroundStyle(.white))
                    .shadow(color: HG.accent.opacity(0.34), radius: 18, y: 10)

                VStack(alignment: .leading, spacing: 10) {
                    Text("读取完整健康历史")
                        .font(.system(size: 34, weight: .bold))
                    Text("先按你选择的范围导入 Apple 健康数据,再在本机训练个人时序 Transformer,并用内置通用基座做第二路预测检查。")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                GlassEffectContainer(spacing: 12) {
                    VStack(spacing: 12) {
                        FeatureRow(icon: "clock.arrow.circlepath", tint: HG.blue,
                                   title: "不再只看一年", subtitle: "默认读取全部历史,也可以手动切换 5 年、3 年或 7 个月。")
                        FeatureRow(icon: "waveform.path.ecg", tint: HG.accent,
                                   title: "短时异常单独抓", subtitle: "心率、血氧等原始样本先过哨兵规则,不会被 4 小时平均掉。")
                        FeatureRow(icon: "point.3.connected.trianglepath.dotted", tint: HG.purple,
                                   title: "双模型同时看", subtitle: "个人模型看你自己的规律,通用基座看跨人的走势偏离。")
                    }
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("导入范围")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    RangeSelector(selection: $state.selectedRange)
                }

                VStack(spacing: 12) {
                    Button { state.importAndAnalyze(range: state.selectedRange) } label: {
                        Label("读取\(state.selectedRange.title)数据并分析", systemImage: "heart.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glassProminent)
                    .tint(HG.accent)
                    .controlSize(.large)
                    .disabled(state.isBusy)

                    Button { state.runDemo() } label: {
                        Label("用示例数据体验", systemImage: "play.circle")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                    .disabled(state.isBusy)
                }

                StatusLine(status: state.status, busy: state.isBusy)
                if case .failed(let message) = state.phase {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
            .padding(20)
            .padding(.bottom, 40)
        }
    }
}

struct FeatureRow: View {
    let icon: String
    let tint: Color
    let title: String
    let subtitle: String

    var body: some View {
        HStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 13, style: .continuous)
                .fill(.quaternary)
                .frame(width: 44, height: 44)
                .overlay(Image(systemName: icon).font(.system(size: 19, weight: .medium)).foregroundStyle(tint))
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.body.weight(.semibold))
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .glassEffect(.regular, in: .rect(cornerRadius: 22))
    }
}

struct RangeSelector: View {
    @Binding var selection: HealthDataRange

    var body: some View {
        HStack(spacing: 8) {
            ForEach(HealthDataRange.allCases) { range in
                Button { selection = range } label: {
                    Text(range.title)
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
                .foregroundStyle(selection == range ? .white : .primary)
                .background(selection == range ? HG.accent : Color.secondary.opacity(0.12), in: .rect(cornerRadius: 14))
            }
        }
        .accessibilityLabel("导入范围")
    }
}

struct StatusLine: View {
    let status: String
    let busy: Bool

    var body: some View {
        HStack(spacing: 8) {
            if busy {
                ProgressView().controlSize(.small)
            } else {
                Circle().fill(.green).frame(width: 8, height: 8)
            }
            Text(status)
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
    }
}

// MARK: - Tabs

struct MainTabView: View {
    var body: some View {
        TabView {
            DashboardView()
                .tabItem { Label("概览", systemImage: "heart.text.square") }
            AlertsView()
                .tabItem { Label("异常", systemImage: "exclamationmark.triangle") }
            DataModelView()
                .tabItem { Label("数据", systemImage: "cylinder.split.1x2") }
        }
    }
}

// MARK: - Dashboard

struct DashboardView: View {
    @EnvironmentObject private var state: AppState
    @State private var selected: HealthReport?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    StatusHeroCard(latest: state.latest, flagged: state.flaggedCount, totalWindows: state.evaluatedWindows)
                    if state.reports.count > 1 {
                        TrendCard(reports: Array(state.reports.suffix(24)))
                    }
                    AlertPreviewSection(reports: state.reports, onSelect: { selected = $0 })
                    DataInfoCard(summary: state.summary, lastAnalyzed: state.lastAnalyzed, trainingEpochs: state.trainingEpochs, evaluatedWindows: state.evaluatedWindows)
                }
                .padding(20)
                .padding(.bottom, 30)
            }
            .navigationTitle("身体节律")
            .scrollEdgeEffectStyle(.soft, for: .top)
            .refreshable { state.importAndAnalyze(range: state.selectedRange) }
            .navigationDestination(item: $selected) { report in
                ReportDetailView(report: report)
            }
        }
    }
}

struct StatusHeroCard: View {
    let latest: HealthReport?
    let flagged: Int
    let totalWindows: Int?

    private var band: RiskBand { latest?.band ?? .normal }

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle().fill(band.tint.opacity(0.16)).frame(width: 116, height: 116)
                Image(systemName: band.symbol)
                    .font(.system(size: 52, weight: .bold))
                    .foregroundStyle(band.tint)
            }
            .padding(.top, 4)

            Text(title)
                .font(.title3.weight(.bold))
                .multilineTextAlignment(.center)

            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            if let latestReport = latest {
                HStack(spacing: 8) {
                    Image(systemName: latestReport.source.symbol)
                    Text(latestReport.source.title)
                    Text(latestReport.durationText)
                }
                .font(.footnote.weight(.semibold))
                .foregroundStyle(latest.source.tint)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(latest.source.tint.opacity(0.12), in: .rect(cornerRadius: 12))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(24)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }

    private var title: String {
        guard let latest else { return "没有达到预警线的异常段" }
        return latest.headline
    }

    private var subtitle: String {
        if latest == nil {
            let countText = totalWindows.map { "模型评估了 \($0) 个历史窗口" } ?? "模型已完成评估"
            return "\(countText)。这不是医学结论,只是当前阈值下没有需要单独列出的时间段。"
        }
        return "共找到 \(flagged) 段需要留意的时间,最近一段为 \(latest?.band.title ?? "")。"
    }
}

struct TrendCard: View {
    let reports: [HealthReport]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("异常强度走势").font(.headline)
                Spacer()
                Text("只显示触发预警的时间段")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            TrendChart(values: reports.map(\.finalPercentile), threshold: Fusion.elevatedThreshold)
                .frame(height: 86)
        }
        .padding(20)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}

struct TrendChart: View {
    let values: [Double]
    var threshold: Double
    private let low = 80.0
    private let high = 100.0

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let height = geo.size.height
            let pts = points(in: CGSize(width: width, height: height))
            let thresholdY = height - CGFloat((threshold - low) / (high - low)) * height
            ZStack {
                Path { path in
                    path.move(to: CGPoint(x: 0, y: thresholdY))
                    path.addLine(to: CGPoint(x: width, y: thresholdY))
                }
                .stroke(style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                .foregroundStyle(.tertiary)
                areaPath(pts, height: height)
                    .fill(LinearGradient(colors: [HG.accent.opacity(0.26), HG.accent.opacity(0)], startPoint: .top, endPoint: .bottom))
                linePath(pts)
                    .stroke(HG.accent, style: StrokeStyle(lineWidth: 2.4, lineCap: .round, lineJoin: .round))
                ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                    if value >= threshold && index < pts.count {
                        Circle().fill(.orange).frame(width: 7, height: 7).position(pts[index])
                    }
                }
            }
        }
    }

    private func points(in size: CGSize) -> [CGPoint] {
        guard values.count > 1 else { return [] }
        return values.enumerated().map { index, value in
            let x = CGFloat(index) / CGFloat(values.count - 1) * size.width
            let clamped = min(high, max(low, value))
            let y = size.height - CGFloat((clamped - low) / (high - low)) * size.height
            return CGPoint(x: x, y: y)
        }
    }

    private func linePath(_ pts: [CGPoint]) -> Path {
        var path = Path()
        guard let first = pts.first else { return path }
        path.move(to: first)
        for point in pts.dropFirst() { path.addLine(to: point) }
        return path
    }

    private func areaPath(_ pts: [CGPoint], height: CGFloat) -> Path {
        var path = linePath(pts)
        if let last = pts.last, let first = pts.first {
            path.addLine(to: CGPoint(x: last.x, y: height))
            path.addLine(to: CGPoint(x: first.x, y: height))
            path.closeSubpath()
        }
        return path
    }
}

struct AlertPreviewSection: View {
    let reports: [HealthReport]
    let onSelect: (HealthReport) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("最近异常段")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            if reports.isEmpty {
                EmptyAlertsCard()
            } else {
                VStack(spacing: 0) {
                    let shown = Array(reports.reversed().prefix(6))
                    ForEach(shown) { report in
                        Button { onSelect(report) } label: { ReportRow(report: report) }
                            .buttonStyle(.plain)
                        if report.id != shown.last?.id { Divider().padding(.leading, 64) }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))
            }
        }
    }
}

struct EmptyAlertsCard: View {
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(.green)
                .font(.title3)
            VStack(alignment: .leading, spacing: 4) {
                Text("没有列出绿色正常项")
                    .font(.subheadline.weight(.semibold))
                Text("只有超过预警线的时间段会进入这里。作息不规律但未超过阈值时,不会再用一排绿色卡片假装有发现。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 22))
    }
}

struct ReportRow: View {
    let report: HealthReport

    var body: some View {
        HStack(spacing: 13) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(report.band.tint.opacity(0.18))
                .frame(width: 40, height: 40)
                .overlay(Image(systemName: report.source.symbol).font(.system(size: 16, weight: .bold)).foregroundStyle(report.source.tint))
            VStack(alignment: .leading, spacing: 4) {
                Text(report.headline)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text(timeLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 6)
            Text(report.band.title)
                .font(.caption2.weight(.bold))
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(report.band.tint.opacity(0.16), in: .rect(cornerRadius: 9))
                .foregroundStyle(report.band.tint)
            Image(systemName: "chevron.right")
                .font(.caption.weight(.bold))
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .contentShape(.rect)
    }

    private var timeLine: String {
        "\(report.startedAt.formatted(date: .abbreviated, time: .shortened)) - \(report.endedAt.formatted(date: .omitted, time: .shortened)) · \(report.durationText)"
    }
}

struct DataInfoCard: View {
    let summary: DatasetSummary?
    let lastAnalyzed: Date?
    let trainingEpochs: Int?
    let evaluatedWindows: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("数据与模型")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.bottom, 4)
            VStack(spacing: 0) {
                if let summary = summary {
                    InfoRow(icon: "calendar", title: "数据范围",
                            value: "\(summary.firstDate.formatted(date: .abbreviated, time: .omitted)) - \(summary.lastDate.formatted(date: .abbreviated, time: .omitted))")
                    InfoRow(icon: "square.grid.3x3", title: "模型桶粒度",
                            value: "\(summary.bucketHours) 小时")
                    InfoRow(icon: "clock.badge.checkmark", title: "数据完整度",
                            value: "\(Int(summary.validRatio * 100))%")
                }
                if let epochs = trainingEpochs {
                    InfoRow(icon: "cpu", title: "个人模型训练",
                            value: "\(epochs) 轮")
                }
                if let windows = evaluatedWindows {
                    InfoRow(icon: "chart.xyaxis.line", title: "评估窗口",
                            value: "\(windows) 个")
                }
                if let analyzedAt = lastAnalyzed {
                    InfoRow(icon: "checkmark.circle", title: "上次分析",
                            value: analyzedAt.formatted(date: .abbreviated, time: .shortened), isLast: true)
                }
            }
            .padding(.horizontal, 18)
            .glassEffect(.regular, in: .rect(cornerRadius: 26))
        }
    }
}

struct InfoRow: View {
    let icon: String
    let title: String
    let value: String
    var isLast = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(.quaternary)
                    .frame(width: 28, height: 28)
                    .overlay(Image(systemName: icon).font(.system(size: 13)).foregroundStyle(.secondary))
                Text(title).foregroundStyle(.secondary)
                Spacer()
                Text(value)
                    .fontWeight(.semibold)
                    .multilineTextAlignment(.trailing)
            }
            .font(.subheadline)
            .padding(.vertical, 13)
            if !isLast { Divider() }
        }
    }
}

// MARK: - Alerts tab

struct AlertsView: View {
    @EnvironmentObject private var state: AppState
    @State private var selected: HealthReport?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 12) {
                    if state.reports.isEmpty {
                        EmptyAlertsCard()
                    } else {
                        ForEach(Array(state.reports.reversed())) { report in
                            Button { selected = report } label: {
                                ReportRow(report: report)
                                    .glassEffect(.regular, in: .rect(cornerRadius: 20))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(20)
                .padding(.bottom, 30)
            }
            .navigationTitle("异常时间段")
            .navigationDestination(item: $selected) { report in
                ReportDetailView(report: report)
            }
        }
    }
}

// MARK: - Data tab

struct DataModelView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("导入范围")
                            .font(.headline)
                        RangeSelector(selection: $state.selectedRange)
                        Text(state.selectedRange.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(18)
                    .glassEffect(.regular, in: .rect(cornerRadius: 24))

                    DataInfoCard(summary: state.summary, lastAnalyzed: state.lastAnalyzed, trainingEpochs: state.trainingEpochs, evaluatedWindows: state.evaluatedWindows)

                    VStack(alignment: .leading, spacing: 10) {
                        ModelRow(icon: "person.crop.circle.badge.exclamationmark", tint: HG.purple,
                                 title: "个人重建模型", value: "端侧训练, AdamW, masked MSE")
                        ModelRow(icon: "point.3.connected.trianglepath.dotted", tint: HG.blue,
                                 title: "通用预测基座", value: "PredictionBase.safetensors, cross-attention")
                        ModelRow(icon: "waveform.path.ecg.rectangle", tint: HG.accent,
                                 title: "原始数据哨兵", value: "心率、血氧、呼吸、睡眠短时规则")
                    }
                    .padding(18)
                    .glassEffect(.regular, in: .rect(cornerRadius: 24))

                    Button { state.importAndAnalyze(range: state.selectedRange) } label: {
                        Label("重新读取\(state.selectedRange.title)数据", systemImage: "arrow.clockwise")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glassProminent)
                    .tint(HG.accent)
                    .controlSize(.large)
                    .disabled(state.isBusy)

                    StatusLine(status: state.status, busy: state.isBusy)
                }
                .padding(20)
                .padding(.bottom, 30)
            }
            .navigationTitle("数据与模型")
        }
    }
}

struct ModelRow: View {
    let icon: String
    let tint: Color
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(tint.opacity(0.14))
                .frame(width: 36, height: 36)
                .overlay(Image(systemName: icon).foregroundStyle(tint))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(value).font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
    }
}

// MARK: - Detail

struct ReportDetailView: View {
    let report: HealthReport

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 14) {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(report.band.tint.opacity(0.16))
                            .frame(width: 54, height: 54)
                            .overlay(Image(systemName: report.source.symbol).font(.system(size: 24, weight: .bold)).foregroundStyle(report.source.tint))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(report.band.title)
                                .font(.title2.bold())
                            Text(report.source.title)
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(report.source.tint)
                        }
                        Spacer()
                    }
                    Text(report.headline).font(.title3.weight(.semibold))
                    Text("\(report.startedAt.formatted(date: .abbreviated, time: .shortened)) - \(report.endedAt.formatted(date: .abbreviated, time: .shortened)) · \(report.durationText)")
                        .font(.subheadline.weight(.semibold))
                    Text(report.detail)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.quaternary, in: .rect(cornerRadius: 15))
                }
                .padding(22)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))

                EvidenceCard(report: report)

                if report.source != .rawGuard {
                    SignalSummaryCard(report: report)
                }

                if report.featureErrors.contains(where: { $0 > 0 }) {
                    FeatureContributionCard(errors: report.featureErrors, top: report.topFeature)
                }

                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "info.circle.fill").foregroundStyle(HG.accent)
                    Text("这不是医学诊断。它只说明这一段和你的历史或通用基座相比更不寻常；如果指标持续异常或身体不适,应咨询医生。")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(16)
                .glassEffect(.regular, in: .rect(cornerRadius: 22))
            }
            .padding(20)
            .padding(.bottom, 30)
        }
        .background(AppBackground().ignoresSafeArea())
        .navigationTitle("时间段详情")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct EvidenceCard: View {
    let report: HealthReport

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("为什么列出来")
                .font(.headline)
            ForEach(report.evidence, id: \.self) { item in
                HStack(alignment: .top, spacing: 10) {
                    Circle().fill(report.source.tint).frame(width: 6, height: 6).padding(.top, 7)
                    Text(item)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
            }
        }
        .padding(20)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}

struct SignalSummaryCard: View {
    let report: HealthReport

    var body: some View {
        VStack(spacing: 16) {
            SignalBar(name: "个人模型", sub: "和你自己的历史相比", value: report.reconPercentile, tint: HG.purple)
            SignalBar(name: "通用基座", sub: "和多人走势规律相比", value: report.predPercentile, tint: HG.blue)
        }
        .padding(20)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}

struct SignalBar: View {
    let name: String
    let sub: String
    let value: Double
    let tint: Color
    private var fill: Double { min(1, max(0, value / 100)) }
    private var thresholdX: Double { HG.watchThreshold / 100 }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).font(.subheadline.weight(.semibold))
                    Text(sub).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text("第 \(Int(value.rounded())) 百分位")
                    .font(.subheadline.weight(.semibold))
                    .monospacedDigit()
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary).frame(height: 9)
                    Capsule().fill(LinearGradient(colors: [tint.opacity(0.7), tint], startPoint: .leading, endPoint: .trailing))
                        .frame(width: geo.size.width * fill, height: 9)
                    Rectangle().fill(.tertiary).frame(width: 2, height: 15).offset(x: geo.size.width * thresholdX - 1)
                }
            }
            .frame(height: 15)
        }
    }
}

struct FeatureContributionCard: View {
    let errors: [Double]
    let top: String
    private var maxError: Double { max(errors.max() ?? 1, 1e-9) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("主要偏离指标").font(.headline)
            VStack(spacing: 9) {
                ForEach(Array(errors.enumerated()), id: \.offset) { index, error in
                    if index < featureDisplayNames.count {
                        let name = featureDisplayNames[index]
                        HStack(spacing: 10) {
                            Text(name)
                                .font(.caption)
                                .frame(width: 82, alignment: .leading)
                                .foregroundStyle(name == top ? HG.accent : .secondary)
                            GeometryReader { geo in
                                Capsule()
                                    .fill(name == top ? HG.accent : Color.secondary.opacity(0.4))
                                    .frame(width: max(3, geo.size.width * CGFloat(error / maxError)), height: 8)
                            }
                            .frame(height: 8)
                        }
                    }
                }
            }
        }
        .padding(20)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}
