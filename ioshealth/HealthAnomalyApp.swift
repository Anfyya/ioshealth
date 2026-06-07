import SwiftUI
import HealthKit

// MARK: - App entry (unchanged logic)

@main
struct HealthAnomalyApp: App {
    @StateObject private var state = AppState()
    var body: some Scene {
        WindowGroup { RootView().environmentObject(state) }
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

// MARK: - Design tokens

enum HG {
    static let accent = Color(red: 1.0, green: 0.216, blue: 0.373)   // #ff375f
    static let blue   = Color(red: 0.039, green: 0.518, blue: 1.0)   // #0a84ff
    static let purple = Color(red: 0.749, green: 0.353, blue: 0.949) // #bf5af2
    static let watchThreshold = 97.0
}

extension RiskLevel {
    var title: String {
        switch self {
        case .normal: return "正常"; case .watch: return "观察"
        case .elevated: return "异常"; case .critical: return "强异常"
        }
    }
    var tint: Color {
        switch self {
        case .normal: return .green; case .watch: return .yellow
        case .elevated: return .orange; case .critical: return .red
        }
    }
    var symbol: String {
        switch self {
        case .normal: return "checkmark"; case .watch: return "exclamationmark"
        case .elevated: return "exclamationmark.triangle.fill"; case .critical: return "exclamationmark.octagon.fill"
        }
    }
}

// MARK: - Root

struct RootView: View {
    @EnvironmentObject private var state: AppState
    var body: some View {
        Group {
            if case .ready = state.phase { DashboardView() } else { OnboardingView() }
        }
        // Liquid Glass loves a content-rich, color-bleeding backdrop behind the glass.
        .background(AmbientBackground().ignoresSafeArea())
    }
}

/// Soft blurred color blobs that give the glass something to refract.
struct AmbientBackground: View {
    var body: some View {
        ZStack {
            Color(.systemBackground)
            Circle().fill(HG.accent.opacity(0.55)).frame(width: 340).blur(radius: 120)
                .offset(x: 130, y: -320)
            Circle().fill(HG.blue.opacity(0.40)).frame(width: 360).blur(radius: 130)
                .offset(x: -150, y: -120)
            Circle().fill(HG.purple.opacity(0.38)).frame(width: 320).blur(radius: 130)
                .offset(x: 150, y: 360)
            Circle().fill(Color.green.opacity(0.28)).frame(width: 280).blur(radius: 130)
                .offset(x: -120, y: 480)
        }
    }
}

// MARK: - Onboarding

struct OnboardingView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(LinearGradient(colors: [Color(red: 1, green: 0.357, blue: 0.498), HG.accent],
                                         startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 74, height: 74)
                    .overlay(Image(systemName: "heart.text.square.fill").font(.system(size: 34)).foregroundStyle(.white))
                    .shadow(color: HG.accent.opacity(0.5), radius: 18, y: 10)
                    .padding(.bottom, 22)

                Text("个人健康\n模式检测")
                    .font(.system(size: 34, weight: .bold)).tracking(-0.5)
                Text("首次启动读取 Apple 健康历史数据，在本机训练个性化模型。多人预测先验随 App 内置，你的数据不上传服务器。")
                    .font(.callout).foregroundStyle(.secondary)
                    .padding(.top, 12).fixedSize(horizontal: false, vertical: true)

                GlassEffectContainer(spacing: 12) {
                    VStack(spacing: 12) {
                        FeatureRow(icon: "waveform.path.ecg", tint: HG.accent,
                                   title: "个人重建模型", subtitle: "学习你各项指标之间的联动关系")
                        FeatureRow(icon: "point.3.connected.trianglepath.dotted", tint: HG.blue,
                                   title: "个人预测模型", subtitle: "从多人先验微调，捕捉渐变与趋势异常")
                        FeatureRow(icon: "gauge.with.dots.needle.50percent", tint: HG.purple,
                                   title: "分位数融合", subtitle: "三路模型统一为清晰的风险等级")
                    }
                }
                .padding(.top, 26)

                VStack(spacing: 12) {
                    Button { state.importAndTrain() } label: {
                        Label("导入 Apple 健康并训练", systemImage: "heart.fill")
                            .font(.headline).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glassProminent)
                    .tint(HG.accent)
                    .controlSize(.large)

                    Button { state.runDemo() } label: {
                        Label("无数据时运行演示", systemImage: "play.circle")
                            .font(.headline).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                }
                .padding(.top, 26)

                HStack(spacing: 8) {
                    Circle().fill(.green).frame(width: 8, height: 8)
                    Text(state.status)
                }
                .font(.footnote).foregroundStyle(.secondary).padding(.top, 18)

                if case .failed(let message) = state.phase {
                    Text(message).font(.footnote).foregroundStyle(.red).padding(.top, 6)
                }
            }
            .padding(20).padding(.bottom, 40)
        }
    }
}

struct FeatureRow: View {
    let icon: String, tint: Color, title: String, subtitle: String
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

// MARK: - Dashboard

struct DashboardView: View {
    @EnvironmentObject private var state: AppState
    @State private var selected: AnomalyReport?

    private var latest: AnomalyReport? { state.reports.last }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    RiskRingCard(report: latest)
                    if state.reports.count > 1 {
                        TrendCard(reports: Array(state.reports.suffix(18)))
                    }
                    ModelStatusCard(summary: state.summary, bundle: state.bundle)
                    RecentSection(reports: state.reports, onSelect: { selected = $0 })
                }
                .padding(20).padding(.bottom, 30)
            }
            .navigationTitle("健康异常")
            .scrollEdgeEffectStyle(.soft, for: .top)
            .refreshable { state.importAndTrain() }
            .navigationDestination(item: $selected) { AnomalyDetailView(report: $0, context: state.reports) }
        }
    }
}

struct RiskRingCard: View {
    let report: AnomalyReport?
    private var level: RiskLevel { report?.level ?? .normal }
    private var progress: Double { min(1, (report?.finalPercentile ?? 60) / 100) }

    var body: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle().stroke(.quaternary, lineWidth: 16)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(
                        AngularGradient(colors: [.green, HG.blue, level.tint], center: .center),
                        style: StrokeStyle(lineWidth: 16, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                VStack(spacing: 2) {
                    Text("当前风险").font(.caption.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                    Text("P\(Int(report?.finalPercentile ?? 0))")
                        .font(.system(size: 52, weight: .bold)).monospacedDigit().tracking(-1)
                    Text("融合分位数").font(.footnote).foregroundStyle(.secondary)
                }
            }
            .frame(width: 200, height: 200)
            .padding(.top, 4)

            Label(level.title, systemImage: "circle.fill")
                .font(.subheadline.weight(.semibold)).foregroundStyle(level.tint)
                .labelStyle(.titleAndIcon).imageScale(.small)

            Text(report?.reason ?? "暂无检测结果")
                .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)

            if let r = report {
                Divider().padding(.top, 4)
                HStack {
                    PercentileStat(label: "重建", value: r.reconstructionPercentile, tint: HG.accent)
                    Spacer()
                    PercentileStat(label: "个人预测", value: r.personalPredictionPercentile, tint: HG.blue)
                    Spacer()
                    PercentileStat(label: "通用", value: r.populationPredictionPercentile, tint: HG.purple)
                }
            }
        }
        .padding(22)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}

struct PercentileStat: View {
    let label: String, value: Double, tint: Color
    var body: some View {
        VStack(spacing: 5) {
            HStack(spacing: 5) {
                Circle().fill(tint).frame(width: 7, height: 7)
                Text(label).font(.caption2).foregroundStyle(.secondary)
            }
            Text("P\(Int(value))").font(.headline).monospacedDigit()
        }
    }
}

struct TrendCard: View {
    let reports: [AnomalyReport]
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("风险趋势").font(.headline)
                Spacer()
                Text("融合分位数 · 阈值 P97").font(.caption).foregroundStyle(.secondary)
            }
            TrendChart(values: reports.map(\.finalPercentile), threshold: HG.watchThreshold)
                .frame(height: 86)
        }
        .padding(20)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}

/// Lightweight area+line spark chart drawn with Path — no Charts dependency.
struct TrendChart: View {
    let values: [Double]
    var threshold: Double = 97
    private let lo = 78.0, hi = 100.0

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let pts = points(in: CGSize(width: w, height: h))
            let ty = h - CGFloat((threshold - lo) / (hi - lo)) * h
            ZStack {
                // threshold line
                Path { p in p.move(to: CGPoint(x: 0, y: ty)); p.addLine(to: CGPoint(x: w, y: ty)) }
                    .stroke(style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    .foregroundStyle(.tertiary)
                // area
                areaPath(pts, height: h)
                    .fill(LinearGradient(colors: [HG.accent.opacity(0.32), HG.accent.opacity(0)],
                                         startPoint: .top, endPoint: .bottom))
                // line
                linePath(pts)
                    .stroke(HG.accent, style: StrokeStyle(lineWidth: 2.4, lineCap: .round, lineJoin: .round))
                // anomaly dots
                ForEach(Array(values.enumerated()), id: \.offset) { i, v in
                    if v >= threshold {
                        Circle().fill(.orange).frame(width: 7, height: 7).position(pts[i])
                    }
                }
            }
        }
    }

    private func points(in size: CGSize) -> [CGPoint] {
        guard values.count > 1 else { return [] }
        return values.enumerated().map { i, v in
            let x = CGFloat(i) / CGFloat(values.count - 1) * size.width
            let y = size.height - CGFloat((min(hi, max(lo, v)) - lo) / (hi - lo)) * size.height
            return CGPoint(x: x, y: y)
        }
    }
    private func linePath(_ pts: [CGPoint]) -> Path {
        var p = Path(); guard let f = pts.first else { return p }
        p.move(to: f); pts.dropFirst().forEach { p.addLine(to: $0) }; return p
    }
    private func areaPath(_ pts: [CGPoint], height: CGFloat) -> Path {
        var p = linePath(pts)
        if let l = pts.last, let f = pts.first {
            p.addLine(to: CGPoint(x: l.x, y: height)); p.addLine(to: CGPoint(x: f.x, y: height)); p.closeSubpath()
        }
        return p
    }
}

struct ModelStatusCard: View {
    let summary: DatasetSummary?
    let bundle: UserModelBundle?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("模型状态").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
                .padding(.bottom, 4)
            VStack(spacing: 0) {
                if let s = summary {
                    StatRow(icon: "calendar", title: "历史范围",
                            value: "\(s.firstDate.formatted(date: .abbreviated, time: .omitted)) – \(s.lastDate.formatted(date: .abbreviated, time: .omitted))")
                    StatRow(icon: "square.grid.3x3.fill", title: "4h 数据块", value: "\(s.bucketCount)")
                    StatRow(icon: "chart.xyaxis.line", title: "训练窗口", value: "\(s.trainWindows)")
                    StatRow(icon: "clock.badge.checkmark", title: "数据有效率",
                            value: "\(Int(s.validRatio * 100))%", valueTint: .green)
                }
                if let b = bundle {
                    StatRow(icon: "waveform.path.ecg", title: "个人重建", value: "\(b.reconstruction.nFeatures) 维")
                    StatRow(icon: "point.3.connected.trianglepath.dotted", title: "预测先验", value: b.populationPrior.name)
                    StatRow(icon: "clock", title: "模型创建",
                            value: b.createdAt.formatted(date: .abbreviated, time: .shortened), isLast: true)
                }
            }
            .padding(.horizontal, 18)
            .glassEffect(.regular, in: .rect(cornerRadius: 26))
        }
    }
}

struct StatRow: View {
    let icon: String, title: String, value: String
    var valueTint: Color = .primary
    var isLast: Bool = false
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 8, style: .continuous).fill(.quaternary)
                    .frame(width: 28, height: 28)
                    .overlay(Image(systemName: icon).font(.system(size: 13)).foregroundStyle(.secondary))
                Text(title).foregroundStyle(.secondary)
                Spacer()
                Text(value).fontWeight(.semibold).monospacedDigit().foregroundStyle(valueTint)
                    .multilineTextAlignment(.trailing)
            }
            .font(.subheadline)
            .padding(.vertical, 13)
            if !isLast { Divider() }
        }
    }
}

struct RecentSection: View {
    let reports: [AnomalyReport]
    let onSelect: (AnomalyReport) -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("最近窗口").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
            if reports.isEmpty {
                Text("暂无检测结果").foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity).padding(.vertical, 24)
                    .glassEffect(.regular, in: .rect(cornerRadius: 26))
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(reports.reversed().prefix(6))) { report in
                        Button { onSelect(report) } label: { AnomalyReportRow(report: report) }
                            .buttonStyle(.plain)
                        if report.id != reports.reversed().prefix(6).last?.id { Divider().padding(.leading, 64) }
                    }
                }
                .padding(.horizontal, 8).padding(.vertical, 4)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))
            }
        }
    }
}

struct AnomalyReportRow: View {
    let report: AnomalyReport
    var body: some View {
        HStack(spacing: 13) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(LinearGradient(colors: [report.level.tint, report.level.tint.opacity(0.7)],
                                     startPoint: .topLeading, endPoint: .bottomTrailing))
                .frame(width: 38, height: 38)
                .overlay(Image(systemName: report.level.symbol).font(.system(size: 15, weight: .bold)).foregroundStyle(.white))
            VStack(alignment: .leading, spacing: 3) {
                Text(report.startedAt, format: .dateTime.month().day().hour().minute())
                    .font(.subheadline.weight(.semibold))
                Text(report.reason).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 6)
            VStack(alignment: .trailing, spacing: 5) {
                Text(report.level.title)
                    .font(.caption2.weight(.bold))
                    .padding(.horizontal, 9).padding(.vertical, 4)
                    .background(report.level.tint.opacity(0.16), in: .rect(cornerRadius: 9))
                    .foregroundStyle(report.level.tint)
                Text("融合 P\(Int(report.finalPercentile))").font(.caption2).foregroundStyle(.tertiary).monospacedDigit()
            }
            Image(systemName: "chevron.right").font(.caption.weight(.bold)).foregroundStyle(.tertiary)
        }
        .padding(.vertical, 10).padding(.horizontal, 12)
        .contentShape(.rect)
    }
}

// MARK: - Detail

struct AnomalyDetailView: View {
    let report: AnomalyReport
    let context: [AnomalyReport]

    private var trend: [Double] {
        guard let idx = context.firstIndex(where: { $0.id == report.id }) else {
            return context.suffix(14).map(\.finalPercentile)
        }
        let lower = max(0, idx - 8), upper = min(context.count - 1, idx + 5)
        return Array(context[lower...upper]).map(\.finalPercentile)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // hero
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 14) {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(LinearGradient(colors: [report.level.tint, report.level.tint.opacity(0.7)],
                                                 startPoint: .topLeading, endPoint: .bottomTrailing))
                            .frame(width: 54, height: 54)
                            .overlay(Image(systemName: report.level.symbol).font(.system(size: 24, weight: .bold)).foregroundStyle(.white))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(report.level.title).font(.title2.bold())
                            Text(report.startedAt.formatted(date: .abbreviated, time: .shortened) + " 起 4 小时窗口")
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text("P\(Int(report.finalPercentile))")
                            .font(.system(size: 40, weight: .bold)).monospacedDigit()
                            .foregroundStyle(report.level.tint)
                        Text("融合分位数 · 阈值 P97 触发观察").font(.footnote).foregroundStyle(.secondary)
                    }
                    Text(report.reason)
                        .font(.callout)
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.quaternary, in: .rect(cornerRadius: 15))
                }
                .padding(22)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))

                // three sources
                VStack(spacing: 18) {
                    SourceBar(name: "个人重建", sub: "指标联动", value: report.reconstructionPercentile, tint: HG.accent)
                    SourceBar(name: "个人预测", sub: "趋势偏离", value: report.personalPredictionPercentile, tint: HG.blue)
                    SourceBar(name: "通用先验", sub: "多人模型", value: report.populationPredictionPercentile, tint: HG.purple)
                }
                .padding(20)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))

                // trend
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("窗口前后趋势").font(.headline)
                        Spacer()
                        Text("融合分位数").font(.caption).foregroundStyle(.secondary)
                    }
                    TrendChart(values: trend, threshold: HG.watchThreshold).frame(height: 130)
                }
                .padding(20)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))

                // disclaimer
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "info.circle.fill").foregroundStyle(HG.accent)
                    Text("该结论由端侧三路模型分位数融合得到，仅作健康参考，不构成医学诊断。如持续异常请咨询医生。")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                .padding(16)
                .glassEffect(.regular, in: .rect(cornerRadius: 22))
            }
            .padding(20).padding(.bottom, 30)
        }
        .background(AmbientBackground().ignoresSafeArea())
        .navigationTitle("异常详情")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct SourceBar: View {
    let name: String, sub: String, value: Double, tint: Color
    private var fill: Double { min(1, max(0, (value - 50) / 50)) }
    private var threshX: Double { (HG.watchThreshold - 50) / 50 }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                HStack(spacing: 8) {
                    Circle().fill(tint).frame(width: 8, height: 8)
                    Text(name).font(.subheadline.weight(.medium))
                    Text(sub).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text("P\(Int(value))").font(.subheadline.weight(.semibold)).monospacedDigit()
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary).frame(height: 9)
                    Capsule()
                        .fill(LinearGradient(colors: [tint.opacity(0.7), tint], startPoint: .leading, endPoint: .trailing))
                        .frame(width: geo.size.width * fill, height: 9)
                    Rectangle().fill(.tertiary).frame(width: 2, height: 15)
                        .offset(x: geo.size.width * threshX - 1)
                }
            }
            .frame(height: 15)
        }
    }
}
