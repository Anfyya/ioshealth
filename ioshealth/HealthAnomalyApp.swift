import SwiftUI
import HealthKit

// MARK: - App entry

@main
struct HealthAnomalyApp: App {
    @StateObject private var state = AppState()
    var body: some Scene {
        WindowGroup { RootView().environmentObject(state) }
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

    private let health = HealthKitDataSource()

    private struct AnalysisResult: Sendable {
        let summary: DatasetSummary
        let reports: [HealthReport]
    }

    init() {
        if let saved = ATStore.load() {
            summary = saved.summary
            reports = saved.reports
            lastAnalyzed = saved.createdAt
            phase = .ready
            status = "已载入上次的分析结果"
        }
    }

    var flaggedCount: Int { reports.filter { $0.band != .normal }.count }
    var latest: HealthReport? { reports.last }

    func importAndAnalyze(days: Int = 730) {
        Task {
            do {
                phase = .importing
                status = "正在请求健康数据权限并读取历史…"
                try await health.requestAuthorization()
                let events = try await health.loadHistory(daysBack: days)
                try await analyze(events: events, save: true)
            } catch {
                phase = .failed(error.localizedDescription)
                status = "出错了:\(error.localizedDescription)"
            }
        }
    }

    func runDemo() {
        Task {
            do {
                status = "正在生成示例数据…"
                let events = SyntheticHealthHistory.makeEvents(days: 240)
                try await analyze(events: events, save: false)
            } catch {
                phase = .failed(error.localizedDescription)
                status = "出错了:\(error.localizedDescription)"
            }
        }
    }

    private func analyze(events: [RawHealthEvent], save: Bool) async throws {
        phase = .preprocessing
        status = "正在整理你的健康数据(每 4 小时一段)…"
        let dataset = try await Task.detached(priority: .userInitiated) {
            try HealthPreprocessor().prepare(events: events)
        }.value
        summary = dataset.summary

        phase = .training
        status = "正在你的 iPhone 上训练专属模型…(全程不上传)"
        let result = try await Task.detached(priority: .userInitiated) { () -> AnalysisResult in
            let engine = try ATEngine()
            engine.trainRecon(dataset.train, epochs: 8, lr: 3e-4, batchSize: 16)

            let reconCal = Calibrator(engine.reconScores(dataset.train))
            let predCal = Calibrator(engine.predScores(dataset.train))

            let recent = Array((dataset.validation + dataset.test).suffix(30))
            let reconR = engine.reconScores(recent)
            let predR = engine.predScores(recent)

            var reports: [HealthReport] = []
            for (i, w) in recent.enumerated() {
                let attr = engine.reconFeatureError(w)
                reports.append(Fusion.report(
                    startedAt: w.start,
                    reconScore: i < reconR.count ? reconR[i] : 0,
                    predScore: i < predR.count ? predR[i] : 0,
                    reconCal: reconCal, predCal: predCal,
                    featureErrors: attr))
            }
            return AnalysisResult(summary: dataset.summary, reports: reports)
        }.value

        reports = result.reports
        lastAnalyzed = Date()
        if save {
            ATStore.save(PersistedState(createdAt: Date(), summary: result.summary, reports: result.reports))
        }
        phase = .ready
        status = "分析完成"
    }
}

// MARK: - Design tokens

enum HG {
    static let accent = Color(red: 1.0, green: 0.216, blue: 0.373)
    static let blue   = Color(red: 0.039, green: 0.518, blue: 1.0)
    static let purple = Color(red: 0.749, green: 0.353, blue: 0.949)
    static let watchThreshold = 97.0
}

extension RiskBand {
    var title: String {
        switch self {
        case .normal: return "正常"
        case .watch: return "注意"
        case .elevated: return "异常"
        case .critical: return "需重点关注"
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
        case .watch: return "exclamationmark.circle.fill"
        case .elevated: return "exclamationmark.triangle.fill"
        case .critical: return "exclamationmark.octagon.fill"
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
        .background(AmbientBackground().ignoresSafeArea())
    }
}

struct AmbientBackground: View {
    var body: some View {
        ZStack {
            Color(.systemBackground)
            Circle().fill(HG.accent.opacity(0.55)).frame(width: 340).blur(radius: 120).offset(x: 130, y: -320)
            Circle().fill(HG.blue.opacity(0.40)).frame(width: 360).blur(radius: 130).offset(x: -150, y: -120)
            Circle().fill(HG.purple.opacity(0.38)).frame(width: 320).blur(radius: 130).offset(x: 150, y: 360)
            Circle().fill(Color.green.opacity(0.28)).frame(width: 280).blur(radius: 130).offset(x: -120, y: 480)
        }
    }
}

// MARK: - Onboarding

struct OnboardingView: View {
    @EnvironmentObject private var state: AppState
    private var busy: Bool {
        switch state.phase { case .importing, .preprocessing, .training: return true; default: return false }
    }

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

                Text("读懂你自己的\n身体节律").font(.system(size: 34, weight: .bold)).tracking(-0.5)
                Text("第一次使用会读取你的 Apple 健康历史,在这台 iPhone 上训练一个只属于你的模型,帮你找出和平时不一样的时段。所有数据都留在本机,不会上传。")
                    .font(.callout).foregroundStyle(.secondary)
                    .padding(.top, 12).fixedSize(horizontal: false, vertical: true)

                GlassEffectContainer(spacing: 12) {
                    VStack(spacing: 12) {
                        FeatureRow(icon: "waveform.path.ecg", tint: HG.accent,
                                   title: "学习你自己的规律", subtitle: "记下你各项健康指标平时的关系和节奏")
                        FeatureRow(icon: "sparkle.magnifyingglass", tint: HG.blue,
                                   title: "发现不对劲的时段", subtitle: "当指标的配合或趋势偏离时提醒你")
                        FeatureRow(icon: "lock.iphone", tint: HG.purple,
                                   title: "完全在本机", subtitle: "不联网、不上传,隐私留在手机里")
                    }
                }
                .padding(.top, 26)

                VStack(spacing: 12) {
                    Button { state.importAndAnalyze() } label: {
                        Label("读取健康数据并分析", systemImage: "heart.fill")
                            .font(.headline).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glassProminent).tint(HG.accent).controlSize(.large).disabled(busy)

                    Button { state.runDemo() } label: {
                        Label("用示例数据先体验", systemImage: "play.circle")
                            .font(.headline).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glass).controlSize(.large).disabled(busy)
                }
                .padding(.top, 26)

                HStack(spacing: 8) {
                    if busy { ProgressView().controlSize(.small) }
                    else { Circle().fill(.green).frame(width: 8, height: 8) }
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
                .fill(.quaternary).frame(width: 44, height: 44)
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
    @State private var selected: HealthReport?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    StatusHeroCard(latest: state.latest, flagged: state.flaggedCount, total: state.reports.count)
                    if state.reports.count > 1 {
                        TrendCard(reports: Array(state.reports.suffix(18)))
                    }
                    if !state.reports.isEmpty {
                        RecentSection(reports: state.reports, onSelect: { selected = $0 })
                    }
                    InfoCard(summary: state.summary, lastAnalyzed: state.lastAnalyzed)
                }
                .padding(20).padding(.bottom, 30)
            }
            .navigationTitle("身体节律")
            .scrollEdgeEffectStyle(.soft, for: .top)
            .refreshable { state.importAndAnalyze() }
            .navigationDestination(item: $selected) { ReportDetailView(report: $0) }
        }
    }
}

struct StatusHeroCard: View {
    let latest: HealthReport?
    let flagged: Int
    let total: Int

    private var band: RiskBand { latest?.band ?? .normal }

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle().fill(band.tint.opacity(0.18)).frame(width: 116, height: 116)
                Image(systemName: band.symbol).font(.system(size: 52, weight: .bold)).foregroundStyle(band.tint)
            }
            .padding(.top, 4)

            Text(latest?.headline ?? "还没有分析结果")
                .font(.title3.weight(.bold)).multilineTextAlignment(.center)

            Text(band == .normal
                 ? "最近 \(total) 个时段都正常。"
                 : "最近 \(total) 个时段里,有 \(flagged) 个值得留意。")
                .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(24)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}

struct TrendCard: View {
    let reports: [HealthReport]
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("近期波动").font(.headline)
                Spacer()
                Text("越高越不寻常").font(.caption).foregroundStyle(.secondary)
            }
            TrendChart(values: reports.map(\.finalPercentile), threshold: HG.watchThreshold)
                .frame(height: 86)
        }
        .padding(20)
        .glassEffect(.regular, in: .rect(cornerRadius: 26))
    }
}

struct TrendChart: View {
    let values: [Double]
    var threshold: Double = 97
    private let lo = 70.0, hi = 100.0

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let pts = points(in: CGSize(width: w, height: h))
            let ty = h - CGFloat((threshold - lo) / (hi - lo)) * h
            ZStack {
                Path { p in p.move(to: CGPoint(x: 0, y: ty)); p.addLine(to: CGPoint(x: w, y: ty)) }
                    .stroke(style: StrokeStyle(lineWidth: 1, dash: [3, 3])).foregroundStyle(.tertiary)
                areaPath(pts, height: h)
                    .fill(LinearGradient(colors: [HG.accent.opacity(0.32), HG.accent.opacity(0)], startPoint: .top, endPoint: .bottom))
                linePath(pts).stroke(HG.accent, style: StrokeStyle(lineWidth: 2.4, lineCap: .round, lineJoin: .round))
                ForEach(Array(values.enumerated()), id: \.offset) { i, v in
                    if v >= threshold { Circle().fill(.orange).frame(width: 7, height: 7).position(pts[i]) }
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

struct RecentSection: View {
    let reports: [HealthReport]
    let onSelect: (HealthReport) -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("最近的时段").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
            VStack(spacing: 0) {
                let shown = Array(reports.reversed().prefix(8))
                ForEach(shown) { report in
                    Button { onSelect(report) } label: { ReportRow(report: report) }.buttonStyle(.plain)
                    if report.id != shown.last?.id { Divider().padding(.leading, 64) }
                }
            }
            .padding(.horizontal, 8).padding(.vertical, 4)
            .glassEffect(.regular, in: .rect(cornerRadius: 26))
        }
    }
}

struct ReportRow: View {
    let report: HealthReport
    var body: some View {
        HStack(spacing: 13) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(LinearGradient(colors: [report.band.tint, report.band.tint.opacity(0.7)],
                                     startPoint: .topLeading, endPoint: .bottomTrailing))
                .frame(width: 38, height: 38)
                .overlay(Image(systemName: report.band.symbol).font(.system(size: 15, weight: .bold)).foregroundStyle(.white))
            VStack(alignment: .leading, spacing: 3) {
                Text(report.startedAt, format: .dateTime.month().day().hour().minute())
                    .font(.subheadline.weight(.semibold))
                Text(report.headline).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 6)
            Text(report.band.title)
                .font(.caption2.weight(.bold))
                .padding(.horizontal, 9).padding(.vertical, 4)
                .background(report.band.tint.opacity(0.16), in: .rect(cornerRadius: 9))
                .foregroundStyle(report.band.tint)
            Image(systemName: "chevron.right").font(.caption.weight(.bold)).foregroundStyle(.tertiary)
        }
        .padding(.vertical, 10).padding(.horizontal, 12)
        .contentShape(.rect)
    }
}

struct InfoCard: View {
    let summary: DatasetSummary?
    let lastAnalyzed: Date?
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("数据与模型").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary).padding(.bottom, 4)
            VStack(spacing: 0) {
                if let s = summary {
                    InfoRow(icon: "calendar", title: "数据范围",
                            value: "\(s.firstDate.formatted(date: .abbreviated, time: .omitted)) – \(s.lastDate.formatted(date: .abbreviated, time: .omitted))")
                    InfoRow(icon: "clock.badge.checkmark", title: "数据完整度",
                            value: "\(Int(s.validRatio * 100))%", valueTint: .green)
                }
                if let d = lastAnalyzed {
                    InfoRow(icon: "checkmark.circle", title: "上次分析",
                            value: d.formatted(date: .abbreviated, time: .shortened))
                }
                InfoRow(icon: "lock.fill", title: "隐私", value: "全部留在本机", isLast: true)
            }
            .padding(.horizontal, 18)
            .glassEffect(.regular, in: .rect(cornerRadius: 26))
        }
    }
}

struct InfoRow: View {
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
                Text(value).fontWeight(.semibold).foregroundStyle(valueTint).multilineTextAlignment(.trailing)
            }
            .font(.subheadline).padding(.vertical, 13)
            if !isLast { Divider() }
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
                            .fill(LinearGradient(colors: [report.band.tint, report.band.tint.opacity(0.7)],
                                                 startPoint: .topLeading, endPoint: .bottomTrailing))
                            .frame(width: 54, height: 54)
                            .overlay(Image(systemName: report.band.symbol).font(.system(size: 24, weight: .bold)).foregroundStyle(.white))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(report.band.title).font(.title2.bold())
                            Text(report.startedAt.formatted(date: .abbreviated, time: .shortened) + " 起的 4 小时")
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    Text(report.headline).font(.title3.weight(.semibold))
                    Text(report.detail)
                        .font(.callout).foregroundStyle(.secondary)
                        .padding(14).frame(maxWidth: .infinity, alignment: .leading)
                        .background(.quaternary, in: .rect(cornerRadius: 15))
                }
                .padding(22)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))

                // Two signals, in plain language
                VStack(spacing: 18) {
                    SignalBar(name: "指标之间的配合", sub: "和你平时的联动关系比", value: report.reconPercentile, tint: HG.accent)
                    SignalBar(name: "变化趋势", sub: "和你以往的走势比", value: report.predPercentile, tint: HG.blue)
                }
                .padding(20)
                .glassEffect(.regular, in: .rect(cornerRadius: 26))

                if report.featureErrors.contains(where: { $0 > 0 }) {
                    FeatureContributionCard(errors: report.featureErrors, top: report.topFeature)
                }

                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "info.circle.fill").foregroundStyle(HG.accent)
                    Text("这是基于你个人历史的参考提示,不是医学诊断。如果某项指标持续异常或你身体不适,请咨询医生。")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                .padding(16)
                .glassEffect(.regular, in: .rect(cornerRadius: 22))
            }
            .padding(20).padding(.bottom, 30)
        }
        .background(AmbientBackground().ignoresSafeArea())
        .navigationTitle("时段详情")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct SignalBar: View {
    let name: String, sub: String, value: Double, tint: Color
    private var fill: Double { min(1, max(0, value / 100)) }
    private var threshX: Double { HG.watchThreshold / 100 }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                HStack(spacing: 8) {
                    Circle().fill(tint).frame(width: 8, height: 8)
                    Text(name).font(.subheadline.weight(.medium))
                    Text(sub).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(Int(value)) / 100").font(.subheadline.weight(.semibold)).monospacedDigit()
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary).frame(height: 9)
                    Capsule().fill(LinearGradient(colors: [tint.opacity(0.7), tint], startPoint: .leading, endPoint: .trailing))
                        .frame(width: geo.size.width * fill, height: 9)
                    Rectangle().fill(.tertiary).frame(width: 2, height: 15).offset(x: geo.size.width * threshX - 1)
                }
            }
            .frame(height: 15)
        }
    }
}

struct FeatureContributionCard: View {
    let errors: [Double]
    let top: String
    private var maxErr: Double { max(errors.max() ?? 1, 1e-9) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("哪些指标最不寻常").font(.headline)
            VStack(spacing: 9) {
                ForEach(Array(errors.enumerated()), id: \.offset) { i, e in
                    if i < featureDisplayNames.count {
                        let name = featureDisplayNames[i]
                        HStack(spacing: 10) {
                            Text(name).font(.caption).frame(width: 78, alignment: .leading)
                                .foregroundStyle(name == top ? HG.accent : .secondary)
                            GeometryReader { geo in
                                Capsule().fill(name == top ? HG.accent : Color.secondary.opacity(0.4))
                                    .frame(width: max(3, geo.size.width * CGFloat(e / maxErr)), height: 8)
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
