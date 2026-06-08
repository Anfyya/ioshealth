import Foundation
import MLX
import MLXNN
import MLXOptimizers

// MARK: - On-device Anomaly Transformer engine
//
// Owns the two networks and the data plumbing:
//   - PredNet  : multi-person prediction base, loaded from the bundled
//                PredictionBase.safetensors (inference only).
//   - ReconNet : personal reconstruction model, trained on-device from the
//                user's own windows (masked MSE + AdamW).
//
// Produces per-window reconstruction / prediction error scores and per-feature
// attribution, which the fusion layer turns into a risk report.

enum ATEngineError: Error { case missingResource(String) }

struct ATScoreDetail: Sendable {
    let score: Double
    let stepIndex: Int
}

private struct PredBaseConfig: Decodable {
    let win_size: Int
    let context_len: Int
    let predict_horizon: Int
    let enc_in: Int
    let c_out: Int
    let d_model: Int
    let n_heads: Int
    let e_layers: Int
    let d_ff: Int
}

final class ATEngine {
    let cfg: ATConfig
    let predNet: PredNet
    private(set) var reconNet: ReconNet?

    /// Loads the bundled prediction base and prepares the engine.
    init() throws {
        guard let cfgURL = Bundle.main.url(forResource: "PredictionBase", withExtension: "json") else {
            throw ATEngineError.missingResource("PredictionBase.json")
        }
        let pbc = try JSONDecoder().decode(PredBaseConfig.self, from: Data(contentsOf: cfgURL))
        var c = ATConfig()
        c.winSize = pbc.win_size; c.contextLen = pbc.context_len; c.horizon = pbc.predict_horizon
        c.encIn = pbc.enc_in; c.cOut = pbc.c_out
        c.dModel = pbc.d_model; c.nHeads = pbc.n_heads; c.eLayers = pbc.e_layers; c.dFF = pbc.d_ff
        self.cfg = c

        self.predNet = PredNet(c)
        guard let wURL = Bundle.main.url(forResource: "PredictionBase", withExtension: "safetensors") else {
            throw ATEngineError.missingResource("PredictionBase.safetensors")
        }
        let raw = try loadArrays(url: wURL)
        let remapped = ATEngine.remapWeights(raw)
        // Use the throwing update with no strict verification: the convenience
        // (non-throwing) update does `try!` + verification and turns any
        // key/shape check into a hard crash. Our keys map 1:1 to the bundled
        // weights, so `.none` loads them all; any real failure now propagates
        // as a catchable Swift error instead of trapping.
        try predNet.update(parameters: ModuleParameters.unflattened(remapped), verify: .none)
        eval(predNet)
    }

    /// Adapt the exported PyTorch weights to the MLX module layout:
    ///  - Conv1d weights [out, in, k] -> MLX [out, k, in]
    ///  - FFN sequential indices ffn.0 / ffn.3 -> named children fc1 / fc2
    ///    (numeric keys would be parsed as array indices by `unflattened`).
    private static func remapWeights(_ w: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in w {
            let value = (k.hasSuffix("tokenConv.weight") || k.hasSuffix("conv1.weight") || k.hasSuffix("conv2.weight"))
                ? v.swappedAxes(1, 2) : v
            var key = k
            key = key.replacingOccurrences(of: "predictor.ffn.0.", with: "predictor.ffn.fc1.")
            key = key.replacingOccurrences(of: "predictor.ffn.3.", with: "predictor.ffn.fc2.")
            out[key] = value
        }
        return out
    }

    // MARK: Data plumbing

    /// Pack windows into ([N,L,F], [N,L]) float tensors.
    static func toBatch(_ windows: [HealthWindow], _ cfg: ATConfig) -> (MLXArray, MLXArray) {
        let N = windows.count, L = cfg.winSize, F = cfg.encIn
        var xs = [Float](repeating: 0, count: N * L * F)
        var ms = [Float](repeating: 0, count: N * L)
        for (n, w) in windows.enumerated() {
            let rows = min(L, w.values.count)
            for t in 0..<rows {
                ms[n * L + t] = Float(w.mask[t])
                let row = w.values[t]
                let cols = min(F, row.count)
                for f in 0..<cols { xs[(n * L + t) * F + f] = Float(row[f]) }
            }
        }
        return (MLXArray(xs, [N, L, F]), MLXArray(ms, [N, L]))
    }

    private static func maskedMSE(_ out: MLXArray, _ target: MLXArray, _ mask: MLXArray) -> MLXArray {
        let diff = square(out - target).mean(axis: -1)   // [B,L]
        let masked = diff * mask
        return masked.sum() / (mask.sum() + 1e-8)
    }

    // MARK: Training (personal reconstruction)

    /// Train the reconstruction model on the user's windows. Reconstruction on
    /// single-person data does not overfit (verified), so this trains from scratch.
    func trainRecon(_ windows: [HealthWindow], epochs: Int = 8, lr: Float = 3e-4, batchSize: Int = 16,
                    progress: ((Int, Double) -> Void)? = nil) {
        guard !windows.isEmpty else { return }
        let model = ReconNet(cfg)
        let (X, M) = ATEngine.toBatch(windows, cfg)
        let N = windows.count
        let opt = AdamW(learningRate: lr, weightDecay: 1e-4)

        let lossAndGrad = valueAndGrad(model: model) { (m: ReconNet, arr: [MLXArray]) -> [MLXArray] in
            let out = m(arr[0])
            return [ATEngine.maskedMSE(out, arr[0], arr[1])]
        }

        var order = Array(0..<N)
        for epoch in 0..<epochs {
            order.shuffle()
            var last = 0.0
            var s = 0
            while s < N {
                let e = min(s + batchSize, N)
                let sel = MLXArray(order[s..<e].map { Int32($0) })
                let xb = take(X, sel, axis: 0)
                let mb = take(M, sel, axis: 0)
                let (vals, grads) = lossAndGrad(model, [xb, mb])
                opt.update(model: model, gradients: grads)
                eval(model, opt)
                last = Double(vals[0].item(Float.self))
                s = e
            }
            progress?(epoch + 1, last)
        }
        reconNet = model
    }

    // MARK: Scoring

    // Scoring runs in chunks: a full-dataset forward would build an attention
    // tensor of [N, heads, L, L] (~0.5 GB for hundreds of windows) and get the
    // app OOM-killed on device. Chunking bounds peak memory to a few dozen MB.
    private static let scoreBatch = 32

    /// Max per-step reconstruction error within each window (valid steps only).
    func reconScores(_ windows: [HealthWindow]) -> [Double] {
        reconScoreDetails(windows).map(\.score)
    }

    /// Max per-step reconstruction error and the bucket that caused it.
    func reconScoreDetails(_ windows: [HealthWindow]) -> [ATScoreDetail] {
        guard let model = reconNet, !windows.isEmpty else {
            return Array(repeating: ATScoreDetail(score: 0, stepIndex: 0), count: windows.count)
        }
        var out: [ATScoreDetail] = []
        var s = 0
        while s < windows.count {
            let e = min(s + ATEngine.scoreBatch, windows.count)
            let (X, M) = ATEngine.toBatch(Array(windows[s..<e]), cfg)
            let err = square(model(X) - X).mean(axis: -1) * M   // [B,L]
            eval(err)
            out.append(contentsOf: ATEngine.details(from: err.asArray(Float.self), rows: e - s, cols: cfg.winSize))
            s = e
        }
        return out
    }

    /// Max per-step prediction error over the forecast horizon for each window.
    func predScores(_ windows: [HealthWindow]) -> [Double] {
        predScoreDetails(windows).map(\.score)
    }

    /// Max prediction error over the forecast horizon and the horizon bucket that caused it.
    func predScoreDetails(_ windows: [HealthWindow]) -> [ATScoreDetail] {
        guard !windows.isEmpty else { return [] }
        let h = cfg.horizon
        var out: [ATScoreDetail] = []
        var s = 0
        while s < windows.count {
            let e = min(s + ATEngine.scoreBatch, windows.count)
            let (X, M) = ATEngine.toBatch(Array(windows[s..<e]), cfg)
            let pred = predNet(X)                                       // [B,horizon,F]
            let future = X[0..., (cfg.winSize - h) ..< cfg.winSize, 0...]
            let fmask = M[0..., (cfg.winSize - h) ..< cfg.winSize]
            let err = square(pred - future).mean(axis: -1) * fmask     // [B,horizon]
            eval(err)
            out.append(contentsOf: ATEngine.details(from: err.asArray(Float.self), rows: e - s, cols: h))
            s = e
        }
        return out
    }

    /// Per-feature reconstruction error at the most anomalous step of a window
    /// (used to explain which metric is off). Returns `cfg.encIn` values.
    func reconFeatureError(_ window: HealthWindow, stepIndex: Int? = nil) -> [Double] {
        guard let model = reconNet else { return Array(repeating: 0, count: cfg.encIn) }
        let (X, M) = ATEngine.toBatch([window], cfg)
        let sq = square(model(X) - X)                 // [1,L,F]
        let worst: Int
        if let providedStep = stepIndex {
            worst = max(0, min(providedStep, cfg.winSize - 1))
        } else {
            let perStep = (sq.mean(axis: -1)[0]) * M[0]   // [L]
            worst = argMax(perStep, axis: 0).item(Int.self)
        }
        return sq[0, worst].asArray(Float.self).map(Double.init)
    }

    private static func details(from flat: [Float], rows: Int, cols: Int) -> [ATScoreDetail] {
        guard rows > 0, cols > 0 else { return [] }
        var out: [ATScoreDetail] = []
        out.reserveCapacity(rows)
        for row in 0..<rows {
            let offset = row * cols
            var best = -Double.infinity
            var bestIndex = 0
            for col in 0..<cols {
                let value = Double(flat[offset + col])
                if value > best {
                    best = value
                    bestIndex = col
                }
            }
            out.append(ATScoreDetail(score: best.isFinite ? best : 0, stepIndex: bestIndex))
        }
        return out
    }
}
