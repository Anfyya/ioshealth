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
        let remapped = ATEngine.remapConvWeights(raw)
        // Use the throwing update with no strict verification: the convenience
        // (non-throwing) update does `try!` + verification and turns any
        // key/shape check into a hard crash. Our keys map 1:1 to the bundled
        // weights, so `.none` loads them all; any real failure now propagates
        // as a catchable Swift error instead of trapping.
        try predNet.update(parameters: ModuleParameters.unflattened(remapped), verify: .none)
        eval(predNet)
    }

    /// PyTorch Conv1d weights are [out, in, k]; MLX Conv1d wants [out, k, in].
    private static func remapConvWeights(_ w: [String: MLXArray]) -> [String: MLXArray] {
        var out = w
        for (k, v) in w where k.hasSuffix("tokenConv.weight") || k.hasSuffix("conv1.weight") || k.hasSuffix("conv2.weight") {
            out[k] = v.swappedAxes(1, 2)
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

    /// Max per-step reconstruction error within each window (valid steps only).
    func reconScores(_ windows: [HealthWindow]) -> [Double] {
        guard let model = reconNet, !windows.isEmpty else {
            return Array(repeating: 0, count: windows.count)
        }
        let (X, M) = ATEngine.toBatch(windows, cfg)
        let err = square(model(X) - X).mean(axis: -1) * M   // [N,L]
        return err.max(axis: 1).asArray(Float.self).map(Double.init)
    }

    /// Max per-step prediction error over the forecast horizon for each window.
    func predScores(_ windows: [HealthWindow]) -> [Double] {
        guard !windows.isEmpty else { return [] }
        let (X, M) = ATEngine.toBatch(windows, cfg)
        let pred = predNet(X)                                       // [N,horizon,F]
        let h = cfg.horizon
        let future = X[0..., (cfg.winSize - h) ..< cfg.winSize, 0...]
        let fmask = M[0..., (cfg.winSize - h) ..< cfg.winSize]
        let err = square(pred - future).mean(axis: -1) * fmask     // [N,horizon]
        return err.max(axis: 1).asArray(Float.self).map(Double.init)
    }

    /// Per-feature reconstruction error at the most anomalous step of a window
    /// (used to explain which metric is off). Returns `cfg.encIn` values.
    func reconFeatureError(_ window: HealthWindow) -> [Double] {
        guard let model = reconNet else { return Array(repeating: 0, count: cfg.encIn) }
        let (X, M) = ATEngine.toBatch([window], cfg)
        let sq = square(model(X) - X)                 // [1,L,F]
        let perStep = (sq.mean(axis: -1)[0]) * M[0]   // [L]
        let worst = argMax(perStep, axis: 0).item(Int.self)
        return sq[0, worst].asArray(Float.self).map(Double.init)
    }
}
