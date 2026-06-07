import Foundation
import MLX
import MLXNN

// MARK: - Anomaly Transformer (MLX-swift port)
//
// Faithful port of the PyTorch model in the `health` project (model.py).
// Two top-level networks share the embedding + encoder:
//   - ReconNet : reconstruction mode (input [B,L,8] -> [B,L,8]); trained on-device.
//   - PredNet  : cross-attention prediction (encode first 228 steps, predict 24);
//                weights loaded from the bundled PredictionBase.safetensors.
//
// Deterministic buffers (sinusoidal positional encoding, the attention distance
// matrix) are NOT parameters — they are regenerated here by the same formulas,
// so they were excluded from the exported weights.

struct ATConfig {
    var winSize = 252
    var contextLen = 228
    var horizon = 24
    var encIn = 8
    var cOut = 8
    var dModel = 64
    var nHeads = 4
    var eLayers = 2
    var dFF = 128
}

// MARK: - Deterministic helpers (cached per sequence length)

enum ATBuffers {
    private static var posCache: [Int: MLXArray] = [:]
    private static var maskCache: [Int: MLXArray] = [:]

    /// Sinusoidal positional encoding, shape [1, L, d]. Matches PyTorch
    /// PositionalEmbedding: div_term = exp(arange(0,d,2) * -(log(10000)/d)).
    static func positional(_ L: Int, _ d: Int) -> MLXArray {
        if let c = posCache[L] { return c }
        var pe = [Float](repeating: 0, count: L * d)
        for pos in 0..<L {
            var i = 0
            while i < d {
                let div = exp(Double(i) * -(log(10000.0) / Double(d)))
                pe[pos * d + i] = Float(sin(Double(pos) * div))
                if i + 1 < d { pe[pos * d + i + 1] = Float(cos(Double(pos) * div)) }
                i += 2
            }
        }
        let a = MLXArray(pe, [1, L, d])
        posCache[L] = a
        return a
    }

    /// Additive causal mask [1,1,L,L]: 0 on/below diagonal, -1e9 above (future).
    static func causalMask(_ L: Int) -> MLXArray {
        if let c = maskCache[L] { return c }
        var m = [Float](repeating: 0, count: L * L)
        for i in 0..<L {
            for j in (i + 1)..<L where j < L { m[i * L + j] = -1e9 }
        }
        let a = MLXArray(m, [1, 1, L, L])
        maskCache[L] = a
        return a
    }
}

// MARK: - Embedding

final class TokenEmbedding: Module {
    @ModuleInfo(key: "tokenConv") var tokenConv: Conv1d

    init(_ cIn: Int, _ dModel: Int) {
        self._tokenConv.wrappedValue = Conv1d(
            inputChannels: cIn, outputChannels: dModel,
            kernelSize: 3, padding: 0, bias: false)
        super.init()
    }

    // x: [B, L, cIn] -> [B, L, dModel], with circular padding of width 1.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let L = x.shape[1]
        let left = x[0..., (L - 1) ..< L, 0...]   // last step
        let right = x[0..., 0 ..< 1, 0...]        // first step
        let padded = concatenated([left, x, right], axis: 1)
        return tokenConv(padded)
    }
}

final class DataEmbedding: Module {
    @ModuleInfo(key: "value_embedding") var valueEmbedding: TokenEmbedding
    let dModel: Int

    init(_ cIn: Int, _ dModel: Int) {
        self.dModel = dModel
        self._valueEmbedding.wrappedValue = TokenEmbedding(cIn, dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        valueEmbedding(x) + ATBuffers.positional(x.shape[1], dModel)
    }
}

// MARK: - Encoder

/// Multi-head self-attention with causal mask. Mirrors AttentionLayer +
/// AnomalyAttention (output_attention=False path). The internal LayerNorm is
/// applied to the projected output, matching the PyTorch AttentionLayer.
final class AnomalyAttentionLayer: Module {
    let nHeads: Int
    let dModel: Int
    @ModuleInfo(key: "query_projection") var queryProjection: Linear
    @ModuleInfo(key: "key_projection") var keyProjection: Linear
    @ModuleInfo(key: "value_projection") var valueProjection: Linear
    @ModuleInfo(key: "sigma_projection") var sigmaProjection: Linear  // loaded, unused at inference
    @ModuleInfo(key: "out_projection") var outProjection: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ dModel: Int, _ nHeads: Int) {
        self.nHeads = nHeads
        self.dModel = dModel
        self._queryProjection.wrappedValue = Linear(dModel, dModel)
        self._keyProjection.wrappedValue = Linear(dModel, dModel)
        self._valueProjection.wrappedValue = Linear(dModel, dModel)
        self._sigmaProjection.wrappedValue = Linear(dModel, nHeads)
        self._outProjection.wrappedValue = Linear(dModel, dModel)
        self._norm.wrappedValue = LayerNorm(dimensions: dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]
        let H = nHeads
        let d = dModel / H
        let scale = 1.0 / Float(d).squareRoot()

        let q = queryProjection(x).reshaped([B, L, H, d]).swappedAxes(1, 2)  // [B,H,L,d]
        let k = keyProjection(x).reshaped([B, L, H, d]).swappedAxes(1, 2)
        let v = valueProjection(x).reshaped([B, L, H, d]).swappedAxes(1, 2)

        var scores = matmul(q, k.swappedAxes(-1, -2)) * scale  // [B,H,L,L]
        scores = scores + ATBuffers.causalMask(L)
        let attn = softmax(scores, axis: -1)
        var out = matmul(attn, v)                              // [B,H,L,d]
        out = out.swappedAxes(1, 2).reshaped([B, L, H * d])    // [B,L,dModel]
        return norm(outProjection(out))
    }
}

/// One encoder block: residual attention + 1x1-conv feed-forward, both followed
/// by LayerNorm. conv1/conv2 are channels-last Conv1d with kernel size 1.
final class EncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: AnomalyAttentionLayer
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    init(_ dModel: Int, _ nHeads: Int, _ dFF: Int) {
        self._attention.wrappedValue = AnomalyAttentionLayer(dModel, nHeads)
        self._conv1.wrappedValue = Conv1d(inputChannels: dModel, outputChannels: dFF, kernelSize: 1)
        self._conv2.wrappedValue = Conv1d(inputChannels: dFF, outputChannels: dModel, kernelSize: 1)
        self._norm1.wrappedValue = LayerNorm(dimensions: dModel)
        self._norm2.wrappedValue = LayerNorm(dimensions: dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = x + attention(x)
        let n = norm1(a)
        let y = conv2(gelu(conv1(n)))   // channels-last, kernel 1
        return norm2(n + y)
    }
}

final class Encoder: Module {
    @ModuleInfo(key: "attn_layers") var attnLayers: [EncoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ dModel: Int, _ nHeads: Int, _ dFF: Int, _ eLayers: Int) {
        self._attnLayers.wrappedValue = (0..<eLayers).map { _ in EncoderLayer(dModel, nHeads, dFF) }
        self._norm.wrappedValue = LayerNorm(dimensions: dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in attnLayers { h = layer(h) }
        return norm(h)
    }
}

// MARK: - Reconstruction network (trained on-device)

final class ReconNet: Module {
    @ModuleInfo(key: "embedding") var embedding: DataEmbedding
    @ModuleInfo(key: "encoder") var encoder: Encoder
    @ModuleInfo(key: "projection") var projection: Linear

    init(_ cfg: ATConfig) {
        self._embedding.wrappedValue = DataEmbedding(cfg.encIn, cfg.dModel)
        self._encoder.wrappedValue = Encoder(cfg.dModel, cfg.nHeads, cfg.dFF, cfg.eLayers)
        self._projection.wrappedValue = Linear(cfg.dModel, cfg.cOut)
        super.init()
    }

    // x: [B, L, encIn] -> reconstruction [B, L, cOut]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(encoder(embedding(x)))
    }
}

// MARK: - Cross-attention prediction decoder (bundled, inference only)

/// Port of PyTorch nn.MultiheadAttention (batch_first) with packed QKV weights.
final class CrossAttention: Module {
    let nHeads: Int
    let dModel: Int
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray  // [3E, E]
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray      // [3E]
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ dModel: Int, _ nHeads: Int) {
        self.nHeads = nHeads
        self.dModel = dModel
        self._inProjWeight.wrappedValue = zeros([3 * dModel, dModel])
        self._inProjBias.wrappedValue = zeros([3 * dModel])
        self._outProj.wrappedValue = Linear(dModel, dModel)
        super.init()
    }

    // query: [B, Lq, E], context: [B, Lk, E]
    func callAsFunction(_ query: MLXArray, _ context: MLXArray) -> MLXArray {
        let E = dModel
        let H = nHeads
        let d = E / H
        let B = query.shape[0]
        let Lq = query.shape[1]
        let Lk = context.shape[1]

        let wq = inProjWeight[0 ..< E]
        let wk = inProjWeight[E ..< (2 * E)]
        let wv = inProjWeight[(2 * E) ..< (3 * E)]
        let bq = inProjBias[0 ..< E]
        let bk = inProjBias[E ..< (2 * E)]
        let bv = inProjBias[(2 * E) ..< (3 * E)]

        var q = matmul(query, wq.swappedAxes(0, 1)) + bq    // [B,Lq,E]
        var k = matmul(context, wk.swappedAxes(0, 1)) + bk  // [B,Lk,E]
        var v = matmul(context, wv.swappedAxes(0, 1)) + bv

        q = q.reshaped([B, Lq, H, d]).swappedAxes(1, 2)     // [B,H,Lq,d]
        k = k.reshaped([B, Lk, H, d]).swappedAxes(1, 2)
        v = v.reshaped([B, Lk, H, d]).swappedAxes(1, 2)

        let scale = 1.0 / Float(d).squareRoot()
        let scores = matmul(q, k.swappedAxes(-1, -2)) * scale  // [B,H,Lq,Lk]
        let attn = softmax(scores, axis: -1)
        var o = matmul(attn, v)                                // [B,H,Lq,d]
        o = o.swappedAxes(1, 2).reshaped([B, Lq, E])
        return outProj(o)
    }
}

/// FFN matching nn.Sequential(Linear, GELU, Dropout, Linear) — keys "0" and "3".
final class FFN: Module {
    @ModuleInfo(key: "0") var fc1: Linear
    @ModuleInfo(key: "3") var fc2: Linear

    init(_ dModel: Int, _ cOut: Int) {
        self._fc1.wrappedValue = Linear(dModel, dModel)
        self._fc2.wrappedValue = Linear(dModel, cOut)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class PredictionDecoder: Module {
    let horizon: Int
    let dModel: Int
    @ParameterInfo(key: "future_queries") var futureQueries: MLXArray  // [1, horizon, E]
    @ModuleInfo(key: "cross_attn") var crossAttn: CrossAttention
    @ModuleInfo(key: "ffn") var ffn: FFN
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ dModel: Int, _ nHeads: Int, _ horizon: Int, _ cOut: Int) {
        self.horizon = horizon
        self.dModel = dModel
        self._futureQueries.wrappedValue = zeros([1, horizon, dModel])
        self._crossAttn.wrappedValue = CrossAttention(dModel, nHeads)
        self._ffn.wrappedValue = FFN(dModel, cOut)
        self._norm.wrappedValue = LayerNorm(dimensions: dModel)
        super.init()
    }

    // encOut: [B, contextLen, E] -> prediction [B, horizon, cOut]
    func callAsFunction(_ encOut: MLXArray) -> MLXArray {
        let B = encOut.shape[0]
        // Broadcast [1,horizon,E] -> [B,horizon,E] by adding a zero tensor.
        let queries = futureQueries + zeros([B, horizon, dModel])
        let attnOut = crossAttn(queries, encOut)
        let o = norm(attnOut + queries)
        return ffn(o)
    }
}

final class PredNet: Module {
    let contextLen: Int
    @ModuleInfo(key: "embedding") var embedding: DataEmbedding
    @ModuleInfo(key: "encoder") var encoder: Encoder
    @ModuleInfo(key: "predictor") var predictor: PredictionDecoder

    init(_ cfg: ATConfig) {
        self.contextLen = cfg.contextLen
        self._embedding.wrappedValue = DataEmbedding(cfg.encIn, cfg.dModel)
        self._encoder.wrappedValue = Encoder(cfg.dModel, cfg.nHeads, cfg.dFF, cfg.eLayers)
        self._predictor.wrappedValue = PredictionDecoder(cfg.dModel, cfg.nHeads, cfg.horizon, cfg.cOut)
        super.init()
    }

    // x: [B, winSize, encIn] -> prediction of last `horizon` steps [B, horizon, cOut]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ctx = x[0..., 0 ..< contextLen, 0...]
        return predictor(encoder(embedding(ctx)))
    }
}
