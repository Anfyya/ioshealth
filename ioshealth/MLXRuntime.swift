import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Thin entry point for the on-device MLX runtime.
///
/// For now this only proves that the `mlx-swift` package resolves, compiles and
/// links for the iOS build (validated in CI). It will grow into the host for the
/// on-device Anomaly Transformer (personal reconstruction training + prediction
/// base inference).
enum MLXRuntime {
    /// Runs a trivial MLX computation so the linker pulls the framework in and
    /// we can confirm the GPU runtime is reachable. Returns a human string.
    @discardableResult
    static func smokeTest() -> String {
        let a = MLXArray([1, 2, 3, 4] as [Float])
        let b = a * 2 + 1            // [3, 5, 7, 9]
        let last = b[3].item(Float.self)
        return "MLX runtime OK (probe=\(last))"
    }
}
