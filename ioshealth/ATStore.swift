import Foundation

// MARK: - Local persistence
//
// We persist only the computed results (Codable) — the data range summary and
// the analyzed reports — in Application Support. The trained reconstruction
// model lives in memory for the session; re-importing Health data retrains it.
// Nothing leaves the device.

struct PersistedState: Codable, Sendable {
    let createdAt: Date
    let summary: DatasetSummary
    let reports: [HealthReport]
}

enum ATStore {
    private static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("ioshealth", isDirectory: true)
    }

    private static var stateURL: URL { directory.appendingPathComponent("state.json") }

    static func save(_ state: PersistedState) {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(state)
            try data.write(to: stateURL, options: .atomic)
        } catch {
            // Persistence is best-effort; a failure here must not crash analysis.
            print("[ATStore] save failed: \(error)")
        }
    }

    static func load() -> PersistedState? {
        guard let data = try? Data(contentsOf: stateURL) else { return nil }
        return try? JSONDecoder().decode(PersistedState.self, from: data)
    }

    static func clear() {
        try? FileManager.default.removeItem(at: stateURL)
    }
}
