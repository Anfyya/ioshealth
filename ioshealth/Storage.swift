import Foundation

struct StoredModelSnapshot: Codable, Sendable {
    let bundle: UserModelBundle
    let summary: DatasetSummary
    let reports: [AnomalyReport]
}

struct LocalModelStore {
    private var folder: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("HealthAnomaly", isDirectory: true)
    }
    private var fileURL: URL { folder.appendingPathComponent("UserModelSnapshot.json") }

    func save(bundle: UserModelBundle, summary: DatasetSummary, reports: [AnomalyReport]) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        let snapshot = StoredModelSnapshot(bundle: bundle, summary: summary, reports: reports)
        try encoder.encode(snapshot).write(to: fileURL, options: [.atomic])
    }

    func load() throws -> StoredModelSnapshot {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(StoredModelSnapshot.self, from: Data(contentsOf: fileURL))
    }
}
