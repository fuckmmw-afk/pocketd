import Foundation

// MARK: - A query that names one repository

public extension HuggingFaceSearch {
    /// What a person typed, as the hub's search wants it.
    ///
    /// A link copied out of a browser — `https://huggingface.co/owner/name`,
    /// `hf.co/owner/name/tree/main`, a file's `/blob/` or `/resolve/` URL —
    /// is cut down to `owner/name`. The hub's search matches ids, not URLs, so
    /// pasting the page you were reading used to find nothing at all, even for
    /// a GGUF repository that would have been listed by name.
    static func normalizedQuery(_ raw: String) -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://", "http://"] where text.lowercased().hasPrefix(prefix) {
            text.removeFirst(prefix.count)
        }
        for host in ["www.huggingface.co/", "huggingface.co/", "hf.co/", "hf://models/", "hf://"]
        where text.lowercased().hasPrefix(host) {
            text.removeFirst(host.count)
            if text.hasPrefix("models/") { text.removeFirst("models/".count) }
            if let cut = text.firstIndex(where: { $0 == "?" || $0 == "#" }) { text = String(text[..<cut]) }
            let parts = text.split(separator: "/", omittingEmptySubsequences: true)
            if parts.count >= 2 { text = "\(parts[0])/\(parts[1])" }
            break
        }
        return text
    }

    /// `owner/name` when the query is exactly one repository id.
    static func repositoryID(in query: String) -> String? {
        let parts = query.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return nil }
        let allowed = { (character: Character) in
            character.isASCII && (character.isLetter || character.isNumber || "-_.".contains(character))
        }
        guard parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(allowed) }) else { return nil }
        return query
    }

    /// One repository, as the hub describes it, whatever it contains.
    struct Details: Sendable, Equatable {
        public var repository: Repository
        /// Whether it publishes any GGUF file — the only kind Pocketd runs.
        public var hasGGUF: Bool
        /// The weight formats its tags name: `coreml`, `mlx`, `onnx`…
        public var formats: [String]

        public init(repository: Repository, hasGGUF: Bool, formats: [String]) {
            self.repository = repository
            self.hasGGUF = hasGGUF
            self.formats = formats
        }
    }

    /// The repository with this id, or nil when there is none — the hub
    /// answers 401 rather than 404 for a repository that does not exist.
    func details(of id: String) async throws -> Details? {
        let url = host.appendingPathComponent("api/models/\(id)")
        let (data, response) = try await session.data(from: url)
        if let http = response as? HTTPURLResponse, [401, 404].contains(http.statusCode) { return nil }
        try Self.check(response)

        struct Sibling: Decodable { let rfilename: String }
        struct Row: Decodable {
            let id: String
            let downloads: Int?
            let likes: Int?
            let gated: GatedFlag?
            let pipeline_tag: String?
            let tags: [String]?
            let siblings: [Sibling]?
        }
        guard let row = try? JSONDecoder().decode(Row.self, from: data) else {
            throw SearchError.malformed
        }
        let tags = row.tags ?? []
        let hasGGUF = tags.contains("gguf") || (row.siblings ?? []).contains { $0.rfilename.hasSuffix(".gguf") }
        return Details(
            repository: Repository(
                id: row.id, downloads: row.downloads ?? 0, likes: row.likes ?? 0,
                gated: row.gated?.isGated ?? false, pipelineTag: row.pipeline_tag, tags: tags
            ),
            hasGGUF: hasGGUF,
            formats: Self.weightFormats.filter { tags.contains($0) }
        )
    }

    /// The tags the hub uses for weight formats Pocketd cannot load.
    static let weightFormats = ["coreml", "mlx", "onnx", "safetensors", "tflite", "openvino", "executorch"]

    /// Why a repository someone named exactly is not in the results, in one
    /// sentence for the screen — or nil when it would be offered.
    ///
    /// Search lists only GGUF chat models a phone can hold, and "Nothing for…"
    /// was the whole answer for anything else: a Core ML speech model looked
    /// exactly like a typo.
    static func whyNotOffered(_ details: Details, for budget: DeviceBudget) -> String? {
        let repository = details.repository
        let task = taskDescription(repository)
        if !details.hasGGUF {
            let format = details.formats.first.map { " in \(formatName($0)) format" } ?? ""
            if let task {
                return "This is a \(task) model\(format). Pocketd runs GGUF chat models with llama.cpp, so it can't use this one."
            }
            let stated = format.isEmpty ? "" : " — it's\(format)"
            return "This repository has no GGUF files\(stated). Pocketd runs GGUF with llama.cpp: search for a GGUF conversion of it instead."
        }
        let id = repository.id.lowercased()
        if blockedNameFragments.contains(where: { id.contains($0) }) {
            return "Pocketd doesn't offer models advertised with their safety training removed."
        }
        if !isChatModel(repository) {
            return task.map { "This is a \($0) model, not a chat model Pocketd can talk to." }
                ?? "This isn't a chat model, so Pocketd can't talk to it."
        }
        if fit(of: repository, for: budget) == .willNotFit {
            return "Even its smallest useful file is too big for this iPhone."
        }
        return nil
    }

    /// "speech recognition", "text-to-speech"… from the hub's pipeline tag,
    /// for anything that is not a chat model.
    private static func taskDescription(_ repository: Repository) -> String? {
        guard let pipeline = repository.pipelineTag?.lowercased(), !chatPipelines.contains(pipeline) else { return nil }
        switch pipeline {
        case "automatic-speech-recognition": return "speech recognition"
        case "feature-extraction", "sentence-similarity": return "embedding"
        case "text-to-image": return "image generation"
        default: return pipeline.replacingOccurrences(of: "-", with: " ")
        }
    }

    private static func formatName(_ tag: String) -> String {
        switch tag {
        case "coreml": "Core ML"
        case "mlx": "MLX"
        case "onnx": "ONNX"
        case "safetensors": "safetensors"
        case "tflite": "TensorFlow Lite"
        case "openvino": "OpenVINO"
        case "executorch": "ExecuTorch"
        default: tag
        }
    }
}
