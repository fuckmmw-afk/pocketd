import Foundation

/// Finds GGUF models on Hugging Face.
///
/// The curated catalogue is eight models that are known to load on a phone. It
/// is the right default and the wrong ceiling: PocketPal ships four curated
/// models and a `+` that reaches all of Hugging Face, and that `+` is the whole
/// difference between "some models" and "any model".
///
/// Two things this deliberately does that a naive search would not. It reads
/// the real byte size of every file from the tree API rather than guessing from
/// the parameter count — the Gemma 4 entry in our own catalogue was wrong by
/// 1.3 GB that way, and the memory fit badge is computed from that number. And
/// it surfaces `gated`, because a gated repository answers 401 to a download
/// this app has no token flow to satisfy, so it must be visible before someone
/// taps rather than after.
public struct HuggingFaceSearch: Sendable {
    public struct Repository: Sendable, Equatable, Identifiable, Codable {
        public let id: String
        public let downloads: Int
        public let likes: Int
        /// HF returns `false`, `"auto"` or `"manual"`. Anything but false needs
        /// a token, which pocketd cannot supply.
        public let gated: Bool
        /// What the hub says the model does — `text-generation`,
        /// `image-text-to-text`, `text-to-speech`… `nil` where the publisher
        /// never said, which is common for GGUF conversions.
        public var pipelineTag: String? = nil
        /// The hub's tags, which carry the same task words when the pipeline
        /// tag is missing.
        public var tags: [String]? = nil

        public var owner: String { id.split(separator: "/").first.map(String.init) ?? id }
        public var name: String { id.split(separator: "/").last.map(String.init) ?? id }

        /// "64K", "1.2M" — never the lakh-style "64L" `.compactName` prints in
        /// an Indian locale.
        public var downloadsText: String { Format.compact(Double(downloads)) }
    }

    public struct File: Sendable, Equatable, Identifiable, Codable {
        public let path: String
        public let sizeBytes: Int64

        public var id: String { path }

        /// `Qwen3-1.7B-Q4_K_M.gguf` -> `Q4_K_M`. Best effort: the convention is
        /// the only thing there is, since GGUF quantisation is not in the API.
        public var quantization: String {
            let stem = path.replacingOccurrences(of: ".gguf", with: "")
            let parts = stem.split(separator: "-")
            for part in parts.reversed() where part.first == "Q" || part.hasPrefix("IQ") || part == "BF16" || part == "F16" {
                return String(part)
            }
            return "unknown"
        }

        public var isProjector: Bool { ModelRecord.isProjector(filename: path) }
    }

    let session: URLSession
    let host: URL

    public init(session: URLSession = .shared, host: URL = URL(string: "https://huggingface.co")!) {
        self.session = session
        self.host = host
    }

    public enum SearchError: Error, Sendable, Equatable {
        case httpStatus(Int)
        case malformed
    }

    /// Repositories matching `query` that publish GGUF files, most-downloaded
    /// first — popularity is the only quality signal available, and on a phone
    /// an obscure broken quantisation costs gigabytes to discover.
    public func repositories(matching query: String, limit: Int = 20) async throws -> [Repository] {
        var components = URLComponents(url: host.appendingPathComponent("api/models"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            .init(name: "search", value: query),
            .init(name: "filter", value: "gguf"),
            .init(name: "sort", value: "downloads"),
            .init(name: "direction", value: "-1"),
            .init(name: "limit", value: String(limit)),
        ]
        let (data, response) = try await session.data(from: components.url!)
        try Self.check(response)

        struct Row: Decodable {
            let id: String
            let downloads: Int?
            let likes: Int?
            let gated: GatedFlag?
            let pipeline_tag: String?
            let tags: [String]?
        }
        guard let rows = try? JSONDecoder().decode([Row].self, from: data) else {
            throw SearchError.malformed
        }
        return rows.map {
            Repository(
                id: $0.id, downloads: $0.downloads ?? 0, likes: $0.likes ?? 0, gated: $0.gated?.isGated ?? false,
                pipelineTag: $0.pipeline_tag, tags: $0.tags
            )
        }
    }

    /// The GGUF files in a repository, with their real sizes.
    public func files(in repository: String) async throws -> [File] {
        let url = host.appendingPathComponent("api/models/\(repository)/tree/main")
        let (data, response) = try await session.data(from: url)
        try Self.check(response)

        struct Row: Decodable {
            let path: String
            let size: Int64?
        }
        guard let rows = try? JSONDecoder().decode([Row].self, from: data) else {
            throw SearchError.malformed
        }
        return rows
            .filter { $0.path.hasSuffix(".gguf") }
            .map { File(path: $0.path, sizeBytes: $0.size ?? 0) }
            .sorted { $0.sizeBytes < $1.sizeBytes }
    }

    /// Turns a chosen repository and file into a catalogue entry.
    ///
    /// The projector is paired automatically when the repository has one: a
    /// vision model downloaded without it loads and then cannot see, which is
    /// the confusing failure rather than the obvious one.
    public static func record(
        repository: String,
        file: File,
        projector: File? = nil,
        contextLength: Int = 4096
    ) -> ModelRecord {
        ModelRecord(
            id: Self.identifier(repository: repository, filename: file.path),
            displayName: file.path.replacingOccurrences(of: ".gguf", with: ""),
            repoID: repository,
            filename: file.path,
            parameters: "—",
            quantization: file.quantization,
            sizeBytes: file.sizeBytes,
            contextLength: contextLength,
            license: "See the model card on Hugging Face",
            projectorFilename: projector?.path,
            projectorSizeBytes: projector?.sizeBytes ?? 0,
            // Unknown rather than no: nobody has checked this template, and
            // claiming either way would be a guess.
            toolSupport: .unknown
        )
    }

    /// A stable id from the repository and file, lowercased and URL-safe so it
    /// can be a filename and a model name on the wire.
    public static func identifier(repository: String, filename: String) -> String {
        let stem = filename.replacingOccurrences(of: ".gguf", with: "")
        let owner = repository.split(separator: "/").first.map(String.init) ?? ""
        let combined = "\(owner)-\(stem)".lowercased()
        return String(combined.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "." ? $0 : "-" })
    }

    static func check(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200...299).contains(http.statusCode) else {
            throw SearchError.httpStatus(http.statusCode)
        }
    }

    // MARK: - What this phone is offered

    /// Matched anywhere in a repository id, case-insensitively. A repo that
    /// advertises its safety training removed is not something Pocketd puts in
    /// front of a person: searching "qwen3" used to list "Uncensored" and
    /// "abliterated" builds above the model itself.
    public static let blockedNameFragments = ["uncensored", "abliterated", "nsfw", "erotic"]

    /// About 4.4 bits a weight: the smallest quantisation still worth talking
    /// to. Below it the hub has files, and they answer badly.
    public static let smallestUsefulBytesPerWeight = 0.55
    /// Q4_K_M, the usual first choice, at about 4.85 bits a weight — what a
    /// repository is ranked by.
    public static let typicalBytesPerWeight = 0.61

    /// Whether a repository may be listed at all: a chat model, and not one
    /// advertising its safety training removed.
    public static func isListable(_ repository: Repository) -> Bool {
        let id = repository.id.lowercased()
        guard !blockedNameFragments.contains(where: { id.contains($0) }) else { return false }
        return isChatModel(repository)
    }

    // MARK: - Only things that talk

    /// What the hub calls a model somebody can talk to. Vision-language models
    /// (`image-text-to-text`) are chat models that can also see.
    public static let chatPipelines: Set<String> = ["text-generation", "image-text-to-text", "text2text-generation", "conversational"]

    /// Hub tasks that are never a brain: speech, transcription, embeddings,
    /// ranking, classification, images.
    public static let nonChatPipelines: Set<String> = [
        "text-to-speech", "text-to-audio", "automatic-speech-recognition", "audio-to-audio", "audio-classification",
        "feature-extraction", "sentence-similarity", "text-ranking", "text-classification", "token-classification",
        "fill-mask", "zero-shot-classification", "text-to-image", "image-to-image", "image-classification",
        "image-feature-extraction", "object-detection", "image-segmentation", "depth-estimation",
        "text-to-video", "voice-activity-detection",
    ]

    /// Words in a repository name that mark something other than a chat
    /// model. Short ones must be a whole word ("asr", not "Kasra"); long ones
    /// may sit inside a longer word ("embeddinggemma", "distil-whisper").
    static let nonChatNameWords: Set<String> = ["tts", "asr", "clip", "vae", "sd"]
    static let nonChatNameFragments = ["whisper", "embedding", "embed", "rerank", "diffusion"]

    /// Whether a repository is something Pocketd can talk to.
    ///
    /// The hub's own pipeline tag decides when it has one: "Qwen3-TTS-GGUF"
    /// is `text-to-speech`, whatever the family name in front of it. Without
    /// one, its tags decide — a repo tagged only `feature-extraction` is an
    /// embedding model. The name is checked either way, because a pipeline
    /// tag is the publisher's claim and some embedding repos claim
    /// `text-generation`.
    public static func isChatModel(_ repository: Repository) -> Bool {
        let name = repository.name.lowercased()
        let words = name.split { !$0.isLetter && !$0.isNumber }.map(String.init)
        if words.contains(where: nonChatNameWords.contains) { return false }
        if nonChatNameFragments.contains(where: { name.contains($0) }) { return false }

        if let pipeline = repository.pipelineTag?.lowercased() {
            return chatPipelines.contains(pipeline)
        }
        let tags = Set((repository.tags ?? []).map { $0.lowercased() })
        if !tags.isDisjoint(with: chatPipelines) { return true }
        return tags.isDisjoint(with: nonChatPipelines)
    }

    /// The parameter count a repository's name states, in billions:
    /// "Qwen3-32B-GGUF" → 32, "Qwen2.5-0.5B-Instruct" → 0.5, "SmolLM2-360M" →
    /// 0.36. The first figure only — "Qwen3-235B-A22B" is a mixture whose
    /// total comes first — and nothing for "8bit" or a name that says no size.
    public static func statedBillions(in name: String) -> Double? {
        let characters = Array(name)
        var index = 0
        while index < characters.count {
            let precededByDigit = index > 0 && (characters[index - 1].isNumber || characters[index - 1] == ".")
            guard characters[index].isNumber, !precededByDigit else { index += 1; continue }
            var end = index
            while end < characters.count, characters[end].isNumber || characters[end] == "." { end += 1 }
            defer { index = end }
            guard end < characters.count, "BbMm".contains(characters[end]) else { continue }
            let followedByLetter = end + 1 < characters.count && characters[end + 1].isLetter
            guard !followedByLetter, let value = Double(String(characters[index..<end])) else { continue }
            return "Mm".contains(characters[end]) ? value / 1000 : value
        }
        return nil
    }

    /// The verdict a repository gets from its name alone, before anyone has
    /// opened it. `.comfortable` when its usual file (Q4_K_M) fits this phone
    /// comfortably; `.tight` when only its smaller files do; `.willNotFit`
    /// when not even the smallest useful one does — search offers only files
    /// that fit comfortably (`Offer.isOffered`), so such a repo would open
    /// onto a page where every size says "Too big for this iPhone". `nil`
    /// when the name says no size, which is a different thing from too big.
    public static func fit(of repository: Repository, for budget: DeviceBudget) -> DeviceBudget.Fit? {
        guard let billions = statedBillions(in: repository.name) else { return nil }
        let smallest = Int64(billions * 1_000_000_000 * smallestUsefulBytesPerWeight)
        let typical = Int64(billions * 1_000_000_000 * typicalBytesPerWeight)
        func verdict(_ bytes: Int64) -> DeviceBudget.Fit {
            budget.fit(for: MemoryEstimate(weightsBytes: bytes, projectorBytes: 0, dimensions: nil, contextTokens: budget.servedContextTokens))
        }
        guard verdict(smallest) == .comfortable else { return .willNotFit }
        return verdict(typical) == .comfortable ? .comfortable : .tight
    }

    /// Search results as this phone should see them: blocked names gone,
    /// repositories with nothing this phone can comfortably run gone, the rest
    /// by fit (usual file comfortable, then only smaller files, then no stated
    /// size) and then by downloads — so phone-sized repos lead, whatever is
    /// most popular on the hub this week.
    public static func ranked(_ repositories: [Repository], for budget: DeviceBudget) -> [Repository] {
        func rank(_ repository: Repository) -> Int {
            switch fit(of: repository, for: budget) {
            case .comfortable: 0
            case .tight: 1
            case nil: 2
            case .willNotFit: 3
            }
        }
        return repositories
            .filter(isListable)
            .filter { fit(of: $0, for: budget) != .willNotFit }
            .enumerated()
            .sorted { lhs, rhs in
                let (l, r) = (rank(lhs.element), rank(rhs.element))
                if l != r { return l < r }
                if lhs.element.downloads != rhs.element.downloads { return lhs.element.downloads > rhs.element.downloads }
                return lhs.offset < rhs.offset
            }
            .map(\.element)
    }

    /// One file as the sizes sheet shows it.
    public struct Offer: Sendable, Equatable, Identifiable {
        public let file: File
        public let fit: DeviceBudget.Fit
        public var id: String { file.id }
        /// Whether the sheet offers it. Only a comfortable fit: nobody has
        /// checked that a file from the hub runs at all, and one that only just
        /// loads leaves no room for the conversation it was downloaded for.
        /// The catalogue's vetted brains keep "Tight fit"; the hub does not.
        public var isOffered: Bool { fit == .comfortable }
        /// What the row says instead of offering a file this phone should not
        /// run.
        public var tooBigNote: String? { isOffered ? nil : "Too big for this iPhone" }
    }

    /// The weights files of a repository with this phone's verdict on each,
    /// smallest first. Projectors are not offered on their own — they are
    /// paired with the weights automatically — and files that are not offered
    /// (`Offer.isOffered`) are left out unless `includingTooBig`, for a screen
    /// that would rather say so than hide them.
    public static func offers(_ files: [File], for budget: DeviceBudget, includingTooBig: Bool = false) -> [Offer] {
        files
            .filter { !$0.isProjector && $0.sizeBytes > 0 }
            .map { file in
                Offer(file: file, fit: budget.fit(for: MemoryEstimate(
                    weightsBytes: file.sizeBytes, projectorBytes: 0, dimensions: nil,
                    contextTokens: budget.servedContextTokens
                )))
            }
            .filter { includingTooBig || $0.isOffered }
            .sorted { $0.file.sizeBytes < $1.file.sizeBytes }
    }

    /// HF sends `false`, `"auto"` or `"manual"` in the same field.
    struct GatedFlag: Decodable {
        let isGated: Bool

        init(from decoder: any Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let flag = try? container.decode(Bool.self) {
                isGated = flag
            } else if let text = try? container.decode(String.self) {
                isGated = text != "false"
            } else {
                isGated = false
            }
        }
    }
}
