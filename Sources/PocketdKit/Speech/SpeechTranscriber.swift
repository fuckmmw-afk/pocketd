import Foundation

/// A speech recognition model Pocketd can download and run with Core ML.
///
/// Not a `ModelRecord`: these are not brains. They do not load into the
/// llama.cpp engine, do not count against its memory, and are never chosen
/// for a conversation. They turn audio into text, for the microphone and for
/// `POST /v1/audio/transcriptions`.
public struct SpeechModel: Sendable, Equatable, Identifiable, Hashable {
    /// Also the `model` a transcription request may name.
    public let id: String
    /// The Hugging Face repository the weights come from.
    public let repository: String
    public let displayName: String
    public let languages: String
    /// What the download comes to, roughly, for the button that offers it.
    public let approximateBytes: Int64

    public init(id: String, repository: String, displayName: String, languages: String, approximateBytes: Int64) {
        self.id = id
        self.repository = repository
        self.displayName = displayName
        self.languages = languages
        self.approximateBytes = approximateBytes
    }
}

public enum SpeechModels {
    /// NVIDIA Parakeet TDT 0.6B v3, converted to Core ML by FluidInference.
    /// Multilingual, and the one offered by default.
    public static let parakeetV3 = SpeechModel(
        id: "parakeet-tdt-0.6b-v3",
        repository: "FluidInference/parakeet-tdt-0.6b-v3-coreml",
        displayName: "Parakeet v3",
        languages: "25 European languages, Russian and English among them",
        approximateBytes: 480_000_000
    )

    /// The English-only predecessor, better on rare English words.
    public static let parakeetV2 = SpeechModel(
        id: "parakeet-tdt-0.6b-v2",
        repository: "FluidInference/parakeet-tdt-0.6b-v2-coreml",
        displayName: "Parakeet v2 (English)",
        languages: "English",
        approximateBytes: 480_000_000
    )

    public static let all = [parakeetV3, parakeetV2]

    /// The model a Hugging Face repository id is, case-insensitively.
    public static func model(forRepository id: String) -> SpeechModel? {
        all.first { $0.repository.lowercased() == id.lowercased() }
    }

    /// The model a transcription request names. Clients written for OpenAI
    /// send `whisper-1`, and refusing them would refuse every such client, so
    /// any name that is not one of ours means "whichever is installed".
    public static func model(named name: String?) -> SpeechModel? {
        guard let name else { return nil }
        return all.first { $0.id == name || $0.repository.lowercased() == name.lowercased() }
    }
}

public struct SpeechTranscription: Sendable, Equatable {
    public var text: String
    /// Seconds of audio, when the transcriber knows.
    public var duration: Double?
    public var model: SpeechModel

    public init(text: String, duration: Double? = nil, model: SpeechModel) {
        self.text = text
        self.duration = duration
        self.model = model
    }
}

public enum SpeechError: Error, Sendable, Equatable {
    /// No speech model has been downloaded.
    case notInstalled
    /// The bytes are not audio this phone can decode.
    case unreadableAudio(String)
    case failed(String)
}

/// Audio in, text out. The app's implementation runs Parakeet on the Neural
/// Engine; tests use a fake.
public protocol SpeechTranscriber: Sendable {
    /// The model that would answer a request naming `preferred`, or nil when
    /// none is installed.
    func installedModel(preferring preferred: SpeechModel?) async -> SpeechModel?

    func transcribe(audioFileAt url: URL, model: SpeechModel) async throws -> SpeechTranscription
}
