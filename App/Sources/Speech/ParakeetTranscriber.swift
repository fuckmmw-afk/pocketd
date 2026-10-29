import FluidAudio
import Foundation
import PocketdKit

/// Parakeet speech recognition, on the Neural Engine, through FluidAudio.
///
/// One instance for the app: the microphone and `/v1/audio/transcriptions`
/// share it, so the model is loaded once and requests take turns on the
/// actor. Models live where FluidAudio keeps them (Application Support), and
/// are downloaded from FluidInference's Core ML conversions on Hugging Face.
actor ParakeetTranscriber: SpeechTranscriber {
    static let shared = ParakeetTranscriber()

    private var manager: AsrManager?
    private var loaded: SpeechModel?

    nonisolated static func version(of model: SpeechModel) -> AsrModelVersion {
        model == SpeechModels.parakeetV2 ? .v2 : .v3
    }

    nonisolated static func directory(for model: SpeechModel) -> URL {
        AsrModels.defaultCacheDirectory(for: version(of: model))
    }

    nonisolated static func isInstalled(_ model: SpeechModel) -> Bool {
        AsrModels.modelsExist(at: directory(for: model), version: version(of: model))
    }

    func installedModel(preferring preferred: SpeechModel?) async -> SpeechModel? {
        if let preferred, Self.isInstalled(preferred) { return preferred }
        return SpeechModels.all.first(where: Self.isInstalled)
    }

    /// Downloads the model's Core ML bundles. Resumable in the sense FluidAudio
    /// gives: files already present are not fetched again.
    func download(_ model: SpeechModel, progress: @escaping @Sendable (Double) -> Void) async throws {
        try await AsrModels.download(
            to: Self.directory(for: model),
            version: Self.version(of: model),
            progressHandler: { progress($0.fractionCompleted) }
        )
    }

    func remove(_ model: SpeechModel) async {
        if loaded == model {
            await manager?.cleanup()
            manager = nil
            loaded = nil
        }
        try? FileManager.default.removeItem(at: Self.directory(for: model))
    }

    /// Mono Float32 at 16 kHz, which is what Parakeet reads.
    func transcribe(samples: [Float], model: SpeechModel) async throws -> SpeechTranscription {
        let manager = try await manager(for: model)
        // Under a second is refused by the model as not audio; a short word
        // padded with silence is still that word.
        var padded = samples
        if padded.count < Self.sampleRate { padded += [Float](repeating: 0, count: Self.sampleRate - padded.count) }
        var state = try TdtDecoderState(decoderLayers: Self.version(of: model).decoderLayers)
        do {
            let result = try await manager.transcribe(padded, decoderState: &state)
            return SpeechTranscription(
                text: result.text.trimmingCharacters(in: .whitespacesAndNewlines),
                duration: Double(samples.count) / Double(Self.sampleRate),
                model: model
            )
        } catch {
            throw SpeechError.failed("Speech recognition failed: \(error.localizedDescription)")
        }
    }

    func transcribe(audioFileAt url: URL, model: SpeechModel) async throws -> SpeechTranscription {
        let samples: [Float]
        do {
            samples = try AudioConverter().resampleAudioFile(url)
        } catch {
            throw SpeechError.unreadableAudio(error.localizedDescription)
        }
        guard !samples.isEmpty else { throw SpeechError.unreadableAudio("the file holds no audio") }
        return try await transcribe(samples: samples, model: model)
    }

    /// Microphone audio at whatever rate the hardware gave, to 16 kHz.
    nonisolated static func resampled(_ samples: [Float], from rate: Double) throws -> [Float] {
        try AudioConverter().resample(samples, from: rate)
    }

    private static let sampleRate = 16_000

    private func manager(for model: SpeechModel) async throws -> AsrManager {
        if let manager, loaded == model { return manager }
        guard Self.isInstalled(model) else { throw SpeechError.notInstalled }
        await manager?.cleanup()
        manager = nil
        loaded = nil
        do {
            let models = try await AsrModels.load(from: Self.directory(for: model), version: Self.version(of: model))
            let fresh = AsrManager(config: .default)
            try await fresh.loadModels(models)
            manager = fresh
            loaded = model
            return fresh
        } catch {
            throw SpeechError.failed("\(model.displayName) could not be loaded: \(error.localizedDescription)")
        }
    }
}

/// What the screens show about speech models, and the buttons that change it.
@MainActor
@Observable
final class SpeechModelStore {
    static let shared = SpeechModelStore()

    enum State: Equatable {
        case notInstalled
        case downloading(Double)
        case installed
        case failed(String)
    }

    private(set) var states: [SpeechModel: State] = [:]

    init() {
        refresh()
    }

    func state(of model: SpeechModel) -> State {
        states[model] ?? .notInstalled
    }

    /// The model the microphone uses, when one is installed.
    var activeModel: SpeechModel? {
        SpeechModels.all.first { state(of: $0) == .installed }
    }

    func refresh() {
        for model in SpeechModels.all where !isDownloading(model) {
            states[model] = ParakeetTranscriber.isInstalled(model) ? .installed : .notInstalled
        }
    }

    func download(_ model: SpeechModel) {
        guard !isDownloading(model) else { return }
        states[model] = .downloading(0)
        Task {
            do {
                try await ParakeetTranscriber.shared.download(model) { fraction in
                    Task { @MainActor in
                        if case .downloading = SpeechModelStore.shared.states[model] {
                            SpeechModelStore.shared.states[model] = .downloading(fraction)
                        }
                    }
                }
                states[model] = ParakeetTranscriber.isInstalled(model)
                    ? .installed
                    : .failed("The download finished but the model is incomplete. Try again.")
            } catch {
                states[model] = .failed("The download didn't finish. Check the connection and try again.")
            }
        }
    }

    func remove(_ model: SpeechModel) {
        Task {
            await ParakeetTranscriber.shared.remove(model)
            states[model] = .notInstalled
        }
    }

    private func isDownloading(_ model: SpeechModel) -> Bool {
        if case .downloading = states[model] { return true }
        return false
    }
}
