import AVFoundation
import Foundation
import Observation
import PocketdKit
import Speech

/// On-device speech-to-text for a microphone button on the Ask and Today
/// screens.
///
/// Everything here runs on-device — `requiresOnDeviceRecognition` is set to
/// `true` once, in `start()`, and nothing in this file ever sets it back to
/// `false`. That is not a performance choice, it is the whole point of the
/// class: this app's argument to the user is that nothing they say to it
/// leaves the phone, and `SFSpeechRecognizer`'s automatic fallback to a
/// server-side recognizer whenever the on-device one is unavailable would
/// break that argument silently rather than loudly. `isSupported` exists so a
/// caller can find out *before* offering the microphone at all, which is what
/// keeps this class from ever having to choose between refusing to listen and
/// quietly cheating.
///
/// On-device recognition does not exist for every locale, and — worth
/// knowing before this ever looks broken in testing — is unavailable in the
/// iOS Simulator. `isSupported` reports `false` there; that is expected, not
/// a bug in this class.
@MainActor
@Observable
final class VoiceInput {

    /// Speech recognition and microphone permission, collapsed into one
    /// value.
    ///
    /// iOS shows these as two separate system prompts with two separate
    /// outcomes, and a caller that only wants to know whether the mic button
    /// should be enabled has no use for that distinction: either both are
    /// granted and listening can start, or one of them is not and it can't.
    /// `.unknown` covers "never asked" and "asked and still undetermined"
    /// alike, since both mean the same thing to a Start button — ask first.
    enum Authorization {
        case unknown
        case granted
        case denied
    }

    private(set) var transcript: String = ""
    private(set) var isListening: Bool = false

    /// 0...1, smoothed. A waveform redrawn straight from a buffer's raw RMS
    /// jumps between silence and full scale on every unvoiced consonant; the
    /// exponential smoothing in `updateLevel` is what turns that into
    /// something that reads as speech rather than static.
    private(set) var level: Float = 0

    /// One plain sentence for whoever is looking at the screen, or `nil`.
    /// Never a raw error description — `SFSpeechRecognitionTask` and
    /// `AVAudioEngine` report failures as framework NSErrors written for a
    /// console, and `plainMessage(forDescription:)` is the only place that
    /// text is looked at, and only to classify it.
    private(set) var error: String?

    private(set) var authorization: Authorization = .unknown

    /// Whether this device can even try, checked live rather than cached: a
    /// recognizer for the current locale exists and can run without reaching
    /// a server. Apple's own guidance is to re-check per session rather than
    /// remember the answer, and the answer genuinely can change — a language
    /// pack finishes downloading, or Settings > General > Language changes
    /// under the app.
    var isSupported: Bool {
        if Self.parakeet != nil { return true }
        guard let recognizer = SFSpeechRecognizer() else { return false }
        return recognizer.supportsOnDeviceRecognition
    }

    /// The downloaded speech model, when there is one. With it the microphone
    /// records and Parakeet transcribes when the button is let go, which needs
    /// only the microphone permission; without it Apple's on-device
    /// recogniser listens live, as it always has.
    private static var parakeet: SpeechModel? { SpeechModelStore.shared.activeModel }

    /// Audio recorded for Parakeet, and the transcription of it once let go.
    private let recording = RecordedAudio()
    private var transcription: Task<Void, Never>?

    private let audioEngine = AVAudioEngine()
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var autoStopTask: Task<Void, Never>?

    /// Identifies one `start()` session. The recognizer can be slow to wind
    /// down after `stop()`, and its last callback for an old session can
    /// arrive after a new one has already begun; every callback below is
    /// stamped with the token that was current when it was scheduled and is
    /// dropped once that is no longer this one — the same trade `LlamaEngine`
    /// makes with its own generation token, so a stale answer never overwrites
    /// the one running now.
    private var token = UUID()

    /// Apple bounds a recognition request to about 60 seconds and ends it
    /// unasked, with an error, once it does. Stopping just ahead of that
    /// turns an internal limit nobody was told about into the same clean stop
    /// as tapping the mic again — whatever was heard is kept, and nothing
    /// that looks like a failure reaches `error`.
    private static let maxListenDuration: Duration = .seconds(58)

    // MARK: - Permission

    /// Puts both system prompts to the user if either has never been shown,
    /// and reports whether listening can start. iOS shows each prompt at most
    /// once; calling this again after a decision re-reads that decision
    /// instead of asking again.
    func requestAuthorization() async -> Bool {
        if Self.parakeet != nil {
            _ = await AVAudioApplication.requestRecordPermission()
            let current = Self.currentAuthorization()
            authorization = current
            return current == .granted
        }
        let speechStatus = await Self.speechAuthorization()
        // Asking for the microphone after Speech Recognition was refused
        // would show a permission sheet for a capability that refusal already
        // makes useless, right after the user just said no to the thing next
        // to it.
        guard speechStatus == .authorized else {
            authorization = .denied
            return false
        }
        _ = await AVAudioApplication.requestRecordPermission()
        let current = Self.currentAuthorization()
        authorization = current
        return current == .granted
    }

    /// Re-derives `authorization` from the two system permissions, without
    /// prompting. A user who flips the Microphone switch off in Settings and
    /// comes back should see that the next time this is checked, not only the
    /// next time this class happens to ask.
    private static func currentAuthorization() -> Authorization {
        if parakeet != nil {
            switch AVAudioApplication.shared.recordPermission {
            case .granted: return .granted
            case .denied: return .denied
            default: return .unknown
            }
        }
        switch (SFSpeechRecognizer.authorizationStatus(), AVAudioApplication.shared.recordPermission) {
        case (.authorized, .granted):
            return .granted
        case (.denied, _), (.restricted, _), (_, .denied):
            return .denied
        default:
            return .unknown
        }
    }

    // MARK: - Listening

    func start() {
        guard !isListening else { return }
        error = nil

        guard isSupported else {
            error = "On-device speech recognition isn't available on this device."
            return
        }

        let liveAuthorization = Self.currentAuthorization()
        authorization = liveAuthorization
        guard liveAuthorization == .granted else {
            error = Self.refusedSentence
            return
        }

        if let model = Self.parakeet {
            startRecording(for: model)
            return
        }

        guard let recognizer = SFSpeechRecognizer(), recognizer.isAvailable else {
            error = Self.unavailableSentence
            return
        }

        let session = AVAudioSession.sharedInstance()
        do {
            // .measurement is Apple's own recommendation for feeding a speech
            // recognizer: it is the one mode that turns off the system's
            // voice-tuned signal processing, which a recognizer trained on
            // unprocessed audio does not expect.
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            self.error = "Couldn't start the microphone. Try again."
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        // Never set to false. See the type-level comment — this one line is
        // the whole privacy guarantee this class makes.
        request.requiresOnDeviceRecognition = true

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0) // defensive: a previous session's tap should already be gone

        let sessionToken = UUID()
        token = sessionToken

        // The tap block runs on the audio engine's real-time render thread,
        // never the main actor — see `tapBlock`, which builds it outside this
        // actor and holds `self` weakly (the block is stored on `audioEngine`,
        // which `self` owns).
        inputNode.installTap(
            onBus: 0,
            bufferSize: 1024,
            format: format,
            block: Self.tapBlock(appendingTo: request, token: sessionToken, owner: self)
        )

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            try? session.setActive(false, options: [.notifyOthersOnDeactivation])
            self.error = "Couldn't start the microphone. Try again."
            return
        }

        self.recognizer = recognizer
        self.request = request
        transcript = ""
        isListening = true

        task = recognizer.recognitionTask(
            with: request,
            resultHandler: Self.resultHandler(token: sessionToken, owner: self)
        )

        scheduleAutoStop(for: sessionToken)
    }

    /// Ends listening and keeps whatever has been transcribed so far.
    func stop() {
        guard isListening else { return }
        isListening = false
        level = 0
        autoStopTask?.cancel()
        autoStopTask = nil
        if let model = recording.model {
            teardownAudioEngineAndSession()
            finishRecording(for: model)
            return
        }
        // Ends the audio rather than cancelling the task, so the recognizer
        // still hands back a final result for whatever was already said —
        // cancelling is what would throw away the last word spoken before the
        // tap landed on Stop. `request`, `task` and `recognizer` are
        // deliberately left in place for that final callback to arrive
        // against; `handle` clears them once it does.
        request?.endAudio()
        teardownAudioEngineAndSession()
    }

    /// Clears the transcript and any error for a fresh turn. Leaves
    /// `isListening` untouched — this is for wiping the slate between uses of
    /// the same instance, not for stopping one that is still listening.
    func reset() {
        transcript = ""
        error = nil
    }

    /// The transcript once it is final. Apple's recogniser sends its last
    /// word just after `stop()`, so that path waits a moment as the composer
    /// always did; Parakeet's transcription is awaited.
    func finalTranscript() async -> String {
        if let transcription {
            await transcription.value
        } else {
            try? await Task.sleep(for: .milliseconds(400))
        }
        return transcript
    }

    // MARK: - Parakeet

    private func startRecording(for model: SpeechModel) {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            self.error = "Couldn't start the microphone. Try again."
            return
        }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        let sessionToken = UUID()
        token = sessionToken
        recording.start(model: model, sampleRate: format.sampleRate)

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1024,
            format: format,
            block: Self.recordingTapBlock(into: recording, token: sessionToken, owner: self)
        )
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            try? session.setActive(false, options: [.notifyOthersOnDeactivation])
            _ = recording.finish()
            self.error = "Couldn't start the microphone. Try again."
            return
        }
        transcript = ""
        transcription = nil
        isListening = true
        scheduleAutoStop(for: sessionToken)
    }

    private func finishRecording(for model: SpeechModel) {
        let (samples, sampleRate) = recording.finish()
        let sessionToken = token
        transcription = Task { [weak self] in
            do {
                let audio = try ParakeetTranscriber.resampled(samples, from: sampleRate)
                let result = try await ParakeetTranscriber.shared.transcribe(samples: audio, model: model)
                guard let self, self.token == sessionToken else { return }
                self.transcript = result.text
                if result.text.isEmpty { self.error = "Pocketd didn't hear anything. Try again." }
            } catch {
                guard let self, self.token == sessionToken else { return }
                self.error = "Pocketd couldn't turn that into words. Try again."
            }
        }
    }

    /// Copies each buffer's first channel out on the render thread and
    /// reports its level, like `tapBlock` does for Apple's recogniser.
    private nonisolated static func recordingTapBlock(
        into recording: RecordedAudio,
        token: UUID,
        owner: VoiceInput
    ) -> AVAudioNodeTapBlock {
        { [weak owner] buffer, _ in
            if let channel = buffer.floatChannelData {
                recording.append(UnsafeBufferPointer(start: channel[0], count: Int(buffer.frameLength)))
            }
            let level = rmsLevel(of: buffer)
            Task { @MainActor [weak owner] in
                owner?.updateLevel(level, for: token)
            }
        }
    }

    deinit {
        // A defensive net for a caller that goes away without calling
        // `stop()` first. Both calls are idempotent and safe even when
        // nothing is running.
        audioEngine.inputNode.removeTap(onBus: 0)
        if audioEngine.isRunning { audioEngine.stop() }
    }

    // MARK: - Callbacks the system makes on its own threads

    // Every closure below is handed to the system and called back on a
    // thread of the system's choosing — the audio engine's real-time render
    // thread, the recogniser's queue, TCC's reply queue. Written inline in
    // this main-actor class, Swift 6 makes each one main-actor-isolated and
    // checks that at run time: tapping "Don't Allow" on the Speech
    // Recognition prompt crashed the app ("BUG IN CLIENT OF LIBDISPATCH:
    // Block was expected to execute on queue com.apple.main-thread"), and the
    // tap block would have done the same on the first buffer of audio. Built
    // in `nonisolated` functions instead, so the closures belong to no actor,
    // and only plain values hop to the main actor.

    private nonisolated static func speechAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    /// The block `installTap` calls per buffer: appends it to the request and
    /// reports its level. `owner` is held weakly — the block is stored on
    /// `audioEngine`, which `owner` owns, so a strong capture is a cycle.
    private nonisolated static func tapBlock(
        appendingTo request: SFSpeechAudioBufferRecognitionRequest,
        token: UUID,
        owner: VoiceInput
    ) -> AVAudioNodeTapBlock {
        { [weak owner] buffer, _ in
            request.append(buffer)
            let level = rmsLevel(of: buffer)
            Task { @MainActor [weak owner] in
                owner?.updateLevel(level, for: token)
            }
        }
    }

    /// The recogniser's result handler. Neither `SFSpeechRecognitionResult`
    /// nor `Error` is Sendable, so everything worth keeping is read out here,
    /// on whatever thread the recogniser calls back on.
    private nonisolated static func resultHandler(
        token: UUID,
        owner: VoiceInput
    ) -> (SFSpeechRecognitionResult?, (any Error)?) -> Void {
        { [weak owner] result, error in
            let transcript = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let description = error.map { String(describing: $0) }
            Task { @MainActor [weak owner] in
                owner?.handle(token: token, transcript: transcript, isFinal: isFinal, errorDescription: description)
            }
        }
    }

    // MARK: - Session bookkeeping

    private func scheduleAutoStop(for sessionToken: UUID) {
        autoStopTask?.cancel()
        autoStopTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: Self.maxListenDuration)
            guard !Task.isCancelled else { return }
            guard let self, self.token == sessionToken else { return }
            self.stop()
        }
    }

    private func handle(token callbackToken: UUID, transcript: String?, isFinal: Bool, errorDescription: String?) {
        guard callbackToken == token else { return } // a stale session's last word, arriving after the next one started
        if let transcript {
            self.transcript = transcript
        }
        guard let errorDescription else {
            if isFinal {
                task = nil
                request = nil
                recognizer = nil
            }
            return
        }
        // A cancellation or "no speech" error that follows our own `stop()`
        // is expected — `request.endAudio()` there is exactly what produces
        // one — and must never overwrite a transcript that already holds the
        // user's words with an error message.
        guard isListening else { return }
        isListening = false
        level = 0
        autoStopTask?.cancel()
        autoStopTask = nil
        teardownAudioEngineAndSession()
        task = nil
        request = nil
        recognizer = nil
        self.error = Self.plainMessage(forDescription: errorDescription)
    }

    private func updateLevel(_ raw: Float, for callbackToken: UUID) {
        guard callbackToken == token, isListening else { return }
        // Exponential smoothing: buffers arrive roughly 40 times a second at
        // this tap's buffer size, and a waveform that jumps straight to each
        // new value looks like noise rather than a level.
        let smoothing: Float = 0.3
        level += (raw - level) * smoothing
    }

    private func teardownAudioEngineAndSession() {
        audioEngine.inputNode.removeTap(onBus: 0)
        if audioEngine.isRunning { audioEngine.stop() }
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    // MARK: - Translating the framework

    /// A buffer's loudness on a log scale, folded to 0...1.
    ///
    /// `nonisolated` and `static`: this runs on the audio engine's own
    /// real-time thread, once per buffer, and must never wait on the main
    /// actor just to compute a square root.
    private nonisolated static func rmsLevel(of buffer: AVAudioPCMBuffer) -> Float {
        guard let channel = buffer.floatChannelData else { return 0 }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return 0 }
        let samples = channel[0]
        var sumOfSquares: Float = 0
        for index in 0..<frameCount {
            let sample = samples[index]
            sumOfSquares += sample * sample
        }
        let rms = sqrt(sumOfSquares / Float(frameCount))
        // -50 dB is close to silence for a phone microphone and 0 dB is
        // clipping; the range between them is what actually moves on a
        // waveform. Below the floor this reads as flat zero rather than a
        // jitter nobody asked to see.
        let decibels = 20 * log10(max(rms, 0.00001))
        let minDecibels: Float = -50
        let normalized = (decibels - minDecibels) / -minDecibels
        return min(max(normalized, 0), 1)
    }

    /// Every error this class can produce, boiled down to one sentence a
    /// person mid-conversation can act on. Matched the same way
    /// `AppModel.friendlyMessage` matches server errors: by substring, on the
    /// framework's own description, because neither `Speech` nor
    /// `AVFoundation` gives a typed reason here worth switching over.
    private static func plainMessage(forDescription text: String) -> String {
        let text = text.lowercased()
        if text.contains("permission") || text.contains("denied") || text.contains("not authorized") {
            return refusedSentence
        }
        if text.contains("no speech") || text.contains("code=1110") {
            return "Pocketd didn't hear anything. Try again."
        }
        // The on-device recogniser itself: its language files are missing or
        // being replaced, or Dictation is off. Seen on the simulator as
        // kLSRErrorDomain 300, "Failed to initialize recognizer", on every
        // attempt — "Try again" there only fails again, so it says why instead.
        if text.contains("klsrerrordomain") || text.contains("initialize recognizer")
            || text.contains("asset") || text.contains("dictation") || text.contains("unavailable") {
            return unavailableSentence
        }
        if text.contains("audio") || text.contains("microphone") {
            return "Pocketd couldn't use the microphone. Try again."
        }
        return "Pocketd couldn't turn that into words. Try again."
    }

    /// Which permission is missing, by name — the Settings switch somebody has
    /// to find is either Microphone or Speech Recognition, and saying both
    /// sends them looking for one that is already on.
    static var refusedSentence: String {
        if parakeet == nil, SFSpeechRecognizer.authorizationStatus() != .authorized {
            return "Speech Recognition is off for Pocketd in iOS Settings."
        }
        if AVAudioApplication.shared.recordPermission != .granted {
            return "Microphone is off for Pocketd in iOS Settings."
        }
        return "Pocketd can't listen right now. Try again."
    }

    static let unavailableSentence = "Speech recognition isn't available on this iPhone right now."

    /// Whether iOS has never asked for Speech Recognition — the moment a
    /// primer goes up, before the system's own prompt.
    static var hasNeverAsked: Bool {
        (parakeet == nil && SFSpeechRecognizer.authorizationStatus() == .notDetermined)
            || AVAudioApplication.shared.recordPermission == .undetermined
    }
}

/// Microphone samples collected for Parakeet, written from the audio render
/// thread and read on the main actor once recording stops.
private final class RecordedAudio: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private var sampleRate: Double = 16_000
    private var recordingFor: SpeechModel?

    /// The model being recorded for, while a recording is open.
    var model: SpeechModel? { lock.withLock { recordingFor } }

    func start(model: SpeechModel, sampleRate: Double) {
        lock.withLock {
            recordingFor = model
            self.sampleRate = sampleRate
            samples = []
            samples.reserveCapacity(Int(sampleRate) * 10)
        }
    }

    func append(_ buffer: UnsafeBufferPointer<Float>) {
        lock.withLock { samples.append(contentsOf: buffer) }
    }

    func finish() -> (samples: [Float], sampleRate: Double) {
        lock.withLock {
            defer { samples = []; recordingFor = nil }
            return (samples, sampleRate)
        }
    }
}
