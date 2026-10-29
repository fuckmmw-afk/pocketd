import Foundation

public struct GenerationRequest: Sendable, Equatable {
    /// The model the client asked for. An engine may reject a request whose
    /// model is not the one currently resident, rather than silently answering
    /// with whatever happens to be loaded.
    public var modelID: String
    public var messages: [ChatMessage]
    public var options: GenerationOptions
    /// The sampler to start from, before `options` is laid over it.
    ///
    /// `nil` — the normal case — means "whatever the engine is configured
    /// with", and only the engine knows that. Pinning it here is for a caller
    /// that wants this one request sampled a specific way regardless of the
    /// server's settings; it does not disable `options`, which still wins for
    /// any field it names. Read it through `resolvedSampling(defaults:)`
    /// rather than directly.
    public var sampling: SamplingParameters?
    /// Defaults to the restricted case on purpose: a call site that forgets to
    /// set this gets the path with no access to personal data, not the
    /// privileged one. Fail closed.
    public var origin: RequestOrigin
    /// Non-nil when the caller runs its own tools. See `ClientTools` for what
    /// that changes; nil — every caller but the Anthropic route — changes
    /// nothing.
    public var clientTools: ClientTools?

    public init(
        modelID: String,
        messages: [ChatMessage],
        options: GenerationOptions = .default,
        sampling: SamplingParameters? = nil,
        origin: RequestOrigin = .network(host: "unknown", port: 0),
        clientTools: ClientTools? = nil
    ) {
        self.modelID = modelID
        self.messages = messages
        self.options = options
        self.sampling = sampling
        self.origin = origin
        self.clientTools = clientTools
    }
}

public enum FinishReason: String, Sendable, Codable {
    case stop
    case length
    case cancelled
}

public struct TokenUsage: Sendable, Codable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int { promptTokens + completionTokens }

    public init(promptTokens: Int = 0, completionTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

public enum GenerationEvent: Sendable, Equatable {
    /// One or more characters of new assistant text. Engines are free to emit
    /// whatever granularity is natural for them; nothing downstream assumes a
    /// chunk is exactly one token.
    case token(String)
    /// A tool is about to run, and after it the model will be re-prompted from
    /// the beginning.
    ///
    /// This exists because a tool turn is slow in a way the user cannot see.
    /// The backend surfaces tool calls only once its token loop has fully
    /// drained, and resuming then re-renders and re-prefills the entire
    /// conversation — two complete prefills per tool turn. At the ~29 tok/s a
    /// phone manages that is seconds of nothing, which reads as a hang. A UI
    /// can show this as "checking…" and clear it on the next `.token`.
    ///
    /// Deliberately not on the wire: neither the OpenAI nor the Ollama
    /// streaming schema has a field for "the server is running a tool on your
    /// behalf", and inventing one breaks clients that parse strictly.
    case toolCallStarted(name: String)
    /// A tool has run and its result is drawable. Emitted before the model has
    /// written a word about it, which is the correct order: the payload is
    /// already authoritative, and the prose that follows is narration over
    /// something the user can already read.
    ///
    /// Deliberately not on the wire, for the same reason as `toolCallStarted`
    /// and one more. Neither streaming schema has a field for it, and a card is
    /// a rendering rather than an answer — a network client that received one
    /// would have to draw it, and the only client here that can is the app on
    /// the phone. The routes drop it; the transcript keeps it.
    case answerCard(AnswerCard)
    case finished(reason: FinishReason, usage: TokenUsage)
}

public enum InferenceError: Error, Sendable, Equatable {
    case noModelLoaded
    case modelNotFound(String)
    case modelMismatch(requested: String, loaded: String)
    case contextExhausted
    case cancelled
    case backend(String)
}

/// One plain sentence per failure, for the person holding the phone.
///
/// This was missing, and its absence was the whole of a P0: every engine
/// failure reached the screen as "The reply could not be completed." with a
/// Dismiss button, because the app reads `errorDescription` and there was none
/// — so a conversation that had outgrown the context failed on every later
/// message with no hint that a new one would work. The sentences say what
/// happened and what to do, in the app's words: no "context", no "model", no
/// identifiers. The word "brain" appears only where the failure can only have
/// come from the Brain screen.
///
/// The server never reads these. The routes match on the case and write their
/// own status codes and messages for other devices, and `.backend` keeps its
/// detail for them — see `detail`.
extension InferenceError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .noModelLoaded:
            "Pocketd isn't ready to answer yet. Try again in a moment."
        case .modelNotFound:
            "This brain isn't on this iPhone any more. Download it again to use it."
        case .modelMismatch:
            "Pocketd was switching over while you asked. Ask again."
        case .contextExhausted:
            "This conversation got too long. Start a new chat to keep going."
        case .cancelled:
            "Stopped."
        case let .backend(detail):
            // The built-in brain's failures are already written for the
            // screen, as a closed, tested set; anything else here is an
            // engine talking to itself, and only the plain sentence goes out.
            // An exact match on that set is what keeps a llama.cpp detail from
            // ever passing for one of them.
            SystemModelFailure.sentences.contains(detail) ? detail : "That one didn't work. Try again."
        }
    }

    /// What went wrong in the engine's own words, for the diagnostics log and
    /// for network clients — never for the screen.
    public var detail: String {
        switch self {
        case .noModelLoaded: "no model loaded"
        case let .modelNotFound(id): "model \(id) not found"
        case let .modelMismatch(requested, loaded): "requested \(requested), loaded \(loaded)"
        case .contextExhausted: "prompt exceeds the context window"
        case .cancelled: "cancelled"
        case let .backend(detail): detail
        }
    }

    /// Whether starting a new conversation is the way out, so a screen can
    /// offer that action next to the sentence.
    public var suggestsNewConversation: Bool { self == .contextExhausted }
}

/// The seam between the HTTP surface and whatever actually runs the weights.
///
/// Everything above this protocol is plain Swift and unit-tested; everything
/// below it is llama.cpp, MLX, or Apple's Foundation Models. Keeping the seam
/// this narrow is what lets the server ship a new backend without the route
/// layer noticing.
public protocol InferenceEngine: Sendable {
    /// Short name for the backend, surfaced in `/health` so a client can tell
    /// what it is talking to. e.g. `"llama.cpp"`.
    var backendName: String { get }

    /// The model currently resident in memory, if any.
    func loadedModel() async -> ModelRecord?

    /// Tokens the loaded client will add to every prompt after the guard has
    /// checked it — tool schemas, mostly. Zero when nothing is registered.
    var promptOverheadTokens: Int { get async }

    /// What a request inherits for every sampling field it does not name.
    ///
    /// Exposed rather than kept private because a client has no other way to
    /// find out what it is being sampled with, and "you get whatever the
    /// operator set" is only an honest answer if the number is reachable.
    var defaultSampling: SamplingParameters { get async }

    /// Bring a model into memory, evicting whatever was there. On a phone there
    /// is only ever room for one.
    func load(model: ModelRecord) async throws

    func unload() async

    /// Stream a completion. The stream must terminate with exactly one
    /// `.finished` event, or throw.
    ///
    /// Cancelling the stream — dropping it, or cancelling the task reading it —
    /// must free the engine promptly, including while the prompt is still
    /// being read, and the next request must work. Stop is the one control a
    /// person has over a brain that has gone on too long, and an engine that
    /// keeps computing after it has been told to stop starves everything that
    /// comes after.
    func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, any Error>

    /// Stops whatever is generating, and everything queued behind it, now.
    ///
    /// For the phone's own decisions that outrank any request in flight —
    /// switching brains, freeing the memory. `load(model:)` deliberately does
    /// not do this itself: the server calls it when another device asks for a
    /// different brain, and a laptop should wait its turn rather than cut off
    /// an answer somebody is reading. Every cancelled stream ends the way a
    /// cancelled stream always does.
    func interrupt() async

    /// How many tokens the prompt for this request comes to, as `generate`
    /// would build it — chat template, tools and all.
    ///
    /// For `count_tokens`, and for refusing a prompt that cannot fit before a
    /// stream has started rather than halfway through one. An engine with a
    /// tokenizer should count with it; the default estimates.
    func countPromptTokens(_ request: GenerationRequest) async throws -> Int
}

public extension InferenceEngine {
    /// An engine with nothing it can stop early stops nothing.
    func interrupt() async {}

    /// Most engines add nothing.
    var promptOverheadTokens: Int { get async { 0 } }

    /// The same estimate `ContextGuard` makes, plus the tools and calls a
    /// client-tool request carries. An engine with no tokenizer of its own
    /// can do no better.
    func countPromptTokens(_ request: GenerationRequest) async throws -> Int {
        var text = request.messages.flatMap { $0.toolCalls }.map { $0.arguments.serialized }.joined()
        if let tools = request.clientTools?.offered {
            text += ChatTemplateInput.nativeTools(tools).map(\.serialized).joined()
        }
        return ContextGuard.estimateTokens(request.messages) + ContextGuard.estimateTokens(text)
            + (await promptOverheadTokens)
    }

    /// An engine with nothing configured is an engine running llama.cpp's own
    /// defaults, which is what an unconfigured `llama-server` would do too.
    var defaultSampling: SamplingParameters { get async { .default } }

    /// Convenience for non-streaming callers: drain the stream into one string.
    func complete(_ request: GenerationRequest) async throws -> (text: String, reason: FinishReason, usage: TokenUsage) {
        var text = ""
        var reason = FinishReason.stop
        var usage = TokenUsage()
        for try await event in try await generate(request) {
            switch event {
            case .token(let chunk):
                text += chunk
            case .toolCallStarted:
                // A buffered caller sees only the answer. The announcement
                // exists to fill the silence of a stream, and there is no
                // silence to fill here.
                break
            case .answerCard:
                // This convenience returns a string, and a card is not one.
                // A caller that wants the cards has to read the stream.
                break
            case .finished(let r, let u):
                reason = r
                usage = u
            }
        }
        return (text, reason, usage)
    }
}
