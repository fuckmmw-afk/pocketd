import Foundation
import llama
import PocketdKit

/// A GGUF opened for its tokenizer alone — `vocab_only`, so no weights are
/// mapped and nothing is uploaded to the GPU.
///
/// Its own model rather than the resident client's, because the two are used
/// from different places at different times: the server counts a queued
/// request's prompt while another request is generating, and the resident
/// client can be rebuilt (and its model freed) in between. The vocabulary is
/// identical either way; it is the same file.
final class LlamaVocabulary: @unchecked Sendable {
    private let model: OpaquePointer
    let vocab: OpaquePointer
    /// llama.cpp's tokenizer keeps caches; one caller at a time.
    private let lock = NSLock()

    init?(url: URL) {
        var parameters = llama_model_default_params()
        parameters.vocab_only = true
        guard let model = llama_model_load_from_file(url.path(percentEncoded: false), parameters) else {
            return nil
        }
        guard let vocab = llama_model_get_vocab(model) else {
            llama_model_free(model)
            return nil
        }
        self.model = model
        self.vocab = vocab
    }

    deinit {
        llama_model_free(model)
    }

    /// The template from the file's own metadata.
    var chatTemplate: String? {
        guard let text = llama_model_chat_template(model, nil) else { return nil }
        return String(cString: text)
    }

    /// Text to tokens, with special tokens in the text read as special tokens
    /// and nothing added — the chat template already wrote the BOS it wants.
    /// The same settings LocalLLMClient decodes a rendered prompt with.
    func tokenize(_ text: String) -> [llama_token] {
        lock.lock()
        defer { lock.unlock() }
        let length = Int32(text.utf8.count)
        // Capacities are taken before the call: reading `tokens.count` while
        // `&tokens` is passed is an overlapping access.
        var capacity = length + 8
        var tokens = [llama_token](repeating: 0, count: Int(capacity))
        var count = llama_tokenize(vocab, text, length, &tokens, capacity, false, true)
        if count < 0 {
            capacity = -count
            tokens = [llama_token](repeating: 0, count: Int(capacity))
            count = llama_tokenize(vocab, text, length, &tokens, capacity, false, true)
        }
        return Array(tokens.prefix(Int(max(0, count))))
    }

    /// The bytes one token stands for. Special tokens are rendered, because
    /// several families' tool-call markers — Qwen's `<tool_call>` among them —
    /// are special tokens, and dropping them would leave a call unparseable.
    func piece(_ token: llama_token) -> [UInt8] {
        lock.lock()
        defer { lock.unlock() }
        var capacity: Int32 = 64
        var buffer = [CChar](repeating: 0, count: Int(capacity))
        var count = llama_token_to_piece(vocab, token, &buffer, capacity, 0, true)
        if count < 0 {
            capacity = -count
            buffer = [CChar](repeating: 0, count: Int(capacity))
            count = llama_token_to_piece(vocab, token, &buffer, capacity, 0, true)
        }
        return buffer.prefix(Int(max(0, count))).map { UInt8(bitPattern: $0) }
    }

    func isEndOfGeneration(_ token: llama_token) -> Bool {
        llama_vocab_is_eog(vocab, token)
    }

    /// The names templates use for the special tokens.
    var specialTokens: [String: String] {
        func text(_ token: llama_token) -> String {
            guard token >= 0, let text = llama_vocab_get_text(vocab, token) else { return "" }
            return String(cString: text)
        }
        return [
            "bos_token": text(llama_vocab_bos(vocab)),
            "eos_token": text(llama_vocab_eos(vocab)),
        ]
    }
}

/// Rendered, tokenized prompts for `ClientTools` requests, and the tokenizer
/// and template that make them.
///
/// Outside the engine's actor on purpose. The server counts a request's prompt
/// before it opens the stream — `message_start` carries the count — and a
/// count that had to wait for the actor would wait behind whatever another
/// request is decoding. Rendering is pure Swift and tokenizing needs only the
/// vocabulary, so neither has to.
///
/// The last prompt is kept: the server counts a request and the engine then
/// renders the very same one, and a Claude Code prompt is long enough that
/// doing it twice is noticeable.
final class ClientToolPrompts: @unchecked Sendable {
    struct Prepared: Sendable {
        let prompt: String
        let tokens: [llama_token]
        /// The template opened a reasoning block, so the reply starts in one.
        let opensReasoning: Bool
    }

    private struct Resident {
        let modelID: String
        let vocabulary: LlamaVocabulary
        let renderer: ChatPromptRenderer
    }

    private struct Key: Equatable {
        let modelID: String
        let messages: [ChatMessage]
        let tools: ClientTools?
        let thinking: Bool
    }

    private let lock = NSLock()
    private var resident: Resident?
    private var thinkingAllowed = true
    private var last: (key: Key, prepared: Prepared)?

    /// Opens the vocabulary for a newly resident model. A rebuild of the same
    /// model keeps the one it has.
    func install(modelID: String, url: URL, chatTemplate: String?) {
        if lock.withLock({ resident?.modelID == modelID }) { return }
        guard let vocabulary = LlamaVocabulary(url: url) else {
            clear()
            return
        }
        let template = chatTemplate ?? vocabulary.chatTemplate ?? ChatPromptRenderer.fallbackTemplate
        let renderer = ChatPromptRenderer(template: template, specialTokens: vocabulary.specialTokens)
        lock.withLock {
            resident = Resident(modelID: modelID, vocabulary: vocabulary, renderer: renderer)
            last = nil
        }
    }

    func clear() {
        lock.withLock {
            resident = nil
            last = nil
        }
    }

    /// The server's "Let the model think first" switch, for other devices.
    func setThinking(allowed: Bool) {
        lock.withLock { thinkingAllowed = allowed }
    }

    func vocabulary(for modelID: String) -> LlamaVocabulary? {
        lock.withLock { resident?.modelID == modelID ? resident?.vocabulary : nil }
    }

    /// The prompt for a request, or nil when its model is not the one whose
    /// vocabulary is open.
    func prepare(_ request: GenerationRequest) throws -> Prepared? {
        let (current, allowed, cached) = lock.withLock { (resident, thinkingAllowed, last) }
        guard let current, request.modelID.isEmpty || request.modelID == current.modelID else { return nil }
        let thinking = ThinkingPolicy.allowsThinking(for: request.origin, serverAllowsThinking: allowed)
        let key = Key(modelID: current.modelID, messages: request.messages, tools: request.clientTools, thinking: thinking)
        if let cached, cached.key == key { return cached.prepared }

        let prompt = try current.renderer.render(
            request.messages,
            tools: request.clientTools ?? ClientTools(),
            thinking: thinking
        )
        let prepared = Prepared(
            prompt: prompt,
            tokens: current.vocabulary.tokenize(prompt),
            opensReasoning: ChatPromptRenderer.opensReasoning(prompt)
        )
        lock.withLock { last = (key, prepared) }
        return prepared
    }
}
