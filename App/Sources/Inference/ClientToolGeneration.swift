import Foundation
import llama
import PocketdKit

/// One `ClientTools` generation, straight on a llama.cpp context.
///
/// Not through LocalLLMClient, for three reasons that all come from serving
/// Claude Code. Its prompt is rendered here (`ChatPromptRenderer`), with tool
/// calls the library has no way to take. Its sampler is built here, per
/// request, so a client that changes temperature does not cost a context
/// rebuild. And its prompt cache is by *token*, kept here: a Claude Code
/// prompt is tens of thousands of tokens of system prompt and tool schemas
/// that do not change between turns, and only the new end of it is read again.
/// LocalLLMClient's cache compares rendered text from the start and gives up at
/// the first difference, which on Qwen 3 is every turn — the empty reasoning
/// block it puts in the generation prompt is not there when the same turn is
/// rendered as history.
///
/// What is in the context's KV cache is tracked as the tokens decoded into it,
/// in order, and nothing else. A request reuses the longest prefix it shares
/// with them and removes the rest; `nil` means the cache is not known to hold
/// anything in particular and is cleared first.
enum ClientToolGeneration {
    struct Outcome {
        /// What the KV cache now holds, or nil if it is not known.
        var cache: [llama_token]?
        var failure: InferenceError?
    }

    /// Feeds `prepared` into `context` and streams the reply into
    /// `continuation`, finishing it. Runs on the caller's actor: the context
    /// is the caller's, and so is the gate that keeps anyone else off it.
    static func run(
        isolation: isolated (any Actor)? = #isolation,
        context: OpaquePointer,
        vocabulary: LlamaVocabulary,
        prepared: ClientToolPrompts.Prepared,
        cache: [llama_token]?,
        window: Int,
        maxTokens: Int?,
        sampling: SamplingParameters,
        continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation
    ) async -> Outcome {
        let tokens = prepared.tokens
        guard !tokens.isEmpty else {
            continuation.finish(throwing: InferenceError.backend("The chat template rendered an empty prompt."))
            return Outcome(cache: cache, failure: nil)
        }
        guard tokens.count < window else {
            continuation.finish(throwing: InferenceError.contextExhausted)
            return Outcome(cache: cache, failure: .contextExhausted)
        }
        guard let memory = llama_get_memory(context) else {
            continuation.finish(throwing: InferenceError.backend("This context has no memory to decode into."))
            return Outcome(cache: nil, failure: nil)
        }

        // Keep the shared prefix, never all of it: the last prompt token is
        // always decoded again, because its logits are what the first reply
        // token is sampled from.
        var reused = 0
        if let cache {
            let limit = min(cache.count, tokens.count - 1)
            while reused < limit, cache[reused] == tokens[reused] { reused += 1 }
            // A recurrent or hybrid model cannot drop part of a sequence;
            // llama.cpp says so, and the whole prompt is read again.
            if !llama_memory_seq_rm(memory, 0, llama_pos(reused), -1) {
                llama_memory_clear(memory, true)
                reused = 0
            }
        } else {
            llama_memory_clear(memory, true)
        }
        var held = Array(tokens[..<reused])

        let capacity = max(1, min(Int(llama_n_batch(context)), 512))
        var batch = llama_batch_init(Int32(capacity), 0, 1)
        defer { llama_batch_free(batch) }

        func decode(_ slice: ArraySlice<llama_token>, at start: Int, logitsForLast: Bool) -> Int32 {
            batch.n_tokens = 0
            for (offset, token) in slice.enumerated() {
                let slot = Int(batch.n_tokens)
                batch.token[slot] = token
                batch.pos[slot] = llama_pos(start + offset)
                batch.n_seq_id[slot] = 1
                batch.seq_id[slot]![0] = 0
                batch.logits[slot] = 0
                batch.n_tokens += 1
            }
            if logitsForLast, batch.n_tokens > 0 {
                batch.logits[Int(batch.n_tokens) - 1] = 1
            }
            return llama_decode(context, batch)
        }

        // The prompt, a batch at a time, giving the actor back between
        // batches: reading a long prompt takes seconds, and the server's
        // questions about what is loaded should not wait for all of it.
        var position = reused
        while position < tokens.count {
            if Task.isCancelled {
                continuation.finish(throwing: InferenceError.cancelled)
                return Outcome(cache: held, failure: .cancelled)
            }
            let end = min(position + capacity, tokens.count)
            let status = decode(tokens[position..<end], at: position, logitsForLast: end == tokens.count)
            guard status == 0 else {
                let failure = Self.failure(for: status)
                continuation.finish(throwing: failure)
                return Outcome(cache: nil, failure: failure)
            }
            held.append(contentsOf: tokens[position..<end])
            position = end
            await Task.yield()
        }

        guard let sampler = makeSampler(sampling) else {
            continuation.finish(throwing: InferenceError.backend("The sampler could not be built."))
            return Outcome(cache: held, failure: nil)
        }
        defer { llama_sampler_free(sampler) }

        var reasoning = ReasoningFilter(startsInsideReasoning: prepared.opensReasoning)
        var bytes: [UInt8] = []
        var produced = 0
        var reason = FinishReason.stop
        let limit = maxTokens ?? window
        var failure: InferenceError?

        func show(_ text: String) {
            guard !text.isEmpty else { return }
            let visible = reasoning.feed(text)
            if !visible.isEmpty { continuation.yield(.token(visible)) }
        }

        while true {
            if Task.isCancelled {
                reason = .cancelled
                break
            }
            if produced >= limit || position >= window {
                reason = .length
                break
            }
            let token = llama_sampler_sample(sampler, context, -1)
            if vocabulary.isEndOfGeneration(token) { break }
            produced += 1
            bytes += vocabulary.piece(token)
            show(takeUTF8(&bytes))

            let status = decode([token][...], at: position, logitsForLast: true)
            guard status == 0 else {
                if status == 1 {
                    // No room left in the KV cache: the reply is as long as
                    // this context allows.
                    reason = .length
                } else {
                    failure = Self.failure(for: status)
                }
                break
            }
            held.append(token)
            position += 1
            await Task.yield()
        }

        if let failure {
            continuation.finish(throwing: failure)
            return Outcome(cache: nil, failure: failure)
        }
        if !bytes.isEmpty { show(String(decoding: bytes, as: UTF8.self)) }
        let tail = reasoning.finish()
        if !tail.isEmpty { continuation.yield(.token(tail)) }
        continuation.yield(.finished(
            reason: reason,
            usage: TokenUsage(promptTokens: tokens.count, completionTokens: produced)
        ))
        continuation.finish()
        return Outcome(cache: held, failure: nil)
    }

    private static func failure(for status: Int32) -> InferenceError {
        switch status {
        case 1: return .contextExhausted
        case 2: return .cancelled
        default: return .backend("llama.cpp could not decode this prompt (status \(status)).")
        }
    }

    /// llama.cpp's own order (penalties, top-k, typical, top-p, temperature,
    /// then the draw), with each stage left out when its setting is a no-op.
    /// A temperature of zero is greedy, which is what a client asking for 0
    /// means.
    static func makeSampler(_ sampling: SamplingParameters) -> UnsafeMutablePointer<llama_sampler>? {
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { return nil }
        if sampling.repeatPenalty != 1, sampling.repeatLastN != 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                Int32(sampling.repeatLastN), Float(sampling.repeatPenalty), 0, 0
            ))
        }
        if sampling.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
            return chain
        }
        if sampling.topK > 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(sampling.topK)))
        }
        if sampling.typicalP < 1 {
            llama_sampler_chain_add(chain, llama_sampler_init_typical(Float(sampling.typicalP), 1))
        }
        if sampling.topP < 1 {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(Float(sampling.topP), 1))
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(Float(sampling.temperature)))
        let seed = sampling.seed.map { UInt32(truncatingIfNeeded: $0) } ?? UInt32.random(in: 0...UInt32.max)
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
        return chain
    }

    /// The longest prefix of `bytes` that is whole UTF-8, removed from it. A
    /// character split across tokens waits for the rest; a run that will never
    /// be valid is let through with replacement characters.
    static func takeUTF8(_ bytes: inout [UInt8]) -> String {
        guard !bytes.isEmpty else { return "" }
        for cut in stride(from: bytes.count, through: max(1, bytes.count - 3), by: -1) {
            if let text = String(bytes: bytes[..<cut], encoding: .utf8) {
                bytes.removeFirst(cut)
                return text
            }
        }
        guard bytes.count > 3 else { return "" }
        let text = String(decoding: bytes, as: UTF8.self)
        bytes.removeAll()
        return text
    }
}
