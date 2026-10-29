import Foundation
import FlyingFox

/// The Anthropic Messages API, so Claude Code — or anything built on the
/// Anthropic SDKs — can use the phone through `ANTHROPIC_BASE_URL`.
///
/// It is a third dialect over the same engine, admission and log as the other
/// two, and differs from them in three ways that all come from Claude Code:
///
/// - **Tools belong to the client.** The model is shown the client's tools
///   through its own chat template, and what it writes when it wants one comes
///   back as a `tool_use` block for the client to run. Nothing runs here.
/// - **Requests queue rather than bounce.** Claude Code sends its main request
///   and a few small ones at once. The other dialects answer the second with a
///   503; here it waits in the engine's queue, and only a queue that is already
///   long is refused, with Anthropic's 529.
/// - **The stream opens at once.** `message_start` goes out before the prompt
///   is read, and a `ping` every ten seconds after, so a client with a read
///   timeout does not give up on a phone that is still reading a long prompt.
extension InferenceServer {
    /// How many Anthropic requests may wait behind the ones being served.
    static let anthropicQueueDepth = 8

    static let overloaded = HTTPStatusCode(529, phrase: "Overloaded")

    func handleAnthropicMessages(_ request: HTTPRequest) async -> HTTPResponse {
        let cors = corsHeaders()
        if let rejection = await authorizeAnthropic(request) { return rejection }

        let payload: Anthropic.MessagesRequest
        switch await decodeAnthropic(request, headers: cors) {
        case let .success(decoded): payload = decoded
        case let .failure(rejection): return rejection.response
        }

        let wantsStream = payload.stream ?? false
        let logID = await beginLog(request, streamed: wantsStream)

        let condition = serveCondition()
        if !condition.isServing {
            await finishLog(logID, status: Self.overloaded.code)
            var headers = cors
            headers[HTTPHeader("Retry-After")] = String(condition.retryAfter)
            return anthropicError(Self.overloaded, type: "overloaded_error", message: condition.message, headers: headers)
        }

        guard beginQueuedRequest() else {
            await finishLog(logID, status: Self.overloaded.code)
            var headers = cors
            headers[HTTPHeader("Retry-After")] = "5"
            return anthropicError(
                Self.overloaded,
                type: "overloaded_error",
                message: "Pocketd's queue is full. Retry shortly.",
                headers: headers
            )
        }

        let prepared: AnthropicPrepared
        do {
            prepared = try await prepareAnthropic(payload, origin: request.requestOrigin)
        } catch {
            endRequest()
            let failure = anthropicFailure(for: error)
            await finishLog(logID, status: failure.status.code)
            return anthropicError(failure.status, type: failure.type, message: failure.message, headers: cors)
        }

        if prepared.promptTokens >= prepared.window {
            endRequest()
            await finishLog(logID, status: 400, model: prepared.model.id)
            return anthropicError(
                .badRequest,
                type: "invalid_request_error",
                message: Self.promptTooLong(tokens: prepared.promptTokens, window: prepared.window),
                headers: cors
            )
        }

        return wantsStream
            ? streamAnthropic(prepared, payload: payload, logID: logID, cors: cors)
            : await bufferedAnthropic(prepared, payload: payload, logID: logID, cors: cors)
    }

    func handleAnthropicCountTokens(_ request: HTTPRequest) async -> HTTPResponse {
        let cors = corsHeaders()
        if let rejection = await authorizeAnthropic(request) { return rejection }

        let payload: Anthropic.MessagesRequest
        switch await decodeAnthropic(request, headers: cors) {
        case let .success(decoded): payload = decoded
        case let .failure(rejection): return rejection.response
        }

        do {
            let prepared = try await prepareAnthropic(payload, origin: request.requestOrigin)
            await logImmediate(request, status: 200)
            return anthropicJSON(["input_tokens": .int(prepared.promptTokens)], headers: cors)
        } catch {
            let failure = anthropicFailure(for: error)
            await logImmediate(request, status: failure.status.code)
            return anthropicError(failure.status, type: failure.type, message: failure.message, headers: cors)
        }
    }

    // MARK: - Preparation

    struct AnthropicPrepared: Sendable {
        var model: ModelRecord
        var generation: GenerationRequest
        var promptTokens: Int
        var window: Int
        var tools: [ToolDefinition]
    }

    struct AnthropicRejection: Error {
        var response: HTTPResponse
    }

    private func decodeAnthropic(
        _ request: HTTPRequest,
        headers: HTTPHeaders
    ) async -> Result<Anthropic.MessagesRequest, AnthropicRejection> {
        let payload: Anthropic.MessagesRequest
        do {
            payload = try JSONDecoder().decode(Anthropic.MessagesRequest.self, from: await request.bodyData)
        } catch {
            await logImmediate(request, status: 400)
            return .failure(AnthropicRejection(response: anthropicError(
                .badRequest,
                type: "invalid_request_error",
                message: "Malformed request body: \(error)",
                headers: headers
            )))
        }
        guard !payload.messages.isEmpty else {
            await logImmediate(request, status: 400)
            return .failure(AnthropicRejection(response: anthropicError(
                .badRequest,
                type: "invalid_request_error",
                message: "messages: at least one message is required",
                headers: headers
            )))
        }
        return .success(payload)
    }

    /// The model, the engine's request, and the prompt's real size.
    ///
    /// The name a client asks for is honoured when it is installed here and
    /// ignored otherwise. Claude Code asks for `claude-…` models by name; a 404
    /// for each of them would make the phone unusable from it, and answering
    /// with whatever brain is resident is what the person pointing Claude Code
    /// at a phone meant.
    private func prepareAnthropic(
        _ payload: Anthropic.MessagesRequest,
        origin: RequestOrigin
    ) async throws -> AnthropicPrepared {
        let installed = await models()
        let requested = installed.contains(where: { $0.id == payload.model }) ? payload.model : ""
        let model = try await resolveModel(id: requested)

        let tools = AnthropicConversion.clientTools(payload.tools, choice: payload.tool_choice)
        let messages = AnthropicConversion.chatMessages(
            system: payload.system?.text,
            messages: payload.messages,
            acceptsImages: model.projectorFilename != nil
        )
        var generation = GenerationRequest(
            modelID: model.id,
            messages: messages,
            options: AnthropicConversion.options(payload),
            origin: origin,
            clientTools: tools
        )
        let promptTokens = try await currentEngine().countPromptTokens(generation)
        let window = min(model.contextLength, contextCap())
        // Whatever the client asked for, the reply cannot run past the window:
        // Claude Code asks for 32,000 tokens on every request.
        let room = max(1, window - promptTokens)
        generation.options.maxTokens = min(payload.max_tokens ?? room, room)
        return AnthropicPrepared(
            model: model,
            generation: generation,
            promptTokens: promptTokens,
            window: window,
            tools: tools.offered
        )
    }

    /// Anthropic's own wording, which clients match on: Claude Code reads
    /// "prompt is too long" as the cue to compact the conversation.
    static func promptTooLong(tokens: Int, window: Int) -> String {
        "prompt is too long: \(tokens) tokens > \(window - 1) maximum. "
            + "Pocketd's context limit is \(window) tokens; raise it in Server settings, or shorten the conversation."
    }

    // MARK: - Answering

    private func bufferedAnthropic(
        _ prepared: AnthropicPrepared,
        payload: Anthropic.MessagesRequest,
        logID: UUID,
        cors: HTTPHeaders
    ) async -> HTTPResponse {
        let started = ContinuousClock.now
        defer { endRequest() }
        var assembler = AnthropicReplyAssembler(stopSequences: payload.stop_sequences ?? [], tools: prepared.tools)
        var text = ""
        var reason = FinishReason.stop
        var usage: TokenUsage?
        do {
            loop: for try await event in try await currentEngine().generate(prepared.generation) {
                switch event {
                case let .token(chunk):
                    text += assembler.feed(chunk)
                    if assembler.stopSequence != nil { break loop }
                case .toolCallStarted, .answerCard:
                    break
                case let .finished(finish, tokens):
                    reason = finish
                    usage = tokens
                }
            }
        } catch {
            let failure = anthropicFailure(for: error)
            await finishLog(logID, status: failure.status.code, model: prepared.model.id, started: started)
            return anthropicError(failure.status, type: failure.type, message: failure.message, headers: cors)
        }

        let (tail, uses) = assembler.finish { AnthropicConversion.identifier(prefix: "toolu_") }
        text += tail
        let outputTokens = usage?.completionTokens ?? Self.estimatedTokens(characters: assembler.producedCharacters)
        await finishLog(
            logID,
            status: 200,
            model: prepared.model.id,
            usage: TokenUsage(promptTokens: prepared.promptTokens, completionTokens: outputTokens),
            started: started
        )
        return anthropicJSON(
            AnthropicMessage.body(
                id: AnthropicConversion.identifier(prefix: "msg_"),
                model: payload.model.isEmpty ? prepared.model.id : payload.model,
                text: text,
                toolUses: uses,
                stopReason: Self.stopReason(uses: uses, stopSequence: assembler.stopSequence, finish: reason),
                stopSequence: assembler.stopSequence,
                inputTokens: prepared.promptTokens,
                outputTokens: outputTokens
            ),
            headers: cors
        )
    }

    private func streamAnthropic(
        _ prepared: AnthropicPrepared,
        payload: Anthropic.MessagesRequest,
        logID: UUID,
        cors: HTTPHeaders
    ) -> HTTPResponse {
        let engine = currentEngine()
        let log = self.log
        var headers = ServerSentEvents.headers
        for (key, value) in cors { headers[key] = value }
        let wireModel = payload.model.isEmpty ? prepared.model.id : payload.model
        let stops = payload.stop_sequences ?? []

        return HTTPResponse.streaming(headers: headers, stoppingWhenClientLeaves: { [weak self] continuation in
            let started = ContinuousClock.now
            var stream = AnthropicMessageStream(id: AnthropicConversion.identifier(prefix: "msg_"), model: wireModel)
            continuation.yield(stream.start(inputTokens: prepared.promptTokens))

            // For the whole stream, not only the prefill: a model writing a
            // long tool call shows nothing until the call is complete, and
            // that can be as quiet as the prompt was.
            let pinger = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(10))
                    if Task.isCancelled { break }
                    continuation.yield(AnthropicMessageStream.ping)
                }
            }
            defer { pinger.cancel() }

            var assembler = AnthropicReplyAssembler(stopSequences: stops, tools: prepared.tools)
            var reason = FinishReason.stop
            var usage: TokenUsage?
            var status = 200

            do {
                loop: for try await event in try await engine.generate(prepared.generation) {
                    switch event {
                    case let .token(chunk):
                        for frame in stream.text(assembler.feed(chunk)) { continuation.yield(frame) }
                        if assembler.stopSequence != nil { break loop }
                    case .toolCallStarted, .answerCard:
                        break
                    case let .finished(finish, tokens):
                        reason = finish
                        usage = tokens
                    }
                }
                pinger.cancel()
                let (tail, uses) = assembler.finish { AnthropicConversion.identifier(prefix: "toolu_") }
                for frame in stream.text(tail) { continuation.yield(frame) }
                let outputTokens = usage?.completionTokens
                    ?? InferenceServer.estimatedTokens(characters: assembler.producedCharacters)
                usage = TokenUsage(promptTokens: prepared.promptTokens, completionTokens: outputTokens)
                for frame in stream.finish(
                    toolUses: uses,
                    stopReason: InferenceServer.stopReason(uses: uses, stopSequence: assembler.stopSequence, finish: reason),
                    stopSequence: assembler.stopSequence,
                    outputTokens: outputTokens
                ) {
                    continuation.yield(frame)
                }
            } catch {
                // The 200 went out with message_start; an error now can only
                // be an event in the stream, after which the stream ends.
                pinger.cancel()
                let failure = InferenceServer.anthropicFailureDescription(for: error)
                status = failure.status.code
                continuation.yield(AnthropicMessageStream.error(type: failure.type, message: failure.message))
            }

            if Task.isCancelled { status = InferenceServer.clientClosedRequest }
            continuation.finish()
            await self?.endRequest()
            let elapsed = started.duration(to: .now).timeInterval
            let finalStatus = status
            let finalUsage = usage
            await log.update(id: logID) { entry in
                entry.statusCode = finalStatus
                entry.model = prepared.model.id
                entry.promptTokens = prepared.promptTokens
                entry.completionTokens = finalUsage?.completionTokens ?? entry.completionTokens
                entry.duration = elapsed
            }
        })
    }

    static func stopReason(uses: [AnthropicToolUse], stopSequence: String?, finish: FinishReason) -> String {
        if !uses.isEmpty { return "tool_use" }
        if stopSequence != nil { return "stop_sequence" }
        if finish == .length { return "max_tokens" }
        return "end_turn"
    }

    /// For a reply cut short by a stop sequence, where the engine never got to
    /// report its count.
    static func estimatedTokens(characters: Int) -> Int {
        max(1, (characters + 3) / 4)
    }

    // MARK: - Auth and errors

    /// The same key as every other route, from either header Anthropic's
    /// clients use: `x-api-key` (ANTHROPIC_API_KEY) or `Authorization: Bearer`
    /// (ANTHROPIC_AUTH_TOKEN). Either one matching is enough, so a client that
    /// sends both with one of them stale still gets in.
    func authorizeAnthropic(_ request: HTTPRequest) async -> HTTPResponse? {
        guard configuration.requiresAuth else { return nil }
        var presented: [String] = []
        if let authorization = request.headers[.authorization] {
            let value = authorization.trimmingCharacters(in: .whitespaces)
            presented.append(value.lowercased().hasPrefix("bearer ") ? String(value.dropFirst(7)).trimmingCharacters(in: .whitespaces) : value)
        }
        if let key = request.headers[HTTPHeader("x-api-key")] {
            presented.append(key.trimmingCharacters(in: .whitespaces))
        }
        if presented.contains(where: { constantTimeEquals($0, configuration.apiKey) }) { return nil }
        await logImmediate(request, status: 401)
        return anthropicError(
            .unauthorized,
            type: "authentication_error",
            message: "Missing or invalid API key. Send Pocketd's key as x-api-key or Authorization: Bearer.",
            headers: corsHeaders()
        )
    }

    nonisolated func anthropicError(_ status: HTTPStatusCode, type: String, message: String, headers: HTTPHeaders) -> HTTPResponse {
        var allHeaders = headers
        allHeaders[.contentType] = "application/json"
        let body = AnthropicMessage.errorBody(type: type, message: message).serialized
        return HTTPResponse(statusCode: status, headers: allHeaders, body: Data(body.utf8))
    }

    nonisolated func anthropicJSON(_ value: JSONValue, headers: HTTPHeaders) -> HTTPResponse {
        var allHeaders = headers
        allHeaders[.contentType] = "application/json"
        return HTTPResponse(statusCode: .ok, headers: allHeaders, body: Data(value.serialized.utf8))
    }

    struct AnthropicFailure: Sendable {
        var status: HTTPStatusCode
        var type: String
        var message: String
    }

    nonisolated func anthropicFailure(for error: any Error) -> AnthropicFailure {
        Self.anthropicFailureDescription(for: error)
    }

    /// Engine failures in Anthropic's error vocabulary. The types are the ones
    /// the SDKs branch on: 529 and `overloaded_error` are retried, 400 is not.
    static func anthropicFailureDescription(for error: any Error) -> AnthropicFailure {
        switch error {
        case InferenceError.noModelLoaded:
            return AnthropicFailure(
                status: .notFound, type: "not_found_error",
                message: "No model is installed on this iPhone. Download one in Pocketd first."
            )
        case InferenceError.modelNotFound(let id):
            return AnthropicFailure(
                status: .notFound, type: "not_found_error",
                message: "Model '\(id)' is not installed on this iPhone."
            )
        case InferenceError.modelMismatch:
            return AnthropicFailure(
                status: overloaded, type: "overloaded_error",
                message: "Pocketd was switching models while this request waited. Retry."
            )
        case InferenceError.contextExhausted:
            return AnthropicFailure(
                status: .badRequest, type: "invalid_request_error",
                message: "prompt is too long for Pocketd's context window. Raise the context limit in Server settings, or shorten the conversation."
            )
        case InferenceError.cancelled:
            return AnthropicFailure(
                status: overloaded, type: "overloaded_error",
                message: "The iPhone stopped this answer to do something else. Retry."
            )
        case let InferenceError.backend(detail):
            return AnthropicFailure(status: .internalServerError, type: "api_error", message: detail)
        default:
            return AnthropicFailure(status: .internalServerError, type: "api_error", message: String(describing: error))
        }
    }
}
