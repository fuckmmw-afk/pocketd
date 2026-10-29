import Foundation
import FlyingFox

enum ErrorStyle {
    case openAI
    case ollama
}

/// Errors are rendered in the dialect of whichever API the client called.
/// A client that expects `{"error": "..."}` and receives OpenAI's nested shape
/// reports "unknown error" and hides the reason from the user.
func errorResponse(
    status: HTTPStatusCode,
    message: String,
    type: String = "invalid_request_error",
    code: String? = nil,
    style: ErrorStyle,
    headers: HTTPHeaders = [:]
) -> HTTPResponse {
    var allHeaders = headers
    allHeaders[.contentType] = "application/json"
    let encoder = JSONEncoder()
    let body: Data
    switch style {
    case .openAI:
        body = (try? encoder.encode(OpenAI.ErrorResponse(
            error: OpenAI.ErrorBody(message: message, type: type, code: code)
        ))) ?? Data()
    case .ollama:
        body = (try? encoder.encode(Ollama.ErrorResponse(error: message))) ?? Data()
    }
    return HTTPResponse(statusCode: status, headers: allHeaders, body: body)
}

func jsonResponse(_ value: some Encodable, headers: HTTPHeaders = [:]) -> HTTPResponse {
    var allHeaders = headers
    allHeaders[.contentType] = "application/json"
    let encoder = JSONEncoder()
    guard let data = try? encoder.encode(value) else {
        return errorResponse(status: .internalServerError, message: "Failed to encode response.", style: .openAI)
    }
    return HTTPResponse(statusCode: .ok, headers: allHeaders, body: data)
}

extension InferenceServer {
    func installRoutes(on server: HTTPServer) async {
        // OpenAI dialect
        await server.appendRoute("GET /v1/models") { [self] request in
            await handleListModels(request)
        }
        await server.appendRoute("GET /v1/models/:model") { [self] request in
            await handleRetrieveModel(request)
        }
        await server.appendRoute("POST /v1/chat/completions") { [self] request in
            await handleChatCompletions(request)
        }
        await server.appendRoute("POST /v1/completions") { [self] request in
            await handleCompletions(request)
        }

        // Anthropic dialect
        await server.appendRoute("POST /v1/messages") { [self] request in
            await handleAnthropicMessages(request)
        }
        await server.appendRoute("POST /v1/messages/count_tokens") { [self] request in
            await handleAnthropicCountTokens(request)
        }
        // Batches and the rest of the Messages API are not served. Said in
        // Anthropic's shape, which is the only one its clients can read.
        await server.appendRoute("POST /v1/messages/*") { [self] request in
            anthropicError(
                .notFound,
                type: "not_found_error",
                message: "No route for \(request.path).",
                headers: await corsHeaders()
            )
        }

        // Ollama dialect
        await server.appendRoute("GET /api/tags") { [self] request in
            await handleOllamaTags(request)
        }
        await server.appendRoute("POST /api/chat") { [self] request in
            await handleOllamaChat(request)
        }
        await server.appendRoute("POST /api/generate") { [self] request in
            await handleOllamaGenerate(request)
        }
        await server.appendRoute("POST /api/show") { [self] request in
            await handleOllamaShow(request)
        }
        await server.appendRoute("GET /api/ps") { [self] request in
            await handleOllamaPs(request)
        }
        await server.appendRoute("POST /api/pull") { [self] request in
            await handlePull(request)
        }
        await server.appendRoute("GET /api/catalogue") { [self] request in
            await handleCatalogue(request)
        }
        await server.appendRoute("GET /api/search") { [self] request in
            await handleSearch(request)
        }
        await server.appendRoute("GET /api/search/files") { [self] request in
            await handleSearchFiles(request)
        }
        await server.appendRoute("POST /api/models/add") { [self] request in
            await handleAddModel(request)
        }
        await server.appendRoute("GET /api/version") { [self] _ in
            jsonResponse(Ollama.VersionResponse(version: PocketdKit.version), headers: await corsHeaders())
        }

        // Embeddings are not implemented. Saying so explicitly matters: a 404
        // reads as a wrong base URL and sends people debugging their config,
        // whereas 501 tells a client to fall back to another provider.
        for route in ["POST /v1/embeddings", "POST /api/embed", "POST /api/embeddings"] {
            await server.appendRoute(HTTPRoute(route)) { [self] request in
                errorResponse(
                    status: HTTPStatusCode(501, phrase: "Not Implemented"),
                    message: "Pocketd does not serve embeddings. Only chat and text completion are implemented.",
                    type: "not_implemented",
                    style: request.path.hasPrefix("/api") ? .ollama : .openAI,
                    headers: await corsHeaders()
                )
            }
        }

        // Shared
        await server.appendRoute("GET /health") { [self] _ in
            await handleHealth()
        }
        await server.appendRoute("OPTIONS /*") { [self] _ in
            HTTPResponse(statusCode: .noContent, headers: await corsHeaders())
        }
        // Some Ollama clients probe the root before anything else and refuse to
        // proceed unless it answers with this exact string. Several of them probe
        // with HEAD rather than GET, and read a 404 as "no server here".
        await server.appendRoute("GET /") { [self] request in
            await handleRoot(request)
        }
        await server.appendRoute("HEAD /") { [self] _ in
            HTTPResponse(
                statusCode: .ok,
                headers: await corsHeaders(),
                body: Data("Ollama is running".utf8)
            )
        }
        // An unambiguous path to type, for when someone was told "open your
        // phone's address" and got the probe string instead.
        await server.appendRoute("GET /setup") { [self] _ in
            await handleSetupPage()
        }
        await server.appendRoute("GET /chat") { [self] _ in
            await handleChatPage()
        }
        await server.appendRoute("POST /pair") { [self] request in
            await handlePair(request)
        }
        await server.appendRoute("HEAD /api/version") { [self] _ in
            jsonResponse(Ollama.VersionResponse(version: PocketdKit.version), headers: await corsHeaders())
        }
        await server.appendRoute("HEAD /api/tags") { [self] request in
            await handleOllamaTags(request)
        }
        await server.appendRoute("GET /*") { [self] request in
            errorResponse(
                status: .notFound,
                message: "No route for \(request.path).",
                style: request.path.hasPrefix("/api") ? .ollama : .openAI,
                headers: await corsHeaders()
            )
        }
    }

    // MARK: - Shared helpers used by both dialects

    func handleHealth() async -> HTTPResponse {
        struct Health: Encodable {
            var status: String
            var version: String
            var backend: String
            var model: String?
            var maxContextTokens: Int
            /// How many generations are running. A phone serves one at a time,
            /// so a client seeing 1 knows a 503 is contention rather than a
            /// fault, and can wait instead of failing over.
            var activeRequests: Int
            var maxConcurrentRequests: Int
            /// The live axis: whether the model resident right now was built
            /// with a projector, so images will be understood without a reload.
            var visionActive: Bool
            /// "ok", "thermal" or "battery". A client that polls this can back
            /// off before it starts getting 503s.
            var condition: String
        }
        let engine = currentEngine()
        return jsonResponse(
            Health(
                status: "ok",
                version: PocketdKit.version,
                backend: engine.backendName,
                model: await engine.loadedModel()?.id,
                maxContextTokens: contextCap(),
                activeRequests: activeRequestCount(),
                maxConcurrentRequests: configuration.maxConcurrentRequests,
                visionActive: await engine.loadedModel()?.declaredCapabilities.vision.isYes ?? false,
                condition: serveCondition().rawValue
            ),
            headers: corsHeaders()
        )
    }

    /// Resolves the model a request asked for, loading it if it is installed but
    /// not resident. Clients written against a desktop Ollama assume switching
    /// models is free; on a phone it costs a few seconds and a memory eviction,
    /// but failing outright would break every one of those clients.
    func resolveModel(id requested: String) async throws -> ModelRecord {
        let engine = currentEngine()
        let loaded = await engine.loadedModel()

        if requested.isEmpty {
            if let loaded { return loaded }
            guard let first = await models().first else { throw InferenceError.noModelLoaded }
            try await engine.load(model: first)
            return first
        }
        if let loaded, loaded.id == requested { return loaded }
        guard let match = await models().first(where: { $0.id == requested }) else {
            throw InferenceError.modelNotFound(requested)
        }
        try await engine.load(model: match)
        return match
    }

    /// Declared capabilities, upgraded with live-session facts when the model
    /// in question is the resident one. The single place these two axes are
    /// combined, so a listing and a status endpoint cannot contradict.
    func capabilities(for model: ModelRecord, resident: ModelRecord?) -> ModelCapabilities {
        var caps = model.declaredCapabilities
        guard resident?.id == model.id else { return caps }
        caps.effectiveContextLength = min(model.contextLength, contextCap())
        // A projector is loaded with the weights or not at all, so residency
        // plus a declared projector is exactly the condition for vision now.
        caps.visionActive = caps.vision.isYes
        return caps
    }

    /// Records a request that is answered immediately.
    ///
    /// Only the two completion routes used to log, so the section headed
    /// "Requests" could read "No requests yet" after five failed pairing
    /// attempts from an unknown address — which is precisely the thing a
    /// security-facing log exists to show.
    func logImmediate(_ request: HTTPRequest, status: Int) async {
        await log.record(RequestLogEntry(
            method: request.method.rawValue,
            path: request.path,
            clientAddress: request.peerAddress,
            statusCode: status
        ))
    }

    func beginLog(_ request: HTTPRequest, streamed: Bool) async -> UUID {
        let entry = RequestLogEntry(
            method: request.method.rawValue,
            path: request.path,
            clientAddress: request.peerAddress,
            streamed: streamed
        )
        await log.record(entry)
        return entry.id
    }

    func finishLog(
        _ id: UUID,
        status: Int,
        model: String? = nil,
        usage: TokenUsage? = nil,
        started: ContinuousClock.Instant? = nil
    ) async {
        let elapsed = started.map { $0.duration(to: .now).timeInterval }
        await log.update(id: id) { entry in
            entry.statusCode = status
            entry.model = model ?? entry.model
            entry.promptTokens = usage?.promptTokens ?? entry.promptTokens
            entry.completionTokens = usage?.completionTokens ?? entry.completionTokens
            entry.duration = elapsed ?? entry.duration
        }
    }

    /// Maps engine failures onto the status code the dialect expects.
    func response(for error: any Error, style: ErrorStyle, headers: HTTPHeaders) -> HTTPResponse {
        switch error {
        case InferenceError.modelNotFound(let id):
            return errorResponse(status: .notFound, message: "Model '\(id)' is not installed.", type: "model_not_found", style: style, headers: headers)
        case InferenceError.noModelLoaded:
            return errorResponse(status: .serviceUnavailable, message: "No model is loaded.", type: "model_not_loaded", style: style, headers: headers)
        case InferenceError.contextExhausted:
            return errorResponse(status: .payloadTooLarge, message: "Prompt exceeds the configured context window.", type: "context_length_exceeded", style: style, headers: headers)
        case InferenceError.cancelled:
            return errorResponse(status: .serviceUnavailable, message: "Generation was cancelled.", type: "cancelled", style: style, headers: headers)
        case InferenceError.backend(let detail):
            // The detail is already a sentence written for a human; describing
            // the enum around it just adds noise the client has to read past.
            return errorResponse(status: .internalServerError, message: detail, type: "server_error", style: style, headers: headers)
        default:
            return errorResponse(status: .internalServerError, message: String(describing: error), type: "server_error", style: style, headers: headers)
        }
    }
}

extension Duration {
    var timeInterval: TimeInterval {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}

public enum PocketdKit {
    /// Reported through `/health` and `/api/version`. Ollama clients version-gate
    /// on the latter, so it has to parse as a semantic version.
    ///
    /// It must also equal the app's MARKETING_VERSION, and it silently did not:
    /// the app shipped 1.0.0 while every `/health` response and every OpenAI
    /// `system_fingerprint` still said 0.1.0. Nothing could notice, because
    /// this package has no bundle to read the real number out of — it is built
    /// and tested without one on purpose, so the server can be exercised on a
    /// Linux runner in seconds.
    ///
    /// So the two are pinned together by `ServerVersionTests` instead, which
    /// reads project.yml. Bumping one without the other is now a test failure
    /// rather than a wrong answer on the wire.
    public static let version = "1.1.1"

    /// Echoed on OpenAI responses. The SDKs surface it verbatim, so it says
    /// what actually served the request rather than imitating another vendor.
    public static let systemFingerprint = "pocketd-\(version)"
}
