import Foundation
import FlyingFox

extension InferenceServer {
    /// `GET /api/search?q=…` — find GGUF repositories on Hugging Face.
    ///
    /// Exposed over HTTP, not just in the app, because this is the one product
    /// here that is a server: searching for a model and picking a quantisation
    /// is typing, and typing happens on a laptop. Every rival makes you do it
    /// on the phone keyboard because a phone app is all they are.
    func handleSearch(_ request: HTTPRequest) async -> HTTPResponse {
        let cors = corsHeaders()
        if let rejection = await authorizeAndLog(request) { return rejection }

        // A pasted huggingface.co link becomes the repository id it names.
        let query = HuggingFaceSearch.normalizedQuery(request.query.first(where: { $0.name == "q" })?.value ?? "")
        guard query.count >= 2 else {
            return errorResponse(
                status: .badRequest,
                message: "Pass ?q= with at least two characters.",
                style: .ollama,
                headers: cors
            )
        }

        struct Row: Encodable {
            var id: String
            var owner: String
            var name: String
            var downloads: Int
            var likes: Int
            /// Surfaced so a caller can grey it out rather than discovering the
            /// 401 after the tap. There is no token flow in this app.
            var gated: Bool
        }
        do {
            let found = try await HuggingFaceSearch().repositories(matching: query)
            return jsonResponse(
                found.map { Row(id: $0.id, owner: $0.owner, name: $0.name,
                                downloads: $0.downloads, likes: $0.likes, gated: $0.gated) },
                headers: cors
            )
        } catch {
            return errorResponse(
                status: .serviceUnavailable,
                message: "Could not reach Hugging Face.",
                style: .ollama,
                headers: cors
            )
        }
    }

    /// `GET /api/search/files?repo=owner/name` — the GGUF files in one
    /// repository, with real sizes and a verdict on whether each fits.
    func handleSearchFiles(_ request: HTTPRequest) async -> HTTPResponse {
        let cors = corsHeaders()
        if let rejection = await authorizeAndLog(request) { return rejection }

        guard let repo = request.query.first(where: { $0.name == "repo" })?.value, repo.contains("/") else {
            return errorResponse(
                status: .badRequest,
                message: "Pass ?repo=owner/name.",
                style: .ollama,
                headers: cors
            )
        }

        struct Row: Encodable {
            var filename: String
            var sizeBytes: Int64
            var quantization: String
            var isProjector: Bool
            /// "comfortable", "tight" or "willNotFit" for THIS device. The
            /// number alone means nothing to someone who does not know what
            /// iOS allows a single app.
            var fit: String
        }
        do {
            let files = try await HuggingFaceSearch().files(in: repo)
            let projector = files.first { $0.isProjector }
            let budget = deviceBudget
            return jsonResponse(
                files.map { file in
                    let record = HuggingFaceSearch.record(
                        repository: repo,
                        file: file,
                        projector: file.isProjector ? nil : projector
                    )
                    return Row(
                        filename: file.path,
                        sizeBytes: file.sizeBytes,
                        quantization: file.quantization,
                        isProjector: file.isProjector,
                        fit: String(describing: budget.fit(for: record))
                    )
                },
                headers: cors
            )
        } catch {
            return errorResponse(
                status: .serviceUnavailable,
                message: "Could not read that repository.",
                style: .ollama,
                headers: cors
            )
        }
    }

    struct AddModelRequest: Codable, Sendable {
        var repo: String
        var filename: String
        /// Optional; when omitted a projector in the same repository is paired
        /// automatically, because a vision model without one loads and then
        /// cannot see.
        var projector: String?
        var contextLength: Int?
    }

    /// `POST /api/models/add` — pull any Hugging Face GGUF, not just a
    /// catalogue entry.
    func handleAddModel(_ request: HTTPRequest) async -> HTTPResponse {
        let cors = corsHeaders()
        if let rejection = await authorizeAndLog(request) { return rejection }

        guard let payload = try? JSONDecoder().decode(AddModelRequest.self, from: await request.bodyData),
              payload.repo.contains("/"), payload.filename.hasSuffix(".gguf")
        else {
            return errorResponse(
                status: .badRequest,
                message: "Send {\"repo\":\"owner/name\",\"filename\":\"model.gguf\"}.",
                style: .ollama,
                headers: cors
            )
        }
        guard let pull = puller else {
            return errorResponse(
                status: HTTPStatusCode(501, phrase: "Not Implemented"),
                message: "This server was built without model downloading.",
                style: .ollama,
                headers: cors
            )
        }

        let record: ModelRecord
        do {
            let files = try await HuggingFaceSearch().files(in: payload.repo)
            guard let weights = files.first(where: { $0.path == payload.filename }) else {
                return errorResponse(
                    status: .notFound,
                    message: "\(payload.repo) does not publish \(payload.filename).",
                    style: .ollama,
                    headers: cors
                )
            }
            let projector = payload.projector.flatMap { name in files.first { $0.path == name } }
                ?? files.first { $0.isProjector }
            record = HuggingFaceSearch.record(
                repository: payload.repo,
                file: weights,
                projector: projector,
                contextLength: payload.contextLength ?? contextCap()
            )
        } catch {
            return errorResponse(
                status: .serviceUnavailable,
                message: "Could not read \(payload.repo).",
                style: .ollama,
                headers: cors
            )
        }

        var headers = cors
        headers[.contentType] = "application/x-ndjson"
        headers[.cacheControl] = "no-cache"
        let added = record

        return HTTPResponse.streaming(headers: headers) { continuation in
            Task {
                let encoder = JSONEncoder()
                func emit(_ value: PullProgress) {
                    guard var data = try? encoder.encode(value) else { return }
                    data.append(0x0A)
                    continuation.yield(data)
                }
                emit(PullProgress(status: "pulling manifest"))
                do {
                    for try await progress in pull(added) {
                        emit(PullProgress(
                            status: "downloading",
                            digest: added.id,
                            total: progress.totalBytes,
                            completed: progress.receivedBytes
                        ))
                    }
                    emit(PullProgress(status: "success"))
                } catch {
                    emit(PullProgress(status: "error: \(error)"))
                }
                continuation.finish()
            }
        }
    }
}
