import Foundation
import FlyingFox

extension InferenceServer {
    /// `POST /v1/audio/transcriptions` — OpenAI's speech-to-text endpoint,
    /// answered by the phone's own speech model (Parakeet, on the Neural
    /// Engine) rather than by the brain.
    ///
    /// OpenAI's shape so the clients that already speak it — the SDKs'
    /// `audio.transcriptions.create`, dictation tools, curl recipes — work by
    /// changing the base URL. `model` is read only to choose between our own
    /// speech models; `whisper-1` and anything else means whichever one is
    /// installed. `prompt` and `temperature` are accepted and ignored: a
    /// transducer model has neither.
    func handleTranscriptions(_ request: HTTPRequest) async -> HTTPResponse {
        let cors = corsHeaders()
        if let rejection = await authorizeAndLog(request) { return rejection }

        guard let transcriber else {
            await logImmediate(request, status: 501)
            return errorResponse(
                status: HTTPStatusCode(501, phrase: "Not Implemented"),
                message: "This build of Pocketd has no speech recognition.",
                type: "not_implemented",
                style: .openAI,
                headers: cors
            )
        }

        let form: MultipartForm
        do {
            form = try MultipartForm.parse(try await request.bodyData, contentType: request.headers[.contentType])
        } catch {
            await logImmediate(request, status: 400)
            return errorResponse(
                status: .badRequest,
                message: "Send the audio as multipart/form-data, in a field named file.",
                style: .openAI,
                headers: cors
            )
        }
        guard let file = form.part(named: "file"), !file.data.isEmpty else {
            await logImmediate(request, status: 400)
            return errorResponse(
                status: .badRequest, message: "The file field is missing or empty.", style: .openAI, headers: cors
            )
        }
        let format = form.value("response_format") ?? "json"
        guard Self.transcriptionFormats.contains(format) else {
            await logImmediate(request, status: 400)
            return errorResponse(
                status: .badRequest,
                message: "response_format \(format) is not supported. Use json, text or verbose_json.",
                type: "unsupported_parameter",
                style: .openAI,
                headers: cors
            )
        }
        guard let model = await transcriber.installedModel(preferring: SpeechModels.model(named: form.value("model"))) else {
            await logImmediate(request, status: 404)
            return errorResponse(
                status: .notFound,
                message: "No speech recognition model is installed on this iPhone. Download one in Pocketd: Server settings › Speech recognition.",
                type: "model_not_found",
                style: .openAI,
                headers: cors
            )
        }

        let logID = await beginLog(request, streamed: false)
        let started = ContinuousClock.now
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pocketd-audio-\(UUID().uuidString)")
            .appendingPathExtension(Self.audioExtension(for: file))
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            try file.data.write(to: url)
            let result = try await transcriber.transcribe(audioFileAt: url, model: model)
            await finishLog(logID, status: 200, model: model.id, started: started)
            return Self.transcriptionResponse(result, format: format, language: form.value("language"), headers: cors)
        } catch {
            let (status, message) = Self.transcriptionFailure(error)
            await finishLog(logID, status: status.code, model: model.id, started: started)
            return errorResponse(status: status, message: message, type: "server_error", style: .openAI, headers: cors)
        }
    }

    static let transcriptionFormats: Set<String> = ["json", "text", "verbose_json"]

    /// Kept, because AVFoundation decides how to read a file partly by its
    /// extension, and the client's filename is the only place it is said.
    static func audioExtension(for part: MultipartForm.Part) -> String {
        let fromName = (part.filename as NSString?)?.pathExtension.lowercased() ?? ""
        if !fromName.isEmpty, fromName.count <= 5, fromName.allSatisfy({ $0.isLetter || $0.isNumber }) { return fromName }
        switch part.contentType?.lowercased() {
        case "audio/wav", "audio/x-wav", "audio/wave": return "wav"
        case "audio/mpeg", "audio/mp3": return "mp3"
        case "audio/mp4", "audio/m4a", "audio/x-m4a": return "m4a"
        case "audio/flac", "audio/x-flac": return "flac"
        case "audio/aac": return "aac"
        case "audio/ogg", "audio/webm": return "caf"
        default: return "wav"
        }
    }

    static func transcriptionResponse(
        _ result: SpeechTranscription,
        format: String,
        language: String?,
        headers: HTTPHeaders
    ) -> HTTPResponse {
        var allHeaders = headers
        switch format {
        case "text":
            allHeaders[.contentType] = "text/plain; charset=utf-8"
            return HTTPResponse(statusCode: .ok, headers: allHeaders, body: Data((result.text + "\n").utf8))
        case "verbose_json":
            allHeaders[.contentType] = "application/json"
            let body: JSONValue = [
                "task": "transcribe",
                "language": language.map(JSONValue.string) ?? .null,
                "duration": result.duration.map(JSONValue.double) ?? .null,
                "text": .string(result.text),
                "segments": [],
            ]
            return HTTPResponse(statusCode: .ok, headers: allHeaders, body: Data(body.serialized.utf8))
        default:
            allHeaders[.contentType] = "application/json"
            let body: JSONValue = ["text": .string(result.text)]
            return HTTPResponse(statusCode: .ok, headers: allHeaders, body: Data(body.serialized.utf8))
        }
    }

    static func transcriptionFailure(_ error: any Error) -> (HTTPStatusCode, String) {
        switch error {
        case SpeechError.unreadableAudio(let detail):
            return (.badRequest, "The audio could not be read: \(detail)")
        case SpeechError.notInstalled:
            return (.notFound, "No speech recognition model is installed on this iPhone.")
        case SpeechError.failed(let detail):
            return (.internalServerError, detail)
        default:
            return (.internalServerError, String(describing: error))
        }
    }
}
