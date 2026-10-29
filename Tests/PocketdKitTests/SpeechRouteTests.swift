import Foundation
import Testing
@testable import PocketdKit

/// Says what it was given, so a test can see the audio arrive intact.
private actor FakeTranscriber: SpeechTranscriber {
    let installed: [SpeechModel]
    private(set) var received: [(bytes: Data, ext: String, model: SpeechModel)] = []

    init(installed: [SpeechModel] = [SpeechModels.parakeetV3]) {
        self.installed = installed
    }

    func installedModel(preferring preferred: SpeechModel?) async -> SpeechModel? {
        if let preferred, installed.contains(preferred) { return preferred }
        return installed.first
    }

    func transcribe(audioFileAt url: URL, model: SpeechModel) async throws -> SpeechTranscription {
        let bytes = try Data(contentsOf: url)
        received.append((bytes, url.pathExtension, model))
        if bytes == Data("garbage".utf8) { throw SpeechError.unreadableAudio("not audio") }
        return SpeechTranscription(text: "привет, мир", duration: 1.5, model: model)
    }
}

@Suite("Speech to text: multipart and /v1/audio/transcriptions")
struct SpeechRouteTests {

    private static let boundary = "----pocketd-test-7f3a"

    private static func form(_ fields: [(String, String)], file: Data?, filename: String = "clip.m4a") -> Data {
        var body = Data()
        for (name, value) in fields {
            body += Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".utf8)
        }
        if let file {
            body += Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\nContent-Type: audio/mp4\r\n\r\n".utf8)
            body += file
            body += Data("\r\n".utf8)
        }
        body += Data("--\(boundary)--\r\n".utf8)
        return body
    }

    /// Binary with the bytes most likely to confuse a text-based parser.
    private static let audio = Data([0x00, 0xFF, 0x0D, 0x0A, 0x2D, 0x2D, 0x0D, 0x0A, 0x0D, 0x0A, 0x41, 0x2D])

    // MARK: - Parsing

    @Test("fields and a binary file come out intact")
    func parses() throws {
        let body = Self.form([("model", "whisper-1"), ("language", "ru")], file: Self.audio)
        let form = try MultipartForm.parse(body, contentType: "multipart/form-data; boundary=\(Self.boundary)")
        #expect(form.value("model") == "whisper-1")
        #expect(form.value("language") == "ru")
        let file = try #require(form.part(named: "file"))
        #expect(file.filename == "clip.m4a")
        #expect(file.contentType == "audio/mp4")
        #expect(file.data == Self.audio)
    }

    @Test("a quoted boundary is read, and anything else is refused")
    func boundaries() throws {
        #expect(MultipartForm.boundary(from: "multipart/form-data; boundary=\"abc\"") == "abc")
        #expect(MultipartForm.boundary(from: "multipart/form-data; charset=utf-8; boundary=xyz") == "xyz")
        #expect(MultipartForm.boundary(from: "application/json") == nil)
        #expect(throws: MultipartForm.ParseError.notMultipart) {
            try MultipartForm.parse(Data(), contentType: "application/json")
        }
        #expect(throws: MultipartForm.ParseError.malformed) {
            try MultipartForm.parse(Data("--abc\r\nno end".utf8), contentType: "multipart/form-data; boundary=abc")
        }
    }

    @Test("the file keeps an extension AVFoundation can go by")
    func extensions() {
        let part = { (filename: String?, type: String?) in
            MultipartForm.Part(name: "file", filename: filename, contentType: type, data: Data())
        }
        #expect(InferenceServer.audioExtension(for: part("speech.MP3", nil)) == "mp3")
        #expect(InferenceServer.audioExtension(for: part("blob", "audio/x-wav")) == "wav")
        #expect(InferenceServer.audioExtension(for: part(nil, nil)) == "wav")
    }

    @Test("an OpenAI model name means whichever speech model is installed")
    func modelNames() {
        #expect(SpeechModels.model(named: "whisper-1") == nil)
        #expect(SpeechModels.model(named: "parakeet-tdt-0.6b-v2") == SpeechModels.parakeetV2)
        #expect(SpeechModels.model(forRepository: "fluidinference/parakeet-tdt-0.6b-v3-coreml") == SpeechModels.parakeetV3)
    }

    // MARK: - Over HTTP

    private func harness(_ transcriber: (any SpeechTranscriber)?) async throws -> TestServer {
        let configuration = ServerConfiguration(port: 0, binding: .loopback)
        let server = InferenceServer(
            configuration: configuration, engine: EchoEngine(), models: { [.echo] }, transcriber: transcriber
        )
        try await server.start()
        guard case let .running(_, port) = await server.currentState() else { throw TestFailure.serverDidNotStart }
        return TestServer(server: server, baseURL: URL(string: "http://127.0.0.1:\(port)")!, apiKey: configuration.apiKey)
    }

    private func post(_ harness: TestServer, _ body: Data, contentType: String? = nil) -> URLRequest {
        var request = URLRequest(url: harness.baseURL.appendingPathComponent("v1/audio/transcriptions"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(harness.apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue(contentType ?? "multipart/form-data; boundary=\(Self.boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        return request
    }

    @Test("transcribes, and answers in OpenAI's json")
    func transcribesJSON() async throws {
        let transcriber = FakeTranscriber()
        let harness = try await harness(transcriber)
        defer { Task { await harness.stop() } }

        let (status, data) = try await harness.send(post(harness, Self.form([("model", "whisper-1")], file: Self.audio)))
        #expect(status == 200)
        #expect(JSONValue.parse(String(decoding: data, as: UTF8.self)) == ["text": "привет, мир"])
        let received = await transcriber.received
        #expect(received.first?.bytes == Self.audio)
        #expect(received.first?.ext == "m4a")
        #expect(received.first?.model == SpeechModels.parakeetV3)
    }

    @Test("text and verbose_json")
    func otherFormats() async throws {
        let harness = try await harness(FakeTranscriber())
        defer { Task { await harness.stop() } }

        let (_, text) = try await harness.send(post(harness, Self.form([("response_format", "text")], file: Self.audio)))
        #expect(String(decoding: text, as: UTF8.self) == "привет, мир\n")

        let (_, verbose) = try await harness.send(post(harness, Self.form(
            [("response_format", "verbose_json"), ("language", "ru")], file: Self.audio
        )))
        let body = JSONValue.parse(String(decoding: verbose, as: UTF8.self))
        #expect(body?["text"] == "привет, мир")
        #expect(body?["duration"] == .double(1.5))
        #expect(body?["language"] == "ru")

        let (status, _) = try await harness.send(post(harness, Self.form([("response_format", "srt")], file: Self.audio)))
        #expect(status == 400)
    }

    @Test("a speech model named by id is the one used")
    func namedModel() async throws {
        let transcriber = FakeTranscriber(installed: [SpeechModels.parakeetV3, SpeechModels.parakeetV2])
        let harness = try await harness(transcriber)
        defer { Task { await harness.stop() } }

        _ = try await harness.send(post(harness, Self.form([("model", "parakeet-tdt-0.6b-v2")], file: Self.audio)))
        #expect(await transcriber.received.first?.model == SpeechModels.parakeetV2)
    }

    @Test("refusals: no file, not multipart, unreadable audio, nothing installed, no transcriber")
    func refusals() async throws {
        let harness = try await harness(FakeTranscriber())
        defer { Task { await harness.stop() } }

        #expect(try await harness.send(post(harness, Self.form([("model", "x")], file: nil))).0 == 400)
        #expect(try await harness.send(post(harness, Data("{}".utf8), contentType: "application/json")).0 == 400)
        #expect(try await harness.send(post(harness, Self.form([], file: Data("garbage".utf8)))).0 == 400)

        let empty = try await self.harness(FakeTranscriber(installed: []))
        defer { Task { await empty.stop() } }
        let (status, data) = try await empty.send(post(empty, Self.form([], file: Self.audio)))
        #expect(status == 404)
        #expect(String(decoding: data, as: UTF8.self).contains("model_not_found"))

        let none = try await self.harness(nil)
        defer { Task { await none.stop() } }
        #expect(try await none.send(post(none, Self.form([], file: Self.audio))).0 == 501)
    }
}
