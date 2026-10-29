import Foundation
import Testing
@testable import PocketdKit

/// Replies with fixed text, so a test can make the "model" write a tool call.
private actor ScriptedEngine: InferenceEngine {
    nonisolated var backendName: String { "scripted" }
    private let chunks: [String]
    private let delay: Duration
    private(set) var requests: [GenerationRequest] = []

    init(_ chunks: [String], delay: Duration = .zero) {
        self.chunks = chunks
        self.delay = delay
    }

    func loadedModel() async -> ModelRecord? { .echo }
    func load(model: ModelRecord) async throws {}
    func unload() async {}

    func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationEvent, any Error> {
        requests.append(request)
        let chunks = self.chunks
        let delay = self.delay
        return AsyncThrowingStream { continuation in
            let task = Task {
                for chunk in chunks {
                    if delay > .zero { try? await Task.sleep(for: delay) }
                    continuation.yield(.token(chunk))
                }
                continuation.yield(.finished(reason: .stop, usage: TokenUsage(promptTokens: 5, completionTokens: 7)))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@Suite("Anthropic Messages API over HTTP")
struct MessagesRouteTests {

    private func post(
        _ harness: TestServer,
        _ path: String,
        _ body: String,
        headers: [String: String]? = nil
    ) -> URLRequest {
        // Appended as a string: `appendingPathComponent` would escape the `?`
        // of `?beta=true`, which Claude Code puts on every request.
        var request = URLRequest(url: URL(string: harness.baseURL.absoluteString + path)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.setValue("claude-code-20250219,interleaved-thinking-2025-05-14", forHTTPHeaderField: "anthropic-beta")
        for (name, value) in headers ?? ["x-api-key": harness.apiKey] {
            request.setValue(value, forHTTPHeaderField: name)
        }
        request.httpBody = Data(body.utf8)
        return request
    }

    private func json(_ data: Data) throws -> JSONValue {
        try #require(JSONValue.parse(String(decoding: data, as: UTF8.self)))
    }

    /// `event:` names and decoded `data:` payloads, in order.
    private func events(_ lines: [String]) -> [(name: String, data: JSONValue)] {
        var out: [(String, JSONValue)] = []
        var name: String?
        for line in lines {
            if line.hasPrefix("event: ") {
                name = String(line.dropFirst(7))
            } else if line.hasPrefix("data: "), let current = name,
                      let data = JSONValue.parse(String(line.dropFirst(6))) {
                out.append((current, data))
                name = nil
            }
        }
        return out
    }

    @Test("answers a claude-* model name with the loaded model, not a 404")
    func nonStreaming() async throws {
        let harness = try await TestServer.start()
        defer { Task { await harness.stop() } }

        let (status, data) = try await harness.send(post(harness, "/v1/messages", """
        {"model":"claude-sonnet-4-5-20250929","max_tokens":1024,"metadata":{"user_id":"u"},
         "system":[{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}],
         "messages":[{"role":"user","content":"hello world"}]}
        """))
        #expect(status == 200)
        let body = try json(data)
        #expect(body["type"] == "message")
        #expect(body["role"] == "assistant")
        #expect(body["model"] == "claude-sonnet-4-5-20250929")
        #expect(body["id"]?.stringValue?.hasPrefix("msg_") == true)
        #expect(body["content"] == [["type": "text", "text": "hello world"]])
        #expect(body["stop_reason"] == "end_turn")
        #expect(body["stop_sequence"] == .null)
        #expect(body["usage"]?["input_tokens"] != nil)
        #expect(body["usage"]?["output_tokens"] != nil)
    }

    @Test("streams the events in Anthropic's order")
    func streaming() async throws {
        let harness = try await TestServer.start()
        defer { Task { await harness.stop() } }

        let lines = try await harness.lines(post(harness, "/v1/messages?beta=true", """
        {"model":"claude-haiku-4-5","max_tokens":100,"stream":true,
         "messages":[{"role":"user","content":"streaming works"}]}
        """))
        let parsed = events(lines)
        let names = parsed.map(\.name)
        #expect(names.first == "message_start")
        #expect(names.suffix(2) == ["message_delta", "message_stop"])
        #expect(names.filter { $0 == "content_block_start" }.count == 1)
        #expect(names.filter { $0 == "content_block_delta" }.count > 1, "one delta means the body was buffered")

        let start = parsed[0].data["message"]
        #expect(start?["usage"]?["output_tokens"] == 0)
        guard case let .int(inputTokens)? = start?["usage"]?["input_tokens"] else {
            Issue.record("message_start carries no input_tokens")
            return
        }
        #expect(inputTokens > 0)

        let text = parsed
            .filter { $0.name == "content_block_delta" }
            .compactMap { $0.data["delta"]?["text"]?.stringValue }
            .joined()
        #expect(text == "streaming works")
        #expect(parsed.first { $0.name == "message_delta" }?.data["delta"]?["stop_reason"] == "end_turn")
    }

    @Test("accepts the key as x-api-key or as a bearer token")
    func authHeaders() async throws {
        let harness = try await TestServer.start()
        defer { Task { await harness.stop() } }
        let body = #"{"model":"m","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}"#

        let (viaHeader, _) = try await harness.send(post(harness, "/v1/messages", body, headers: ["x-api-key": harness.apiKey]))
        #expect(viaHeader == 200)
        let (viaBearer, _) = try await harness.send(post(harness, "/v1/messages", body, headers: ["Authorization": "Bearer \(harness.apiKey)"]))
        #expect(viaBearer == 200)
        let (bothOneStale, _) = try await harness.send(post(harness, "/v1/messages", body, headers: [
            "Authorization": "Bearer pk-stale", "x-api-key": harness.apiKey,
        ]))
        #expect(bothOneStale == 200)
    }

    @Test("a wrong key is a 401 in Anthropic's error shape")
    func unauthorized() async throws {
        let harness = try await TestServer.start()
        defer { Task { await harness.stop() } }

        let (status, data) = try await harness.send(post(harness, "/v1/messages",
            #"{"model":"m","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}"#,
            headers: ["x-api-key": "pk-wrong"]))
        #expect(status == 401)
        let body = try json(data)
        #expect(body["type"] == "error")
        #expect(body["error"]?["type"] == "authentication_error")
    }

    @Test("a broken body is a 400 invalid_request_error")
    func malformed() async throws {
        let harness = try await TestServer.start()
        defer { Task { await harness.stop() } }

        let (status, data) = try await harness.send(post(harness, "/v1/messages", "{not json"))
        #expect(status == 400)
        #expect(try json(data)["error"]?["type"] == "invalid_request_error")
    }

    @Test("a prompt larger than the context is a 400 Claude Code can compact on")
    func promptTooLong() async throws {
        let harness = try await TestServer.start(
            configuration: ServerConfiguration(port: 0, binding: .loopback, maxContextTokens: 64)
        )
        defer { Task { await harness.stop() } }

        let long = String(repeating: "word ", count: 400)
        let (status, data) = try await harness.send(post(harness, "/v1/messages",
            #"{"model":"m","max_tokens":5,"stream":true,"messages":[{"role":"user","content":"\#(long)"}]}"#))
        #expect(status == 400)
        let body = try json(data)
        #expect(body["error"]?["type"] == "invalid_request_error")
        #expect(body["error"]?["message"]?.stringValue?.hasPrefix("prompt is too long") == true)
    }

    @Test("count_tokens answers with the prompt's size")
    func countTokens() async throws {
        let harness = try await TestServer.start()
        defer { Task { await harness.stop() } }

        let (status, data) = try await harness.send(post(harness, "/v1/messages/count_tokens", """
        {"model":"claude-sonnet-4","system":"sys","messages":[{"role":"user","content":"count these words please"}],
         "tools":[{"name":"Bash","description":"Run","input_schema":{"type":"object","properties":{}}}]}
        """))
        #expect(status == 200)
        let body = try json(data)
        guard case let .int(count)? = body["input_tokens"] else {
            Issue.record("input_tokens missing: \(body)")
            return
        }
        #expect(count > 0)
    }

    @Test("a tool call the model writes comes back as tool_use")
    func toolUseBuffered() async throws {
        let engine = ScriptedEngine(["Checking.\n", "<tool_call>\n{\"name\": \"Bash\", ", "\"arguments\": {\"command\": \"ls\"}}\n</tool_call>"])
        let harness = try await TestServer.start(engine: engine)
        defer { Task { await harness.stop() } }

        let (status, data) = try await harness.send(post(harness, "/v1/messages", """
        {"model":"claude-sonnet-4","max_tokens":200,
         "tools":[{"name":"Bash","description":"Run a command","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}],
         "messages":[{"role":"user","content":"list files"}]}
        """))
        #expect(status == 200)
        let body = try json(data)
        #expect(body["stop_reason"] == "tool_use")
        let content = try #require(body["content"]?.arrayValue)
        #expect(content.count == 2)
        #expect(content[0]["type"] == "text")
        #expect(content[1]["type"] == "tool_use")
        #expect(content[1]["name"] == "Bash")
        #expect(content[1]["input"] == ["command": "ls"])
        #expect(content[1]["id"]?.stringValue?.hasPrefix("toolu_") == true)

        let sent = try #require(await engine.requests.first)
        #expect(sent.clientTools?.tools.map(\.name) == ["Bash"])
    }

    @Test("a streamed tool call is a tool_use block with input_json_delta")
    func toolUseStreamed() async throws {
        let engine = ScriptedEngine(["<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/a\"}}</tool_call>"])
        let harness = try await TestServer.start(engine: engine)
        defer { Task { await harness.stop() } }

        let parsed = events(try await harness.lines(post(harness, "/v1/messages", """
        {"model":"claude-sonnet-4","max_tokens":200,"stream":true,
         "tools":[{"name":"Read","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}}}}],
         "messages":[{"role":"user","content":"read it"}]}
        """)))
        #expect(parsed.map(\.name) == [
            "message_start", "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ])
        #expect(parsed[1].data["content_block"]?["type"] == "tool_use")
        #expect(parsed[1].data["content_block"]?["input"] == [:])
        let partial = try #require(parsed[2].data["delta"]?["partial_json"]?.stringValue)
        #expect(JSONValue.parse(partial) == ["file_path": "/a"])
        #expect(parsed[4].data["delta"]?["stop_reason"] == "tool_use")
    }

    @Test("tool history reaches the engine as calls and results")
    func toolHistory() async throws {
        let engine = ScriptedEngine(["Done."])
        let harness = try await TestServer.start(engine: engine)
        defer { Task { await harness.stop() } }

        let (status, _) = try await harness.send(post(harness, "/v1/messages", """
        {"model":"claude-sonnet-4","max_tokens":50,"messages":[
          {"role":"user","content":"list"},
          {"role":"assistant","content":[{"type":"tool_use","id":"toolu_9","name":"Bash","input":{"command":"ls"}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_9","content":"a b"}]}]}
        """))
        #expect(status == 200)
        let sent = try #require(await engine.requests.first)
        #expect(sent.messages.map(\.role) == [.user, .assistant, .tool])
        #expect(sent.messages[1].toolCalls.first?.id == "toolu_9")
        #expect(sent.messages[2].toolCallID == "toolu_9")
        #expect(sent.messages[2].content == "a b")
    }

    @Test("concurrent requests wait their turn instead of bouncing")
    func queues() async throws {
        let harness = try await TestServer.start(
            configuration: ServerConfiguration(port: 0, binding: .loopback, maxConcurrentRequests: 1),
            engine: EchoEngine(delay: .milliseconds(20))
        )
        defer { Task { await harness.stop() } }

        let body = #"{"model":"claude-haiku-4-5","max_tokens":50,"messages":[{"role":"user","content":"queued answer"}]}"#
        let statuses = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<3 {
                group.addTask { try await harness.send(post(harness, "/v1/messages", body)).0 }
            }
            return try await group.reduce(into: []) { $0.append($1) }
        }
        #expect(statuses == [200, 200, 200])
    }

    @Test("the queue has a bottom, and past it the answer is 529")
    func queueLimit() async throws {
        let server = InferenceServer(
            configuration: ServerConfiguration(port: 0, binding: .loopback, maxConcurrentRequests: 1),
            engine: EchoEngine(),
            models: { [.echo] }
        )
        for _ in 0..<(1 + InferenceServer.anthropicQueueDepth) {
            #expect(await server.beginQueuedRequest())
        }
        #expect(await server.beginQueuedRequest() == false)
        // The other dialects see a busy phone, as they always have.
        #expect(await server.beginRequest() == false)
        #expect(InferenceServer.overloaded.code == 529)
    }

    @Test("an unserved Messages API path is a 404 in Anthropic's shape")
    func unknownPath() async throws {
        let harness = try await TestServer.start()
        defer { Task { await harness.stop() } }

        let (status, data) = try await harness.send(post(harness, "/v1/messages/batches", "{}"))
        #expect(status == 404)
        #expect(try json(data)["error"]?["type"] == "not_found_error")
    }
}
