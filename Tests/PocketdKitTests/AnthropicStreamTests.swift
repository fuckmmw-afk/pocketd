import Foundation
import Testing
@testable import PocketdKit

@Suite("Anthropic SSE serialization")
struct AnthropicStreamTests {

    /// One parsed SSE event: the `event:` name and the decoded `data:` JSON.
    struct Event: Equatable {
        var name: String
        var data: JSONValue
    }

    private func events(_ frames: [Data]) throws -> [Event] {
        try frames.map { frame in
            let text = String(decoding: frame, as: UTF8.self)
            // Every event is exactly two lines and a blank one.
            #expect(text.hasSuffix("\n\n"))
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            #expect(lines.count == 4)
            #expect(lines[0].hasPrefix("event: "))
            #expect(lines[1].hasPrefix("data: "))
            let name = String(lines[0].dropFirst("event: ".count))
            let data = try #require(JSONValue.parse(String(lines[1].dropFirst("data: ".count))))
            // The event line and the type inside agree.
            #expect(data["type"]?.stringValue == name)
            return Event(name: name, data: data)
        }
    }

    @Test("a text answer: start, one text block, delta, stop")
    func textAnswer() throws {
        var stream = AnthropicMessageStream(id: "msg_1", model: "claude-sonnet-4")
        var frames = [stream.start(inputTokens: 42)]
        frames += stream.text("Hel")
        frames += stream.text("lo")
        frames += stream.text("")
        frames += stream.finish(toolUses: [], stopReason: "end_turn", stopSequence: nil, outputTokens: 2)

        let parsed = try events(frames)
        #expect(parsed.map(\.name) == [
            "message_start",
            "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ])

        let start = parsed[0].data["message"]
        #expect(start?["id"] == "msg_1")
        #expect(start?["role"] == "assistant")
        #expect(start?["model"] == "claude-sonnet-4")
        #expect(start?["content"] == [])
        #expect(start?["stop_reason"] == .null)
        #expect(start?["usage"] == ["input_tokens": 42, "output_tokens": 0])

        #expect(parsed[1].data == ["type": "content_block_start", "index": 0, "content_block": ["type": "text", "text": ""]])
        #expect(parsed[2].data["delta"] == ["type": "text_delta", "text": "Hel"])
        #expect(parsed[3].data["delta"]?["text"] == "lo")
        #expect(parsed[4].data == ["type": "content_block_stop", "index": 0])
        #expect(parsed[5].data == [
            "type": "message_delta",
            "delta": ["stop_reason": "end_turn", "stop_sequence": nil],
            "usage": ["output_tokens": 2],
        ])
        #expect(parsed[6].data == ["type": "message_stop"])
    }

    @Test("text then tool_use: the text block closes before the tool block opens")
    func toolUseAnswer() throws {
        var stream = AnthropicMessageStream(id: "msg_2", model: "m")
        var frames = [stream.start(inputTokens: 10)]
        frames += stream.text("Checking.")
        frames += stream.finish(
            toolUses: [
                AnthropicToolUse(id: "toolu_a", name: "Bash", input: ["command": "ls"]),
                AnthropicToolUse(id: "toolu_b", name: "Read", input: ["file_path": "/x"]),
            ],
            stopReason: "tool_use",
            stopSequence: nil,
            outputTokens: 30
        )

        let parsed = try events(frames)
        #expect(parsed.map(\.name) == [
            "message_start",
            "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ])
        #expect(parsed[3].data["index"] == 0)

        #expect(parsed[4].data == [
            "type": "content_block_start",
            "index": 1,
            "content_block": ["type": "tool_use", "id": "toolu_a", "name": "Bash", "input": [:]],
        ])
        #expect(parsed[5].data["delta"]?["type"] == "input_json_delta")
        let partial = try #require(parsed[5].data["delta"]?["partial_json"]?.stringValue)
        #expect(JSONValue.parse(partial) == ["command": "ls"])
        #expect(parsed[6].data == ["type": "content_block_stop", "index": 1])
        #expect(parsed[7].data["index"] == 2)
        #expect(parsed[7].data["content_block"]?["id"] == "toolu_b")
        #expect(parsed[10].data["delta"]?["stop_reason"] == "tool_use")
    }

    @Test("a reply that is only a tool call starts its block at index 0")
    func toolOnly() throws {
        var stream = AnthropicMessageStream(id: "msg_3", model: "m")
        let parsed = try events([stream.start(inputTokens: 1)] + stream.finish(
            toolUses: [AnthropicToolUse(id: "toolu_a", name: "Bash", input: [:])],
            stopReason: "tool_use", stopSequence: nil, outputTokens: 5
        ))
        #expect(parsed.map(\.name) == [
            "message_start", "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ])
        #expect(parsed[1].data["index"] == 0)
    }

    @Test("ping and error events")
    func pingAndError() throws {
        let parsed = try events([
            AnthropicMessageStream.ping,
            AnthropicMessageStream.error(type: "overloaded_error", message: "busy"),
        ])
        #expect(parsed[0] == Event(name: "ping", data: ["type": "ping"]))
        #expect(parsed[1] == Event(name: "error", data: [
            "type": "error", "error": ["type": "overloaded_error", "message": "busy"],
        ]))
    }

    @Test("the non-streamed body")
    func messageBody() {
        let body = AnthropicMessage.body(
            id: "msg_x", model: "claude-haiku", text: "Hi",
            toolUses: [AnthropicToolUse(id: "toolu_1", name: "Bash", input: ["command": "ls"])],
            stopReason: "tool_use", stopSequence: nil, inputTokens: 7, outputTokens: 3
        )
        #expect(body == [
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku",
            "content": [
                ["type": "text", "text": "Hi"],
                ["type": "tool_use", "id": "toolu_1", "name": "Bash", "input": ["command": "ls"]],
            ],
            "stop_reason": "tool_use",
            "stop_sequence": nil,
            "usage": ["input_tokens": 7, "output_tokens": 3],
        ])
    }

    // MARK: - From model output to events

    private func assemble(_ chunks: [String], stops: [String] = [], tools: [ToolDefinition] = []) -> (streamed: String, tail: String, uses: [AnthropicToolUse], stop: String?) {
        var assembler = AnthropicReplyAssembler(stopSequences: stops, tools: tools)
        let streamed = chunks.map { assembler.feed($0) }.joined()
        var counter = 0
        let (tail, uses) = assembler.finish {
            counter += 1
            return "toolu_\(counter)"
        }
        return (streamed, tail, uses, assembler.stopSequence)
    }

    @Test("reasoning is cut out of the text")
    func stripsThinking() {
        let result = assemble(["<think>", "plan things", "</think>\n\n", "Answer."])
        #expect(result.streamed + result.tail == "Answer.")
    }

    @Test("a stop sequence ends the text and is reported")
    func stopSequence() {
        let result = assemble(["one two ##", "# three"], stops: ["###"])
        #expect(result.streamed + result.tail == "one two ")
        #expect(result.stop == "###")
        #expect(InferenceServer.stopReason(uses: [], stopSequence: result.stop, finish: .stop) == "stop_sequence")
    }

    @Test("a tool call in the output becomes a tool_use with an id")
    func toolCallToToolUse() {
        let tools = [ToolDefinition(name: "Bash")]
        let result = assemble(
            ["Let me check.\n", "<tool_call>\n{\"name\": \"Bash\", ", "\"arguments\": {\"command\": \"ls\"}}\n</tool_call>"],
            tools: tools
        )
        #expect(result.streamed == "Let me check.\n")
        #expect(result.uses == [AnthropicToolUse(id: "toolu_1", name: "Bash", input: ["command": "ls"])])
        #expect(InferenceServer.stopReason(uses: result.uses, stopSequence: nil, finish: .stop) == "tool_use")
    }

    @Test("max_tokens and end_turn")
    func stopReasons() {
        #expect(InferenceServer.stopReason(uses: [], stopSequence: nil, finish: .length) == "max_tokens")
        #expect(InferenceServer.stopReason(uses: [], stopSequence: nil, finish: .stop) == "end_turn")
    }
}
