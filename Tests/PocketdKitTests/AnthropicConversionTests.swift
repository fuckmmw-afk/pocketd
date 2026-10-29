import Foundation
import Testing
@testable import PocketdKit

@Suite("Anthropic → internal conversion")
struct AnthropicConversionTests {

    private func decode(_ json: String) throws -> Anthropic.MessagesRequest {
        try JSONDecoder().decode(Anthropic.MessagesRequest.self, from: Data(json.utf8))
    }

    private func convert(_ json: String, acceptsImages: Bool = false) throws -> [ChatMessage] {
        let request = try decode(json)
        return AnthropicConversion.chatMessages(
            system: request.system?.text,
            messages: request.messages,
            acceptsImages: acceptsImages
        )
    }

    @Test("a system prompt sent as a string becomes the first turn")
    func systemString() throws {
        let messages = try convert("""
        {"model":"claude-sonnet-4","max_tokens":10,"system":"Be brief.",
         "messages":[{"role":"user","content":"hi"}]}
        """)
        #expect(messages.map(\.role) == [.system, .user])
        #expect(messages[0].content == "Be brief.")
        #expect(messages[1].content == "hi")
    }

    @Test("a system prompt sent as blocks is joined, and cache_control is ignored")
    func systemBlocks() throws {
        let messages = try convert("""
        {"model":"claude-sonnet-4","max_tokens":10,
         "system":[{"type":"text","text":"You are Claude Code."},
                   {"type":"text","text":"Use tools.","cache_control":{"type":"ephemeral"}}],
         "messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]}
        """)
        #expect(messages.first?.role == .system)
        #expect(messages.first?.content == "You are Claude Code.\n\nUse tools.")
    }

    @Test("unknown top-level fields and unknown blocks do not fail the request")
    func ignoresUnknowns() throws {
        let request = try decode("""
        {"model":"claude-opus-4","max_tokens":10,"metadata":{"user_id":"x"},"service_tier":"auto",
         "thinking":{"type":"enabled","budget_tokens":1024},"container":"c","context_management":{},
         "messages":[{"role":"user","content":"hi"},
                     {"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"s"},
                                                    {"type":"redacted_thinking","data":"x"},
                                                    {"type":"text","text":"hello"}]}]}
        """)
        let messages = AnthropicConversion.chatMessages(system: nil, messages: request.messages, acceptsImages: false)
        #expect(messages.map(\.role) == [.user, .assistant])
        #expect(messages[1].content == "hello")
    }

    @Test("tool_use becomes a call on the assistant turn, tool_result a tool turn")
    func toolRoundTrip() throws {
        let messages = try convert("""
        {"model":"m","max_tokens":10,"messages":[
          {"role":"user","content":"What is in /tmp?"},
          {"role":"assistant","content":[
            {"type":"text","text":"Let me look."},
            {"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls /tmp","timeout":5}},
            {"type":"tool_use","id":"toolu_2","name":"Read","input":{"file_path":"/tmp/a"}}]},
          {"role":"user","content":[
            {"type":"tool_result","tool_use_id":"toolu_1","content":"a\\nb"},
            {"type":"tool_result","tool_use_id":"toolu_2","is_error":true,
             "content":[{"type":"text","text":"No such file"}]},
            {"type":"text","text":"Thanks"}]}
        ]}
        """)
        #expect(messages.map(\.role) == [.user, .assistant, .tool, .tool, .user])

        let assistant = messages[1]
        #expect(assistant.content == "Let me look.")
        #expect(assistant.toolCalls == [
            ToolCall(id: "toolu_1", name: "Bash", arguments: ["command": "ls /tmp", "timeout": 5]),
            ToolCall(id: "toolu_2", name: "Read", arguments: ["file_path": "/tmp/a"]),
        ])

        #expect(messages[2].toolCallID == "toolu_1")
        #expect(messages[2].content == "a\nb")
        #expect(messages[3].toolCallID == "toolu_2")
        #expect(messages[3].content == "Error: No such file")
        #expect(messages[4].content == "Thanks")
    }

    @Test("an image for a model that cannot see becomes a placeholder")
    func imageOmitted() throws {
        let messages = try convert("""
        {"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
          {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}},
          {"type":"text","text":"What is this?"}]}]}
        """)
        #expect(messages.count == 1)
        #expect(messages[0].content == "[image omitted]\n\nWhat is this?")
        #expect(messages[0].images.isEmpty)
    }

    @Test("an image for a vision model is passed through as bytes")
    func imageKept() throws {
        let messages = try convert("""
        {"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
          {"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}},
          {"type":"text","text":"What is this?"}]}]}
        """, acceptsImages: true)
        #expect(messages[0].images == [Data("hello".utf8)])
        #expect(messages[0].content == "What is this?")
    }

    @Test("an image by URL is never fetched")
    func imageURLOmitted() throws {
        let messages = try convert("""
        {"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
          {"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}
        """, acceptsImages: true)
        #expect(messages[0].content == "[image omitted]")
        #expect(messages[0].images.isEmpty)
    }

    @Test("an image inside a tool result becomes a placeholder too")
    func imageInToolResult() throws {
        let messages = try convert("""
        {"model":"m","max_tokens":10,"messages":[
          {"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Read","input":{}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[
            {"type":"text","text":"screenshot:"},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]}]}]}
        """)
        #expect(messages[1].content == "screenshot:\n[image omitted]")
    }

    @Test("tools and tool_choice are carried over; server tools are dropped")
    func tools() throws {
        let request = try decode("""
        {"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],
         "tools":[{"name":"get_weather","description":"Weather","input_schema":
                    {"type":"object","properties":{"city":{"type":"string"}},"required":["city"],
                     "additionalProperties":false}},
                  {"type":"web_search_20250305","name":"web_search","max_uses":5}],
         "tool_choice":{"type":"tool","name":"get_weather"}}
        """)
        let tools = AnthropicConversion.clientTools(request.tools, choice: request.tool_choice)
        #expect(tools.tools.map(\.name) == ["get_weather"])
        #expect(tools.tools[0].parameters["additionalProperties"] == .bool(false))
        #expect(tools.choice == .tool("get_weather"))
    }

    @Test("tool_choice any and none")
    func toolChoices() {
        let tool = Anthropic.Tool(name: "t", description: nil, input_schema: ["type": "object"], type: nil)
        #expect(AnthropicConversion.clientTools([tool], choice: .init(type: "any")).choice == .any)
        #expect(AnthropicConversion.clientTools([tool], choice: .init(type: "none")).choice == .none)
        #expect(AnthropicConversion.clientTools([tool], choice: .init(type: "none")).offered.isEmpty)
        #expect(AnthropicConversion.clientTools([tool], choice: nil).choice == .auto)
    }

    @Test("sampling fields map onto generation options")
    func options() throws {
        let request = try decode("""
        {"model":"m","max_tokens":321,"temperature":0.2,"top_p":0.9,"top_k":40,
         "stop_sequences":["###"],"stream":true,"messages":[{"role":"user","content":"hi"}]}
        """)
        let options = AnthropicConversion.options(request)
        #expect(options.maxTokens == 321)
        #expect(options.temperature == 0.2)
        #expect(options.topP == 0.9)
        #expect(options.topK == 40)
        #expect(request.stop_sequences == ["###"])
        #expect(request.stream == true)
    }

    @Test("ids have Anthropic's prefixes")
    func identifiers() {
        let id = AnthropicConversion.identifier(prefix: "toolu_")
        #expect(id.hasPrefix("toolu_"))
        #expect(id.count == "toolu_".count + 24)
        #expect(AnthropicConversion.identifier(prefix: "msg_") != AnthropicConversion.identifier(prefix: "msg_"))
    }

    // MARK: - What the chat template is handed

    @Test("native template input carries tool_calls and tool_call_id")
    func nativeTemplateInput() throws {
        let messages = try convert("""
        {"model":"m","max_tokens":10,"system":"sys","messages":[
          {"role":"user","content":"hi"},
          {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}]}
        """)
        let input = ChatTemplateInput.nativeMessages(messages)
        #expect(input.count == 4)
        #expect(input[2]["tool_calls"] == [[
            "id": "toolu_1",
            "type": "function",
            "function": ["name": "Bash", "arguments": ["command": "ls"]],
        ]])
        #expect(input[3]["role"] == "tool")
        #expect(input[3]["tool_call_id"] == "toolu_1")
        #expect(input[3]["name"] == "Bash")

        let stringArguments = ChatTemplateInput.nativeMessages(messages, argumentsAsObjects: false)
        #expect(stringArguments[2]["tool_calls"]?.arrayValue?.first?["function"]?["arguments"] == #"{"command":"ls"}"#)
    }

    @Test("tool_choice any is spelled out in the system turn")
    func toolChoiceInstruction() {
        let input = ChatTemplateInput.nativeMessages([.system("sys"), .user("hi")], choice: .any)
        #expect(input[0]["content"]?.stringValue?.hasPrefix("sys\n\nYou must call at least one") == true)
    }

    @Test("a flattened conversation alternates roles and describes the tools")
    func flattened() {
        let messages: [ChatMessage] = [
            .system("sys"),
            .user("hi"),
            ChatMessage(role: .assistant, content: "Looking.", toolCalls: [
                ToolCall(id: "a", name: "Bash", arguments: ["command": "ls"]),
            ]),
            ChatMessage(role: .tool, content: "x", toolCallID: "a"),
            .user("and?"),
        ]
        let tools = ClientTools(tools: [ToolDefinition(name: "Bash", description: "Run")])
        let flat = ChatTemplateInput.flattened(messages, tools: tools)
        #expect(flat.map(\.role) == [.system, .user, .assistant, .user])
        #expect(flat[0].content.hasPrefix("sys\n\n# Tools"))
        #expect(flat[0].content.contains(#""name":"Bash""#))
        #expect(flat[2].content == "Looking.\n<tool_call>\n{\"name\": \"Bash\", \"arguments\": {\"command\":\"ls\"}}\n</tool_call>")
        #expect(flat[3].content == "<tool_response>\nx\n</tool_response>\n\nand?")
    }

    @Test("rendering the same tools twice is byte-identical")
    func deterministicTools() {
        let schema: JSONValue = ["type": "object", "properties": ["b": ["type": "string"], "a": ["type": "integer"]]]
        let first = ChatTemplateInput.nativeTools([ToolDefinition(name: "t", parameters: schema)]).map(\.serialized)
        let second = ChatTemplateInput.nativeTools([ToolDefinition(name: "t", parameters: schema)]).map(\.serialized)
        #expect(first == second)
        #expect(first[0].contains(#""properties":{"a":{"type":"integer"},"b":{"type":"string"}}"#))
    }

    @Test("a transcript saved before tool calls existed still decodes, and saves the same")
    func chatMessageCompatibility() throws {
        let old = Data(#"{"role":"user","content":"hi","images":[]}"#.utf8)
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: old)
        #expect(decoded.toolCalls.isEmpty)
        #expect(decoded.toolCallID == nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let reencoded = String(decoding: try encoder.encode(decoded), as: UTF8.self)
        #expect(!reencoded.contains("toolCalls"))
        #expect(!reencoded.contains("toolCallID"))

        let withCall = ChatMessage(role: .assistant, content: "", toolCalls: [ToolCall(id: "1", name: "x", arguments: [:])])
        let roundTripped = try JSONDecoder().decode(ChatMessage.self, from: try encoder.encode(withCall))
        #expect(roundTripped == withCall)
    }
}

@Suite("Claude Code setup commands")
struct ClaudeCodeSetupTests {
    @Test("the four exports, with the address and key filled in")
    func exports() {
        #expect(ClaudeCodeSetup.exports(host: "192.168.1.42", port: 11434, apiKey: "pk-abc") == """
        export ANTHROPIC_BASE_URL=http://192.168.1.42:11434
        export ANTHROPIC_AUTH_TOKEN=pk-abc
        export API_TIMEOUT_MS=600000
        export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
        """)
    }

    @Test("a server with no key still gives Claude Code a token to send")
    func noKey() {
        #expect(ClaudeCodeSetup.exports(host: "h", port: 1, apiKey: nil).contains("ANTHROPIC_AUTH_TOKEN=not-needed"))
    }
}
