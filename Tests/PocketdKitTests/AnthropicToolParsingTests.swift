import Foundation
import Testing
@testable import PocketdKit

@Suite("Tool calls in model output → tool_use")
struct AnthropicToolParsingTests {

    private let tools = [
        ToolDefinition(name: "Bash", parameters: [
            "type": "object",
            "properties": ["command": ["type": "string"], "timeout": ["type": "integer"]],
        ]),
        ToolDefinition(name: "Read", parameters: [
            "type": "object",
            "properties": ["file_path": ["type": "string"], "limit": ["type": "integer"]],
        ]),
    ]

    @Test("Hermes / Qwen <tool_call> JSON")
    func hermes() {
        let parsed = ToolCallParser.parse("""
        I'll list it.
        <tool_call>
        {"name": "Bash", "arguments": {"command": "ls -la", "timeout": 10}}
        </tool_call>
        """, tools: tools)
        #expect(parsed.text == "I'll list it.\n")
        #expect(parsed.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls -la", "timeout": 10])])
    }

    @Test("several calls in a row, and arguments written as a JSON string")
    func severalCalls() {
        let parsed = ToolCallParser.parse("""
        <tool_call>
        {"name": "Read", "arguments": "{\\"file_path\\": \\"/a\\"}"}
        </tool_call>
        <tool_call>
        {"name": "Read", "arguments": {"file_path": "/b"}}
        </tool_call>
        """, tools: tools)
        #expect(parsed.text.isEmpty)
        #expect(parsed.calls.map(\.arguments) == [["file_path": "/a"], ["file_path": "/b"]])
    }

    @Test("an unclosed <tool_call> at the end of the reply still counts")
    func unclosed() {
        let parsed = ToolCallParser.parse(#"<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}"#, tools: tools)
        #expect(parsed.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "pwd"])])
    }

    @Test("Qwen3-Coder XML parameters are typed by the schema")
    func qwenCoderXML() {
        let parsed = ToolCallParser.parse("""
        <tool_call>
        <function=Read>
        <parameter=file_path>
        /etc/hosts
        </parameter>
        <parameter=limit>
        20
        </parameter>
        </function>
        </tool_call>
        """, tools: tools)
        #expect(parsed.calls == [ParsedToolCall(name: "Read", arguments: ["file_path": "/etc/hosts", "limit": 20])])
    }

    @Test("a string parameter that looks like a number stays a string")
    func xmlStringStaysString() {
        let parsed = ToolCallParser.parse(
            "<function=Bash>\n<parameter=command>\n42\n</parameter>\n</function>",
            tools: tools
        )
        #expect(parsed.calls.first?.arguments["command"] == "42")
    }

    @Test("Mistral [TOOL_CALLS], both the array and the [ARGS] form")
    func mistral() {
        let array = ToolCallParser.parse(#"[TOOL_CALLS][{"name": "Bash", "arguments": {"command": "ls"}}]"#, tools: tools)
        #expect(array.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls"])])

        let args = ToolCallParser.parse(#"[TOOL_CALLS]Bash[ARGS]{"command": "ls"}"#, tools: tools)
        #expect(args.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls"])])
    }

    @Test("Llama 3: <|python_tag|> and a bare call object")
    func llama() {
        let tagged = ToolCallParser.parse(#"<|python_tag|>{"name": "Bash", "parameters": {"command": "ls"}}"#, tools: tools)
        #expect(tagged.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls"])])

        let bare = ToolCallParser.parse(#"{"name": "Bash", "parameters": {"command": "ls"}}"#, tools: tools)
        #expect(bare.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls"])])
        #expect(bare.text.isEmpty)
    }

    @Test("a fenced call object counts, for an offered tool")
    func fenced() {
        let parsed = ToolCallParser.parse("```json\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/x\"}}\n```", tools: tools)
        #expect(parsed.calls == [ParsedToolCall(name: "Read", arguments: ["file_path": "/x"])])
    }

    @Test("a JSON answer that names no offered tool stays an answer")
    func jsonAnswerIsNotACall() {
        let answer = #"{"name": "Alice", "age": 30}"#
        let parsed = ToolCallParser.parse(answer, tools: tools)
        #expect(parsed.calls.isEmpty)
        #expect(parsed.text == answer)
    }

    @Test("Phi-4 <|tool_call|> and Gemma 4 <|tool_call>")
    func phiAndGemma() {
        let phi = ToolCallParser.parse(#"<|tool_call|>[{"name": "Bash", "arguments": {"command": "ls"}}]<|/tool_call|>"#, tools: tools)
        #expect(phi.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls"])])

        let gemma = ToolCallParser.parse(#"<|tool_call>call:Bash{command:<|"|>ls -la<|"|>,timeout:5}<tool_call|>"#, tools: tools)
        #expect(gemma.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls -la", "timeout": 5])])
    }

    @Test("LFM2's Python-style calls")
    func lfm2() {
        let parsed = ToolCallParser.parse(
            "Sure.<|tool_call_start|>[Bash(command='ls -la', timeout=5), Read(file_path=\"/a b\")]<|tool_call_end|>",
            tools: tools
        )
        #expect(parsed.text == "Sure.")
        #expect(parsed.calls == [
            ParsedToolCall(name: "Bash", arguments: ["command": "ls -la", "timeout": 5]),
            ParsedToolCall(name: "Read", arguments: ["file_path": "/a b"]),
        ])

        let literals = ToolCallParser.parse(
            "<|tool_call_start|>[f(flag=True, none=None, items=['a', 2], opts={'k': 'v'}, x=1.5)]<|tool_call_end|>",
            tools: tools
        )
        #expect(literals.calls.first?.arguments == [
            "flag": true, "none": nil, "items": ["a", 2], "opts": ["k": "v"], "x": .double(1.5),
        ])
    }

    @Test("a flattened call reads back as the same call")
    func hermesRoundTrip() {
        let written = ChatTemplateInput.hermesCall(name: "Bash", arguments: ["command": "echo \"hi\""])
        #expect(ToolCallParser.parse(written, tools: tools).calls == [
            ParsedToolCall(name: "Bash", arguments: ["command": "echo \"hi\""]),
        ])
    }

    @Test("a marker with a body that will not parse is left as text")
    func brokenCallIsText() {
        let parsed = ToolCallParser.parse("<tool_call>\nnot json at all\n</tool_call>", tools: tools)
        #expect(parsed.calls.isEmpty)
        #expect(parsed.text.contains("not json at all"))
    }

    @Test("trailing commas are forgiven")
    func trailingComma() {
        let parsed = ToolCallParser.parse(#"<tool_call>{"name": "Bash", "arguments": {"command": "ls",},}</tool_call>"#, tools: tools)
        #expect(parsed.calls == [ParsedToolCall(name: "Bash", arguments: ["command": "ls"])])
    }

    @Test("plain text has no calls")
    func plain() {
        let parsed = ToolCallParser.parse("Just an answer.", tools: tools)
        #expect(parsed.calls.isEmpty)
        #expect(parsed.text == "Just an answer.")
    }

    // MARK: - Streaming

    private func split(_ chunks: [String], holdsBareCalls: Bool = true) -> (shown: String, text: String, markup: String) {
        var splitter = ToolMarkupSplitter(holdsBareCalls: holdsBareCalls)
        let shown = chunks.map { splitter.feed($0) }.joined()
        let (text, markup) = splitter.finish()
        return (shown, text, markup)
    }

    @Test("text streams; a marker split across tokens is never shown")
    func splitterHoldsMarker() {
        let result = split(["Sure", ". <too", "l_call>\n{\"name\"", ": \"Bash\"}</tool_call>"])
        #expect(result.shown == "Sure. ")
        #expect(result.markup.hasPrefix("<tool_call>"))
        #expect(!result.shown.contains("<"))
    }

    @Test("a reply that opens with { is held until the end")
    func splitterHoldsBareJSON() {
        let result = split(["  {\"name\":", " \"Bash\", \"parameters\": {}}"])
        #expect(result.shown.isEmpty)
        #expect(result.markup.contains("\"Bash\""))
    }

    @Test("without tools, a JSON answer streams as it comes")
    func splitterWithoutTools() {
        let result = split(["{\"a\":", " 1}"], holdsBareCalls: false)
        #expect(result.shown == "{\"a\": 1}")
    }

    @Test("a < that is not a marker is released")
    func splitterReleasesLessThan() {
        let result = split(["a < b", " and done"])
        #expect(result.shown + result.text == "a < b and done")
        #expect(result.markup.isEmpty)
    }
}
