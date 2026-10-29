import Foundation

/// A finished tool call, ready for the wire.
struct AnthropicToolUse: Sendable, Equatable {
    var id: String
    var name: String
    var input: JSONValue
}

/// The events of one streamed Anthropic message, in the order the API
/// defines and the SDKs enforce.
///
/// `message_start`, then for each content block `content_block_start`, its
/// deltas, `content_block_stop`, then `message_delta` with the stop reason and
/// `message_stop`. `ping` may appear anywhere after `message_start`. The SDKs
/// assemble the message from these strictly — a delta for a block that was
/// never started, or a second `message_start`, is an exception in the client —
/// so the ordering lives in one type rather than in the route's control flow.
struct AnthropicMessageStream: Sendable {
    let id: String
    let model: String
    private var textBlockOpen = false
    private var nextIndex = 0

    init(id: String, model: String) {
        self.id = id
        self.model = model
    }

    /// `message_start`. Its usage carries the real prompt size and no output.
    func start(inputTokens: Int) -> Data {
        Self.event("message_start", [
            "type": "message_start",
            "message": [
                "id": .string(id),
                "type": "message",
                "role": "assistant",
                "model": .string(model),
                "content": [],
                "stop_reason": nil,
                "stop_sequence": nil,
                "usage": ["input_tokens": .int(inputTokens), "output_tokens": 0],
            ],
        ])
    }

    /// Text for the open text block, opening one first if there is none.
    mutating func text(_ text: String) -> [Data] {
        guard !text.isEmpty else { return [] }
        var frames: [Data] = []
        if !textBlockOpen {
            frames.append(Self.event("content_block_start", [
                "type": "content_block_start",
                "index": .int(nextIndex),
                "content_block": ["type": "text", "text": ""],
            ]))
            textBlockOpen = true
        }
        frames.append(Self.event("content_block_delta", [
            "type": "content_block_delta",
            "index": .int(nextIndex),
            "delta": ["type": "text_delta", "text": .string(text)],
        ]))
        return frames
    }

    /// Closes the text block if one is open, then writes every tool call as
    /// its own block, then the stop reason and the end of the message.
    mutating func finish(
        toolUses: [AnthropicToolUse],
        stopReason: String,
        stopSequence: String?,
        outputTokens: Int
    ) -> [Data] {
        var frames: [Data] = []
        closeTextBlock(into: &frames)
        for use in toolUses {
            frames.append(Self.event("content_block_start", [
                "type": "content_block_start",
                "index": .int(nextIndex),
                "content_block": [
                    "type": "tool_use",
                    "id": .string(use.id),
                    "name": .string(use.name),
                    "input": [:],
                ],
            ]))
            frames.append(Self.event("content_block_delta", [
                "type": "content_block_delta",
                "index": .int(nextIndex),
                "delta": ["type": "input_json_delta", "partial_json": .string(use.input.serialized)],
            ]))
            frames.append(Self.event("content_block_stop", [
                "type": "content_block_stop",
                "index": .int(nextIndex),
            ]))
            nextIndex += 1
        }
        frames.append(Self.event("message_delta", [
            "type": "message_delta",
            "delta": [
                "stop_reason": .string(stopReason),
                "stop_sequence": stopSequence.map(JSONValue.string) ?? .null,
            ],
            "usage": ["output_tokens": .int(outputTokens)],
        ]))
        frames.append(Self.event("message_stop", ["type": "message_stop"]))
        return frames
    }

    private mutating func closeTextBlock(into frames: inout [Data]) {
        guard textBlockOpen else { return }
        frames.append(Self.event("content_block_stop", [
            "type": "content_block_stop",
            "index": .int(nextIndex),
        ]))
        textBlockOpen = false
        nextIndex += 1
    }

    static var ping: Data { event("ping", ["type": "ping"]) }

    static func error(type: String, message: String) -> Data {
        event("error", AnthropicMessage.errorBody(type: type, message: message))
    }

    /// One SSE event. Anthropic names every event, and the SDKs dispatch on the
    /// `event:` line as well as the `type` in the data.
    static func event(_ name: String, _ data: JSONValue) -> Data {
        Data("event: \(name)\ndata: \(data.serialized)\n\n".utf8)
    }
}

/// The non-streamed message, and the error body both modes share.
enum AnthropicMessage {
    static func body(
        id: String,
        model: String,
        text: String,
        toolUses: [AnthropicToolUse],
        stopReason: String,
        stopSequence: String?,
        inputTokens: Int,
        outputTokens: Int
    ) -> JSONValue {
        var content: [JSONValue] = []
        if !text.isEmpty {
            content.append(["type": "text", "text": .string(text)])
        }
        for use in toolUses {
            content.append([
                "type": "tool_use",
                "id": .string(use.id),
                "name": .string(use.name),
                "input": use.input,
            ])
        }
        return [
            "id": .string(id),
            "type": "message",
            "role": "assistant",
            "model": .string(model),
            "content": .array(content),
            "stop_reason": .string(stopReason),
            "stop_sequence": stopSequence.map(JSONValue.string) ?? .null,
            "usage": ["input_tokens": .int(inputTokens), "output_tokens": .int(outputTokens)],
        ]
    }

    static func errorBody(type: String, message: String) -> JSONValue {
        ["type": "error", "error": ["type": .string(type), "message": .string(message)]]
    }
}

/// Watches a stream of text for the client's stop sequences.
///
/// Holds back the tail that could still turn into one, so a stop sequence
/// split across two tokens is never half-shown. Which one was hit is the point:
/// `stop_reason: "stop_sequence"` comes with the sequence itself.
struct StopSequenceScanner: Sendable {
    private let stops: [String]
    private var pending = ""
    private(set) var hit: String?

    init(_ stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
    }

    /// Text that is certainly before any stop sequence. After a hit, nothing.
    mutating func feed(_ chunk: String) -> String {
        guard hit == nil else { return "" }
        guard !stops.isEmpty else { return chunk }
        pending += chunk
        if let match = stops
            .compactMap({ stop in pending.range(of: stop).map { (range: $0, stop: stop) } })
            .min(by: { $0.range.lowerBound < $1.range.lowerBound }) {
            hit = match.stop
            let before = String(pending[..<match.range.lowerBound])
            pending = ""
            return before
        }
        let keep = heldBackCount()
        let out = String(pending.dropLast(keep))
        pending = String(pending.suffix(keep))
        return out
    }

    mutating func finish() -> String {
        defer { pending = "" }
        return hit == nil ? pending : ""
    }

    private func heldBackCount() -> Int {
        let longest = stops.map(\.count).max() ?? 0
        for length in stride(from: min(longest - 1, pending.count), to: 0, by: -1) {
            let tail = pending.suffix(length)
            if stops.contains(where: { $0.hasPrefix(tail) }) { return length }
        }
        return 0
    }
}

/// The text side of one reply, from raw model output to what goes on the wire:
/// stop sequences, then reasoning, then tool-call markup, in that order.
///
/// Shared by the streamed and the buffered route so the two cannot disagree
/// about what a reply contained.
struct AnthropicReplyAssembler: Sendable {
    private var stops: StopSequenceScanner
    private var reasoning = ReasoningFilter()
    private var splitter: ToolMarkupSplitter
    private let tools: [ToolDefinition]
    private(set) var producedCharacters = 0

    init(stopSequences: [String], tools: [ToolDefinition]) {
        stops = StopSequenceScanner(stopSequences)
        splitter = ToolMarkupSplitter(holdsBareCalls: !tools.isEmpty)
        self.tools = tools
    }

    var stopSequence: String? { stops.hit }

    /// Text that can be shown now.
    mutating func feed(_ chunk: String) -> String {
        producedCharacters += chunk.count
        let beforeStop = stops.feed(chunk)
        return splitter.feed(reasoning.feed(beforeStop))
    }

    /// The last of the text, and the tool calls the reply made.
    mutating func finish(makeID: () -> String) -> (text: String, toolUses: [AnthropicToolUse]) {
        let tail = reasoning.feed(stops.finish()) + reasoning.finish()
        let fed = splitter.feed(tail)
        let (text, markup) = splitter.finish()
        guard !markup.isEmpty else { return (fed + text, []) }
        let parsed = ToolCallParser.parse(markup, tools: tools)
        let uses = parsed.calls.map { AnthropicToolUse(id: makeID(), name: $0.name, input: $0.arguments) }
        return (fed + text + parsed.text, uses)
    }
}
