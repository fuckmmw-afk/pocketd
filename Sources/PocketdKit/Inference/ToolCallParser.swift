import Foundation

/// A tool call recovered from a model's text, before it has been given an id.
public struct ParsedToolCall: Sendable, Equatable {
    public var name: String
    /// Always a JSON object.
    public var arguments: JSONValue

    public init(name: String, arguments: JSONValue) {
        self.name = name
        self.arguments = arguments
    }
}

/// Reads the tool calls a model wrote back out of the text it wrote them in.
///
/// A small model's tool calls are text — there is no structured channel below
/// the chat template — and every family writes them its own way. This knows
/// the shapes the models that fit on a phone actually produce:
///
/// - `<tool_call>{"name": …, "arguments": …}</tool_call>`: Hermes, Qwen 2.5
///   and 3, SmolLM3, and anything prompted with `ChatTemplateInput.flattened`.
/// - `<tool_call><function=name><parameter=key>value</parameter></function>`:
///   Qwen3-Coder, with or without the outer tag.
/// - `<|tool_call|>[…]<|/tool_call|>`: Phi-4-mini.
/// - `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`: Gemma 4.
/// - `[TOOL_CALLS][…]` and `[TOOL_CALLS]name[ARGS]{…}`: Mistral, both eras.
/// - `<|tool_call_start|>[name(key='value')]<|tool_call_end|>`: LFM2, which
///   writes calls as Python.
/// - `<|python_tag|>{…}`, or a reply that is nothing but `{"name": …,
///   "parameters": …}`: Llama 3.x. The bare form, and a bare object in a code
///   fence, only count when the name is a tool that was offered — otherwise a
///   model answering with JSON would have its answer turned into a call.
///
/// Anything that is not a call stays text. A marker whose body will not parse
/// is left as text too: a client handed a call it cannot decode has nothing to
/// run, while text at least shows the person what the model tried.
public enum ToolCallParser {
    /// Strings whose appearance means "a tool call starts here".
    public static let markers = [
        "<tool_call>", "<|tool_call|>", "<|tool_call>", "<|tool_call_start|>", "[TOOL_CALLS]", "<|python_tag|>",
        "<function=",
    ]

    public static func parse(_ text: String, tools: [ToolDefinition]) -> (text: String, calls: [ParsedToolCall]) {
        let known = Dictionary(tools.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        guard let (start, marker) = earliestMarker(in: text) else {
            if let calls = bareCalls(text, known: known), !calls.isEmpty {
                return ("", calls)
            }
            return (text, [])
        }

        var prose = String(text[..<start])
        var calls: [ParsedToolCall] = []
        var rest = Substring(text[start...])
        var current: String? = marker

        while let marker = current {
            let body = rest.dropFirst(marker.count)
            let (parsed, remainder, ok) = parseBlock(after: marker, body: body, known: known)
            if ok, !parsed.isEmpty {
                calls += parsed
            } else {
                // Not a call after all: keep what the model wrote.
                prose += rest[..<remainder.startIndex]
            }
            rest = remainder
            if let (next, nextMarker) = earliestMarker(in: rest) {
                prose += rest[..<next]
                rest = rest[next...]
                current = nextMarker
            } else {
                prose += rest
                current = nil
            }
        }
        return (prose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : prose, calls)
    }

    // MARK: - Markers

    static func earliestMarker<S: StringProtocol>(in text: S) -> (S.Index, String)? where S.SubSequence == Substring {
        var best: (S.Index, String)?
        for marker in markers {
            guard let range = text.range(of: marker) else { continue }
            if best == nil || range.lowerBound < best!.0 {
                best = (range.lowerBound, marker)
            }
        }
        return best
    }

    /// One marker's worth of calls, and what follows it.
    private static func parseBlock(
        after marker: String,
        body: Substring,
        known: [String: ToolDefinition]
    ) -> (calls: [ParsedToolCall], remainder: Substring, ok: Bool) {
        switch marker {
        case "<tool_call>":
            let (inner, rest) = enclosed(body, close: "</tool_call>")
            let trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("<function=") {
                let calls = xmlFunctions(trimmed, known: known)
                return (calls, rest, !calls.isEmpty)
            }
            let calls = jsonCalls(trimmed)
            return (calls, rest, !calls.isEmpty)

        case "<function=":
            let (inner, rest) = enclosed(body, close: "</function>")
            let calls = xmlFunctions("<function=" + inner + "</function>", known: known)
            return (calls, rest, !calls.isEmpty)

        case "<|tool_call|>":
            let (inner, rest) = enclosed(body, close: "<|/tool_call|>")
            let calls = jsonCalls(inner.trimmingCharacters(in: .whitespacesAndNewlines))
            return (calls, rest, !calls.isEmpty)

        case "<|tool_call>":
            let (inner, rest) = enclosed(body, close: "<tool_call|>")
            let calls = gemmaCalls(inner.trimmingCharacters(in: .whitespacesAndNewlines))
            return (calls, rest, !calls.isEmpty)

        case "<|tool_call_start|>":
            let (inner, rest) = enclosed(body, close: "<|tool_call_end|>")
            let trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines)
            var python = PythonCallReader(trimmed)
            let calls = trimmed.hasPrefix("[{") || trimmed.hasPrefix("{") ? jsonCalls(trimmed) : (python.calls() ?? [])
            return (calls, rest, !calls.isEmpty)

        case "[TOOL_CALLS]":
            // Runs to the end, or to the next `[TOOL_CALLS]`; Mistral writes
            // nothing but calls once it has started.
            let end = body.range(of: "[TOOL_CALLS]")?.lowerBound ?? body.endIndex
            let inner = body[..<end].trimmingCharacters(in: .whitespacesAndNewlines)
            let calls = mistralCalls(inner)
            return (calls, body[end...], !calls.isEmpty)

        case "<|python_tag|>":
            let end = body.range(of: "<|eom_id|>")?.lowerBound ?? body.endIndex
            let inner = body[..<end].trimmingCharacters(in: .whitespacesAndNewlines)
            let calls = inner
                .split(separator: ";")
                .flatMap { jsonCalls($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            let rest = end == body.endIndex ? body[end...] : body[body.index(end, offsetBy: "<|eom_id|>".count)...]
            return (calls, rest, !calls.isEmpty)

        default:
            return ([], body, false)
        }
    }

    /// The text up to `close` and what follows it. An unclosed block runs to
    /// the end, which is what a model that hit its end-of-turn token right
    /// after the JSON leaves behind.
    private static func enclosed(_ body: Substring, close: String) -> (Substring, Substring) {
        guard let range = body.range(of: close) else { return (body, body[body.endIndex...]) }
        return (body[..<range.lowerBound], body[range.upperBound...])
    }

    // MARK: - JSON shapes

    /// A call object, an array of them, or a code-fenced one.
    static func jsonCalls(_ text: String) -> [ParsedToolCall] {
        guard let value = lenientJSON(unfenced(text)) else { return [] }
        switch value {
        case let .array(items):
            let calls = items.compactMap(call(from:))
            return calls.count == items.count ? calls : []
        case .object:
            return call(from: value).map { [$0] } ?? []
        default:
            return []
        }
    }

    /// `{"name", "arguments"}` and the spellings around it: `parameters`
    /// (Llama), `input` (Anthropic-tuned), and OpenAI's `function` wrapper.
    static func call(from value: JSONValue) -> ParsedToolCall? {
        guard case let .object(fields) = value else { return nil }
        if let function = fields["function"], case .object = function, fields["name"] == nil {
            return call(from: function)
        }
        guard let name = fields["name"]?.stringValue, !name.isEmpty else { return nil }
        let raw = fields["arguments"] ?? fields["parameters"] ?? fields["input"] ?? .object([:])
        guard let arguments = argumentsObject(raw) else { return nil }
        return ParsedToolCall(name: name, arguments: arguments)
    }

    /// Arguments as an object. A string is parsed, because several templates
    /// teach the model to write them as an escaped JSON string.
    static func argumentsObject(_ raw: JSONValue) -> JSONValue? {
        switch raw {
        case .object:
            return raw
        case .null:
            return .object([:])
        case let .string(text):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { return .object([:]) }
            if let parsed = lenientJSON(trimmed), case .object = parsed { return parsed }
            return nil
        default:
            return nil
        }
    }

    /// Mistral: an array of calls, or `name[ARGS]{…}` (v11+), or `name{…}`.
    private static func mistralCalls(_ text: String) -> [ParsedToolCall] {
        if text.hasPrefix("[") || text.hasPrefix("{") { return jsonCalls(text) }
        if let args = text.range(of: "[ARGS]") {
            let name = text[..<args.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            let json = text[args.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty, let arguments = lenientJSON(json).flatMap(argumentsObject) else { return [] }
            return [ParsedToolCall(name: name, arguments: arguments)]
        }
        if let brace = text.firstIndex(of: "{") {
            let name = text[..<brace].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty, let arguments = lenientJSON(String(text[brace...])).flatMap(argumentsObject) else { return [] }
            return [ParsedToolCall(name: name, arguments: arguments)]
        }
        return []
    }

    /// Gemma 4's `call:name{key:<|"|>value<|"|>,…}`, one or more.
    private static func gemmaCalls(_ text: String) -> [ParsedToolCall] {
        var body = text
        if body.hasPrefix("call:") { body.removeFirst("call:".count) }
        if body.hasPrefix("{") || body.hasPrefix("[") { return jsonCalls(body) }
        guard let brace = body.firstIndex(of: "{") else { return [] }
        let name = body[..<brace].trimmingCharacters(in: .whitespacesAndNewlines)
        var args = String(body[brace...]).replacingOccurrences(of: "<|\"|>", with: "\"")
        args = quotingBareKeys(args)
        guard !name.isEmpty, let arguments = lenientJSON(args).flatMap(argumentsObject) else { return [] }
        return [ParsedToolCall(name: name, arguments: arguments)]
    }

    /// The Llama 3 habit of answering with nothing but a call object, and the
    /// small-model habit of fencing one. Only for an offered tool's name.
    private static func bareCalls(_ text: String, known: [String: ToolDefinition]) -> [ParsedToolCall]? {
        guard !known.isEmpty else { return nil }
        let trimmed = unfenced(text.trimmingCharacters(in: .whitespacesAndNewlines))
        guard trimmed.hasPrefix("{") || trimmed.hasPrefix("[") else { return nil }
        let calls = jsonCalls(trimmed)
        guard !calls.isEmpty, calls.allSatisfy({ known[$0.name] != nil }) else { return nil }
        return calls
    }

    // MARK: - XML shape

    /// `<function=name><parameter=key>value</parameter>…</function>`, as many
    /// as there are. Values are text; one whose schema type is not a string is
    /// read as JSON, so `<parameter=limit>5</parameter>` arrives as `5`.
    private static func xmlFunctions(_ text: String, known: [String: ToolDefinition]) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        var rest = Substring(text)
        while let open = rest.range(of: "<function=") {
            let afterOpen = rest[open.upperBound...]
            guard let close = afterOpen.firstIndex(of: ">") else { break }
            let name = afterOpen[..<close].trimmingCharacters(in: .whitespacesAndNewlines)
            let bodyStart = afterOpen.index(after: close)
            let end = afterOpen[bodyStart...].range(of: "</function>")
            let body = afterOpen[bodyStart..<(end?.lowerBound ?? afterOpen.endIndex)]
            rest = end.map { afterOpen[$0.upperBound...] } ?? afterOpen[afterOpen.endIndex...]
            guard !name.isEmpty else { continue }

            let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedBody.hasPrefix("{"), let arguments = lenientJSON(trimmedBody).flatMap(argumentsObject) {
                // Llama 3.1's `<function=name>{…}</function>`.
                calls.append(ParsedToolCall(name: name, arguments: arguments))
                continue
            }
            let properties = known[name]?.parameters["properties"]?.objectValue ?? [:]
            var fields: [String: JSONValue] = [:]
            var params = Substring(body)
            while let pOpen = params.range(of: "<parameter=") {
                let afterP = params[pOpen.upperBound...]
                guard let pClose = afterP.firstIndex(of: ">") else { break }
                let key = afterP[..<pClose].trimmingCharacters(in: .whitespacesAndNewlines)
                let valueStart = afterP.index(after: pClose)
                let pEnd = afterP[valueStart...].range(of: "</parameter>")
                var value = String(afterP[valueStart..<(pEnd?.lowerBound ?? afterP.endIndex)])
                params = pEnd.map { afterP[$0.upperBound...] } ?? afterP[afterP.endIndex...]
                // The template puts the value on its own line; those two
                // newlines are framing, not content.
                if value.hasPrefix("\n") { value.removeFirst() }
                if value.hasSuffix("\n") { value.removeLast() }
                fields[key] = typedValue(value, schema: properties[key])
            }
            calls.append(ParsedToolCall(name: name, arguments: .object(fields)))
        }
        return calls
    }

    private static func typedValue(_ text: String, schema: JSONValue?) -> JSONValue {
        let declared = schema?["type"]?.stringValue
        if declared == "string" { return .string(text) }
        if let parsed = lenientJSON(text.trimmingCharacters(in: .whitespacesAndNewlines)) {
            if declared == nil, case .string = parsed { return .string(text) }
            return parsed
        }
        return .string(text)
    }

    // MARK: - Lenient JSON

    /// Strict JSON first, then the mistakes small models make most: doubled
    /// braces copied from the template's own `{{` (Qwen 2.5 at 0.5B does this
    /// on its first call), trailing commas, and a stray close brace or two.
    static func lenientJSON(_ text: String) -> JSONValue? {
        if let value = JSONValue.parse(text) { return value }
        if text.hasPrefix("{{"), text.hasSuffix("}}"),
           let value = lenientJSON(String(text.dropFirst().dropLast())) {
            return value
        }
        let noTrailingCommas = text.replacingOccurrences(
            of: #",\s*([}\]])"#, with: "$1", options: .regularExpression
        )
        if let value = JSONValue.parse(noTrailingCommas) { return value }
        var trimmed = noTrailingCommas
        for _ in 0..<2 where trimmed.hasSuffix("}") || trimmed.hasSuffix("]") {
            trimmed.removeLast()
            trimmed = trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
            if let value = JSONValue.parse(trimmed) { return value }
        }
        return nil
    }

    /// The contents of a ```json fence, or the text unchanged.
    static func unfenced(_ text: String) -> String {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("```") else { return trimmed }
        trimmed.removeFirst(3)
        if let newline = trimmed.firstIndex(of: "\n") {
            let tag = trimmed[..<newline]
            if !tag.contains("{") { trimmed = String(trimmed[trimmed.index(after: newline)...]) }
        }
        if let close = trimmed.range(of: "```", options: .backwards) {
            trimmed = String(trimmed[..<close.lowerBound])
        }
        return trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func quotingBareKeys(_ text: String) -> String {
        text.replacingOccurrences(
            of: #"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:"#,
            with: "$1\"$2\":",
            options: .regularExpression
        )
    }
}

/// Reads `[name(key='value', n=5), other()]`: Python call syntax with Python
/// literals, which is how LFM2 writes tool calls.
struct PythonCallReader {
    private let characters: [Character]
    private var index = 0

    init(_ text: String) {
        characters = Array(text)
    }

    /// Every call, or nil if any part is not a call.
    mutating func calls() -> [ParsedToolCall]? {
        skipSpace()
        let bracketed = consume("[")
        var calls: [ParsedToolCall] = []
        while true {
            skipSpace()
            if index >= characters.count || (bracketed && peek == "]") { break }
            guard let name = identifier(), consume("(") else { return nil }
            var arguments: [String: JSONValue] = [:]
            skipSpace()
            while !consume(")") {
                guard let key = identifier(), consume("="), let value = value() else { return nil }
                arguments[key] = value
                skipSpace()
                _ = consume(",")
                skipSpace()
            }
            calls.append(ParsedToolCall(name: name, arguments: .object(arguments)))
            skipSpace()
            _ = consume(",")
        }
        return calls.isEmpty ? nil : calls
    }

    private var peek: Character? { index < characters.count ? characters[index] : nil }

    private mutating func skipSpace() {
        while let character = peek, character.isWhitespace { index += 1 }
    }

    private mutating func consume(_ character: Character) -> Bool {
        skipSpace()
        guard peek == character else { return false }
        index += 1
        return true
    }

    private mutating func identifier() -> String? {
        skipSpace()
        let start = index
        while let character = peek, character.isLetter || character.isNumber || "_.-".contains(character) {
            index += 1
        }
        return index > start ? String(characters[start..<index]) : nil
    }

    private mutating func value() -> JSONValue? {
        skipSpace()
        guard let character = peek else { return nil }
        switch character {
        case "'", "\"":
            return string(quote: character)
        case "[", "(":
            index += 1
            let close: Character = character == "[" ? "]" : ")"
            var items: [JSONValue] = []
            while !consume(close) {
                guard let item = value() else { return nil }
                items.append(item)
                _ = consume(",")
            }
            return .array(items)
        case "{":
            index += 1
            var fields: [String: JSONValue] = [:]
            while !consume("}") {
                guard let key = value()?.stringValue, consume(":"), let item = value() else { return nil }
                fields[key] = item
                _ = consume(",")
            }
            return .object(fields)
        default:
            let start = index
            while let next = peek, !",)]}".contains(next), !next.isWhitespace { index += 1 }
            let word = String(characters[start..<index])
            switch word {
            case "True", "true": return .bool(true)
            case "False", "false": return .bool(false)
            case "None", "null": return .null
            default:
                if let integer = Int(word) { return .int(integer) }
                if let double = Double(word) { return .double(double) }
                return word.isEmpty ? nil : .string(word)
            }
        }
    }

    private mutating func string(quote: Character) -> JSONValue? {
        index += 1
        var out = ""
        while let character = peek {
            index += 1
            if character == quote { return .string(out) }
            if character == "\\", let escaped = peek {
                index += 1
                switch escaped {
                case "n": out.append("\n")
                case "t": out.append("\t")
                case "r": out.append("\r")
                default: out.append(escaped)
                }
            } else {
                out.append(character)
            }
        }
        return nil
    }
}

/// Splits a streamed reply into text that can be shown now and tool-call
/// markup that has to wait for the end.
///
/// Text is passed through as it arrives, except for a tail that could be the
/// start of a marker — `<tool_` might become `<tool_call>` on the next token.
/// From the first marker on, everything is held: a call is only parseable once
/// it is complete, and a half-written call must never be shown as prose.
///
/// With tools on offer, a reply that *opens* with `{` or a code fence is held
/// whole too, because that is how a Llama-style bare call looks while it is
/// being written. If it turns out to be an answer, it is released at the end.
public struct ToolMarkupSplitter: Sendable {
    private let holdsBareCalls: Bool
    private var pending = ""
    private var held = ""
    private var holding = false
    private var shownAnything = false

    public init(holdsBareCalls: Bool) {
        self.holdsBareCalls = holdsBareCalls
    }

    /// Text that may be shown now.
    public mutating func feed(_ chunk: String) -> String {
        if holding {
            held += chunk
            return ""
        }
        pending += chunk
        if let (start, _) = ToolCallParser.earliestMarker(in: pending) {
            let visible = String(pending[..<start])
            held = String(pending[start...])
            pending = ""
            holding = true
            return show(visible)
        }
        if holdsBareCalls, !shownAnything {
            let lead = pending.drop { $0.isWhitespace }
            if lead.isEmpty { return "" }
            if lead.hasPrefix("{") || lead.hasPrefix("[") || lead.hasPrefix("`") {
                held = pending
                pending = ""
                holding = true
                return ""
            }
        }
        let keep = heldBackCount(pending)
        let visible = String(pending.dropLast(keep))
        pending = String(pending.suffix(keep))
        return show(visible)
    }

    /// What is left: text to show, and markup to parse.
    public mutating func finish() -> (text: String, markup: String) {
        defer { pending = ""; held = "" }
        return (pending, held)
    }

    private mutating func show(_ text: String) -> String {
        if !text.isEmpty { shownAnything = true }
        return text
    }

    private func heldBackCount(_ text: String) -> Int {
        let longest = ToolCallParser.markers.map(\.count).max() ?? 0
        for length in stride(from: min(longest - 1, text.count), to: 0, by: -1) {
            let tail = text.suffix(length)
            if ToolCallParser.markers.contains(where: { $0.hasPrefix(tail) }) { return length }
        }
        return 0
    }
}
