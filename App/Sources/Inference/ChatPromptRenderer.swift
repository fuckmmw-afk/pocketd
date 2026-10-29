import Foundation
import Jinja
import PocketdKit

/// Renders a client-tool conversation (`ClientTools`) into the exact prompt
/// text the model reads, with the model's own chat template.
///
/// The other routes reach the template through LocalLLMClient, which renders
/// it with this same Jinja engine but only knows the phone's own tools — their
/// schemas are fixed at compile time — and only plain text turns. A client's
/// tools are arbitrary JSON Schema that arrives per request, and its history
/// has tool calls and tool results in it, so this renders the template itself,
/// the way llama.cpp's own server does: OpenAI-style `messages` with
/// `tool_calls` and `tool` turns, and `tools` in OpenAI's function format.
///
/// A template that knows nothing about tools gets the flattened conversation
/// instead (see `ChatTemplateInput.flattened`), and so does one whose native
/// rendering throws — a template that `raise_exception`s on a role it does not
/// expect fails as a Swift error here, never as a crash.
struct ChatPromptRenderer: Sendable {
    let template: String
    /// `bos_token`, `eos_token` and the rest, as templates name them.
    let specialTokens: [String: String]

    /// Whether the template renders a conversation with tools in it: the tool
    /// list *and* the calls already made.
    ///
    /// Stricter than `ContextGuard.templateIsToolNative`, which only asks
    /// whether tools are mentioned. Checked against real templates: SmolLM3
    /// takes its tools in a variable of its own and Phi-4-mini in a field of
    /// the system turn, and neither renders an assistant's `tool_calls` at
    /// all — handed native input, the model would never see the calls it made.
    var rendersToolsNatively: Bool {
        template.contains("tools") && template.contains("tool_calls")
    }

    /// Whether a call's arguments go in as an object or as a JSON string.
    ///
    /// An object unless the template prints the field as it is with no sign of
    /// handling one: most templates `tojson` it, Qwen3-Coder iterates its
    /// `items`, and LFM2 raises if handed a string. An object printed as it is
    /// comes out as a dictionary literal, which is not JSON.
    var takesObjectArguments: Bool {
        let handlesObjects = [
            "arguments | tojson", "arguments|tojson", "is string", "is mapping",
            "arguments|items", "arguments | items", "arguments.items()",
        ].contains { template.contains($0) }
        let printsAsIs = ["arguments }}", "arguments}}", "arguments'] }}", "arguments\"] }}"]
            .contains { template.contains($0) }
        return handlesObjects || !printsAsIs
    }

    /// Whether the template reads `enable_thinking`, Qwen 3's switch.
    var understandsThinkingSwitch: Bool { template.contains("enable_thinking") }

    func render(_ messages: [ChatMessage], tools: ClientTools, thinking: Bool) throws -> String {
        let usesTools = !tools.offered.isEmpty || messages.contains { !$0.toolCalls.isEmpty || $0.role == .tool }
        if rendersToolsNatively || !usesTools {
            do {
                return try renderNative(messages, tools: tools, thinking: thinking)
            } catch {
                guard usesTools else { throw error }
                // Fall through: the flattened shape uses nothing but the three
                // roles every template accepts.
            }
        }
        let flat = ChatTemplateInput.flattened(messages, tools: tools)
        return try renderTemplate(
            messages: ChatTemplateInput.nativeMessages(flat),
            tools: [],
            thinking: thinking
        )
    }

    private func renderNative(_ messages: [ChatMessage], tools: ClientTools, thinking: Bool) throws -> String {
        try renderTemplate(
            messages: ChatTemplateInput.nativeMessages(
                messages,
                choice: tools.choice,
                argumentsAsObjects: takesObjectArguments
            ),
            tools: ChatTemplateInput.nativeTools(tools.offered),
            thinking: thinking
        )
    }

    private func renderTemplate(messages: [JSONValue], tools: [JSONValue], thinking: Bool) throws -> String {
        var context: [String: Value] = [
            "messages": try Value(any: messages.map(Self.jinja)),
            "add_generation_prompt": .boolean(true),
        ]
        if !tools.isEmpty {
            context["tools"] = try Value(any: tools.map(Self.jinja))
        }
        for (name, text) in specialTokens {
            context[name] = .string(text)
        }
        if !thinking, understandsThinkingSwitch {
            context["enable_thinking"] = .boolean(false)
        }
        // Exactly as LocalLLMClient renders, so a conversation with no tools
        // reads the same here as it does through the other routes.
        let environment = Environment()
        environment.lstripBlocks = true
        environment.trimBlocks = true
        return try Template(template).render(context, environment: environment)
    }

    /// JSON as the values `Value(any:)` reads — the same path LocalLLMClient
    /// feeds its messages through. Scalars go in as `Value` already: a `Bool`
    /// left as `Any` is one bridging cast away from becoming `1`.
    static func jinja(_ json: JSONValue) -> any Sendable {
        switch json {
        case .null: return Value.null
        case let .bool(value): return Value.boolean(value)
        case let .int(value): return Value.int(value)
        case let .double(value): return Value.double(value)
        case let .string(value): return Value.string(value)
        case let .array(items): return items.map(jinja)
        case let .object(fields): return fields.mapValues(jinja)
        }
    }

    /// What LocalLLMClient falls back to for a model with no template: the
    /// ChatML shape, which is what most small models were tuned on.
    static let fallbackTemplate = """
    {% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
    """

    /// Whether a rendered prompt ends inside an open reasoning block — a
    /// template that prefills `<think>` — so the reply's first words are
    /// working, not answer.
    static func opensReasoning(_ prompt: String) -> Bool {
        let tail = prompt.suffix(96)
        guard let open = tail.range(of: "<think>", options: .backwards) else { return false }
        return tail[open.upperBound...].range(of: "</think>") == nil
    }
}
