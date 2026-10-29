import Foundation

/// A tool the *client* owns: described to the model, called by the model, and
/// executed on the other device — never on this phone.
///
/// Distinct from the phone's own tools (`PersonalDataTools` and friends),
/// which run here and are refused to network callers. A client tool is only a
/// schema as far as this server is concerned; what the model writes when it
/// wants one is handed back to the client to run.
public struct ToolDefinition: Sendable, Equatable {
    public var name: String
    public var description: String
    /// JSON Schema for the arguments, as the client sent it.
    public var parameters: JSONValue

    public init(name: String, description: String = "", parameters: JSONValue = ["type": "object", "properties": [:]]) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// One call the model made, or made earlier in the conversation.
public struct ToolCall: Sendable, Equatable, Codable {
    public var id: String
    public var name: String
    /// A JSON object.
    public var arguments: JSONValue

    public init(id: String, name: String, arguments: JSONValue) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public enum ToolChoice: Sendable, Equatable {
    /// The model decides.
    case auto
    /// The model must call at least one tool.
    case any
    /// The model must call this tool.
    case tool(String)
    /// The model must not call a tool.
    case none
}

/// Set on a `GenerationRequest` whose caller runs its own tools.
///
/// Its presence changes what an engine does, not only what it is told:
/// the conversation is rendered with the model's own chat template, tool calls
/// and tool results included, nothing on the phone is executed, any reasoning
/// the model writes is removed, and whatever tool-call markup the model emits
/// is left in the text for the caller to parse. Present with no tools is still
/// meaningful — a conversation can carry earlier calls with no tools offered
/// now.
public struct ClientTools: Sendable, Equatable {
    public var tools: [ToolDefinition]
    public var choice: ToolChoice

    public init(tools: [ToolDefinition] = [], choice: ToolChoice = .auto) {
        self.tools = tools
        self.choice = choice
    }

    /// The tools the model is actually shown. `none` shows none: describing
    /// tools to a model and then asking it not to use them only invites a call.
    public var offered: [ToolDefinition] {
        choice == .none ? [] : tools
    }
}

/// What a chat template is handed for a client-tool conversation.
///
/// Built here, as JSON, rather than in the engine, so the shape every template
/// sees is decided once and tested without llama.cpp. Two shapes:
///
/// - **Native**, for a template that renders tools itself — OpenAI-style
///   messages with `tool_calls` on assistant turns and `tool` turns carrying
///   `tool_call_id`, and the tools in OpenAI's function format. This is what
///   llama.cpp's own server hands a template, and what the template was
///   written for.
/// - **Flattened**, for a template that knows nothing of tools. The tools are
///   described in the system prompt in the Hermes `<tool_call>` convention,
///   calls become assistant text in that convention, results become user
///   turns, and consecutive turns of one role are merged so templates that
///   insist on alternating roles do not refuse the conversation.
public enum ChatTemplateInput {
    /// Messages for a template that renders tools itself.
    ///
    /// - Parameter argumentsAsObjects: whether a call's arguments go in as an
    ///   object or as a JSON string. Most templates `tojson` an object; some
    ///   print the field verbatim, and an object printed verbatim is not JSON.
    public static func nativeMessages(
        _ messages: [ChatMessage],
        choice: ToolChoice = .auto,
        argumentsAsObjects: Bool = true
    ) -> [JSONValue] {
        var namesByID: [String: String] = [:]
        for message in messages {
            for call in message.toolCalls { namesByID[call.id] = call.name }
        }
        return withToolChoiceInstruction(messages, choice: choice).map { message in
            var fields: [String: JSONValue] = [
                "role": .string(message.role.rawValue),
                "content": .string(message.content),
            ]
            if !message.toolCalls.isEmpty {
                fields["tool_calls"] = .array(message.toolCalls.map { call in
                    [
                        "id": .string(call.id),
                        "type": "function",
                        "function": [
                            "name": .string(call.name),
                            "arguments": argumentsAsObjects ? call.arguments : .string(call.arguments.serialized),
                        ],
                    ]
                })
            }
            if message.role == .tool, let id = message.toolCallID {
                fields["tool_call_id"] = .string(id)
                if let name = namesByID[id] { fields["name"] = .string(name) }
            }
            return .object(fields)
        }
    }

    /// Tools in OpenAI's function format, which is what tool-aware templates
    /// iterate over.
    public static func nativeTools(_ tools: [ToolDefinition]) -> [JSONValue] {
        tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": tool.parameters,
                ],
            ]
        }
    }

    /// The conversation for a template with no notion of tools.
    public static func flattened(_ messages: [ChatMessage], tools: ClientTools) -> [ChatMessage] {
        var out: [ChatMessage] = []
        func append(_ role: ChatMessage.Role, _ text: String, images: [Data] = []) {
            if let last = out.indices.last, out[last].role == role {
                if !text.isEmpty {
                    out[last].content += out[last].content.isEmpty ? text : "\n\n" + text
                }
                out[last].images += images
            } else {
                out.append(ChatMessage(role: role, content: text, images: images))
            }
        }

        var systemText = messages.filter { $0.role == .system }.map(\.content).joined(separator: "\n\n")
        let offered = tools.offered
        if !offered.isEmpty {
            let instructions = toolInstructions(offered)
            systemText = systemText.isEmpty ? instructions : systemText + "\n\n" + instructions
        }
        if let rule = toolChoiceInstruction(tools.choice) {
            systemText = systemText.isEmpty ? rule : systemText + "\n\n" + rule
        }
        if !systemText.isEmpty { out.append(.system(systemText)) }

        for message in messages where message.role != .system {
            switch message.role {
            case .system:
                break
            case .user:
                append(.user, message.content, images: message.images)
            case .assistant:
                var parts: [String] = []
                if !message.content.isEmpty { parts.append(message.content) }
                for call in message.toolCalls {
                    parts.append(hermesCall(name: call.name, arguments: call.arguments))
                }
                append(.assistant, parts.joined(separator: "\n"))
            case .tool:
                append(.user, "<tool_response>\n\(message.content)\n</tool_response>", images: message.images)
            }
        }
        return out
    }

    /// The Hermes convention, which is also what most small open models were
    /// tuned on and what `ToolCallParser` reads first.
    public static func toolInstructions(_ tools: [ToolDefinition]) -> String {
        let signatures = nativeTools(tools).map(\.serialized).joined(separator: "\n")
        return """
        # Tools

        You may call one or more functions to assist with the user query.

        You are provided with function signatures within <tools></tools> XML tags:
        <tools>
        \(signatures)
        </tools>

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>
        """
    }

    /// Name first, the order the models were tuned on, which sorted keys
    /// would reverse.
    public static func hermesCall(name: String, arguments: JSONValue) -> String {
        "<tool_call>\n{\"name\": \(JSONValue.string(name).serialized), \"arguments\": \(arguments.serialized)}\n</tool_call>"
    }

    /// A sentence for the choices a template cannot express. Templates only
    /// ever offer tools; "you must call one" has to be said in words.
    public static func toolChoiceInstruction(_ choice: ToolChoice) -> String? {
        switch choice {
        case .auto:
            return nil
        case .any:
            return "You must call at least one of the available tools in your reply."
        case let .tool(name):
            return "You must call the tool \"\(name)\" in your reply."
        case .none:
            return "Do not call any tools. Reply with text only."
        }
    }

    /// The tool-choice sentence appended to the system turn, creating one if
    /// there is none.
    static func withToolChoiceInstruction(_ messages: [ChatMessage], choice: ToolChoice) -> [ChatMessage] {
        guard let rule = toolChoiceInstruction(choice) else { return messages }
        var updated = messages
        if let index = updated.firstIndex(where: { $0.role == .system }) {
            updated[index].content += updated[index].content.isEmpty ? rule : "\n\n" + rule
        } else {
            updated.insert(.system(rule), at: 0)
        }
        return updated
    }
}
