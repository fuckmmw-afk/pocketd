import Foundation

/// The Anthropic Messages API, as much of it as a local model can honour.
///
/// Decoding is deliberately forgiving. Claude Code and the Anthropic SDKs send
/// fields this server has no use for — `metadata`, `cache_control`,
/// `thinking`, `service_tier`, `container`, beta flags — and a request that
/// failed on any of them would fail on every request. Unknown fields are
/// ignored; unknown content blocks are dropped; only a body that is not the
/// right shape at all is refused.
enum Anthropic {
    struct MessagesRequest: Decodable, Sendable {
        var model: String
        var system: SystemPrompt?
        var messages: [Message]
        var tools: [Tool]?
        var tool_choice: ToolChoiceSpec?
        var max_tokens: Int?
        var temperature: Double?
        var top_p: Double?
        var top_k: Int?
        var stop_sequences: [String]?
        var stream: Bool?

        private enum CodingKeys: String, CodingKey {
            case model, system, messages, tools, tool_choice, max_tokens, temperature, top_p, top_k, stop_sequences, stream
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            model = (try? container.decodeIfPresent(String.self, forKey: .model)) ?? ""
            system = try container.decodeIfPresent(SystemPrompt.self, forKey: .system)
            messages = try container.decode([Message].self, forKey: .messages)
            tools = try container.decodeIfPresent([Tool].self, forKey: .tools)
            tool_choice = try? container.decodeIfPresent(ToolChoiceSpec.self, forKey: .tool_choice)
            max_tokens = try container.decodeIfPresent(Int.self, forKey: .max_tokens)
            temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
            top_p = try container.decodeIfPresent(Double.self, forKey: .top_p)
            top_k = try container.decodeIfPresent(Int.self, forKey: .top_k)
            stop_sequences = try container.decodeIfPresent([String].self, forKey: .stop_sequences)
            stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        }
    }

    /// `"system": "…"` or `"system": [{"type": "text", "text": "…"}, …]`.
    /// Claude Code sends the array, with `cache_control` on the blocks.
    struct SystemPrompt: Decodable, Sendable {
        var text: String

        init(from decoder: any Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let plain = try? container.decode(String.self) {
                text = plain
            } else {
                let blocks = try container.decode([ContentBlock].self)
                text = blocks.compactMap { block in
                    if case let .text(text) = block { return text }
                    return nil
                }.joined(separator: "\n\n")
            }
        }
    }

    struct Message: Decodable, Sendable {
        var role: String
        var content: [ContentBlock]

        private enum CodingKeys: String, CodingKey { case role, content }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            role = try container.decode(String.self, forKey: .role)
            if let plain = try? container.decode(String.self, forKey: .content) {
                content = [.text(plain)]
            } else {
                content = try container.decode([ContentBlock].self, forKey: .content)
            }
        }

        init(role: String, content: [ContentBlock]) {
            self.role = role
            self.content = content
        }
    }

    enum ImageSource: Sendable, Equatable {
        /// Decoded bytes.
        case data(Data)
        /// A URL, or anything else this server will not fetch.
        case unavailable
    }

    enum ContentBlock: Decodable, Sendable, Equatable {
        case text(String)
        case image(ImageSource)
        case toolUse(id: String, name: String, input: JSONValue)
        case toolResult(toolUseID: String, content: [ContentBlock], isError: Bool)
        /// Thinking, redacted thinking, server tools, documents with no text,
        /// and whatever is added next: nothing a local model can be given.
        case ignored

        private enum CodingKeys: String, CodingKey {
            case type, text, source, id, name, input, tool_use_id, content, is_error, data, media_type
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = (try? container.decode(String.self, forKey: .type)) ?? ""
            switch type {
            case "text":
                self = .text((try? container.decode(String.self, forKey: .text)) ?? "")
            case "image":
                self = .image(Self.imageSource(container))
            case "tool_use":
                self = .toolUse(
                    id: (try? container.decode(String.self, forKey: .id)) ?? "",
                    name: (try? container.decode(String.self, forKey: .name)) ?? "",
                    input: (try? container.decode(JSONValue.self, forKey: .input)) ?? .object([:])
                )
            case "tool_result":
                let blocks: [ContentBlock]
                if let plain = try? container.decode(String.self, forKey: .content) {
                    blocks = [.text(plain)]
                } else {
                    blocks = (try? container.decode([ContentBlock].self, forKey: .content)) ?? []
                }
                self = .toolResult(
                    toolUseID: (try? container.decode(String.self, forKey: .tool_use_id)) ?? "",
                    content: blocks,
                    isError: (try? container.decode(Bool.self, forKey: .is_error)) ?? false
                )
            case "document":
                // A plain-text document is text; a PDF is beyond a GGUF model.
                if let source = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .source),
                   (try? source.decode(String.self, forKey: .type)) == "text",
                   let data = try? source.decode(String.self, forKey: .data) {
                    self = .text(data)
                } else {
                    self = .ignored
                }
            default:
                self = .ignored
            }
        }

        /// Only base64 is accepted, for the same reason the OpenAI route only
        /// accepts data: URIs — fetching a URL would have the phone make
        /// outbound requests on a caller's behalf.
        private static func imageSource(_ container: KeyedDecodingContainer<CodingKeys>) -> ImageSource {
            guard let source = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .source),
                  (try? source.decode(String.self, forKey: .type)) == "base64",
                  let encoded = try? source.decode(String.self, forKey: .data),
                  let bytes = Data(base64Encoded: encoded, options: .ignoreUnknownCharacters)
            else { return .unavailable }
            return .data(bytes)
        }
    }

    struct Tool: Decodable, Sendable {
        var name: String
        var description: String?
        var input_schema: JSONValue?
        /// Absent or `"custom"` for a client tool. Anything else is one of
        /// Anthropic's server tools — web search, code execution — which only
        /// Anthropic can run.
        var type: String?
    }

    struct ToolChoiceSpec: Decodable, Sendable {
        var type: String
        var name: String?
    }
}

/// From the Anthropic wire shape to the one the engines read.
enum AnthropicConversion {
    /// What an image becomes for a model that cannot see.
    static let imageOmitted = "[image omitted]"

    /// - Parameter acceptsImages: whether the resident model has a vision
    ///   projector. Without one an image becomes a line of text saying so,
    ///   rather than a request the engine refuses.
    static func chatMessages(
        system: String?,
        messages: [Anthropic.Message],
        acceptsImages: Bool
    ) -> [ChatMessage] {
        var out: [ChatMessage] = []
        if let system, !system.isEmpty {
            out.append(.system(system))
        }

        for message in messages {
            switch message.role {
            case "assistant":
                var text: [String] = []
                var calls: [ToolCall] = []
                for block in message.content {
                    switch block {
                    case let .text(value):
                        text.append(value)
                    case let .toolUse(id, name, input):
                        calls.append(ToolCall(id: id, name: name, arguments: input.objectValue == nil ? .object([:]) : input))
                    case .image:
                        text.append(imageOmitted)
                    case .toolResult, .ignored:
                        break
                    }
                }
                out.append(ChatMessage(role: .assistant, content: text.joined(separator: "\n\n"), toolCalls: calls))

            default:
                // A user turn: tool results first, as their own turns, then
                // whatever the person wrote. Anthropic requires results to lead
                // the turn, and templates expect `tool` right after the call.
                var text: [String] = []
                var images: [Data] = []
                for block in message.content {
                    switch block {
                    case let .text(value):
                        text.append(value)
                    case let .image(source):
                        if acceptsImages, case let .data(bytes) = source {
                            images.append(bytes)
                        } else {
                            text.append(imageOmitted)
                        }
                    case let .toolResult(id, content, isError):
                        var resultImages: [Data] = []
                        let body = flatten(content, acceptsImages: acceptsImages, images: &resultImages)
                        out.append(ChatMessage(
                            role: .tool,
                            content: isError ? "Error: " + body : body,
                            images: resultImages,
                            toolCallID: id
                        ))
                    case .toolUse, .ignored:
                        break
                    }
                }
                if !text.isEmpty || !images.isEmpty {
                    out.append(ChatMessage(role: .user, content: text.joined(separator: "\n\n"), images: images))
                }
            }
        }
        return out
    }

    private static func flatten(_ blocks: [Anthropic.ContentBlock], acceptsImages: Bool, images: inout [Data]) -> String {
        blocks.compactMap { block -> String? in
            switch block {
            case let .text(value):
                return value
            case let .image(source):
                if acceptsImages, case let .data(bytes) = source {
                    images.append(bytes)
                    return nil
                }
                return imageOmitted
            default:
                return nil
            }
        }.joined(separator: "\n")
    }

    static func clientTools(_ tools: [Anthropic.Tool]?, choice: Anthropic.ToolChoiceSpec?) -> ClientTools {
        let definitions = (tools ?? []).compactMap { tool -> ToolDefinition? in
            if let type = tool.type, type != "custom", tool.input_schema == nil { return nil }
            return ToolDefinition(
                name: tool.name,
                description: tool.description ?? "",
                parameters: tool.input_schema ?? ["type": "object", "properties": [:]]
            )
        }
        let resolved: ToolChoice
        switch choice?.type {
        case "any": resolved = .any
        case "none": resolved = .none
        case "tool": resolved = choice?.name.map(ToolChoice.tool) ?? .any
        default: resolved = .auto
        }
        return ClientTools(tools: definitions, choice: definitions.isEmpty && resolved != .none ? .auto : resolved)
    }

    /// Stop sequences are left out on purpose: the route watches for them
    /// itself, because only it can say which one was hit.
    static func options(_ request: Anthropic.MessagesRequest) -> GenerationOptions {
        GenerationOptions(
            temperature: request.temperature,
            topP: request.top_p,
            topK: request.top_k,
            maxTokens: request.max_tokens
        )
    }

    /// `msg_` and `toolu_` followed by random base-62, the shape Anthropic's
    /// ids have and some clients check for.
    static func identifier(prefix: String) -> String {
        let alphabet = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        return prefix + String((0..<24).map { _ in alphabet.randomElement()! })
    }
}
