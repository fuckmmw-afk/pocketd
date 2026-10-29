import Foundation

/// A single turn in a conversation, in the shape both the OpenAI and Ollama
/// wire formats agree on.
public struct ChatMessage: Sendable, Codable, Equatable {
    public enum Role: String, Sendable, Codable, CaseIterable {
        case system, user, assistant, tool
    }

    public var role: Role
    public var content: String
    /// Decoded image bytes attached to this turn. Kept as raw data rather than
    /// a platform image type so the wire layer, which has no UIKit, can carry
    /// them to the engine, which does.
    public var images: [Data]
    /// What the tools this turn ran actually found, drawn rather than narrated.
    ///
    /// Here rather than in a parallel array beside the transcript because a
    /// card belongs to the turn that produced it: reordering, deleting or
    /// reopening a conversation must not be able to leave a card attached to
    /// the wrong answer.
    ///
    /// Never reaches the model. `ContextGuard.estimateTokens` reads `content`
    /// alone, and it is right to — the payload these were built from was
    /// already charged to the prompt once, as the tool result, and charging it
    /// again would shrink the usable window for a rendering the model cannot
    /// see. Nor does it reach the wire: the OpenAI and Ollama routes serialise
    /// `OpenAI.Message` and `Ollama.Message`, which are separate types with
    /// their own fields, so nothing here can appear in an HTTP response.
    public var cards: [AnswerCard]

    /// When this turn was said, for the conversation's day separators.
    ///
    /// Optional because nothing before this field existed carries one, and a
    /// guessed date would put a separator in the wrong place in every old
    /// transcript. Stamped by the phone's own chat when a turn is added; the
    /// server's routes build their messages from the wire and leave it `nil`.
    ///
    /// Never reaches the model or the wire, for the same reasons `cards` does
    /// not: the prompt is built from `content`, and the OpenAI and Ollama routes
    /// serialise their own message types.
    public var sentAt: Date?

    /// Calls an assistant turn made to tools the client runs (`ClientTools`).
    ///
    /// Empty for every turn the phone writes: its own tools run here and their
    /// results are folded into the transcript as text. Only a client that
    /// executes tools itself — the Anthropic route — carries calls back in, so
    /// the chat template can render them the way the model wrote them.
    public var toolCalls: [ToolCall]

    /// On a `.tool` turn, which call this is the result of.
    public var toolCallID: String?

    public init(
        role: Role,
        content: String,
        images: [Data] = [],
        cards: [AnswerCard] = [],
        sentAt: Date? = nil,
        toolCalls: [ToolCall] = [],
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.images = images
        self.cards = cards
        self.sentAt = sentAt
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }

    public static func system(_ content: String) -> ChatMessage { .init(role: .system, content: content) }
    public static func user(_ content: String) -> ChatMessage { .init(role: .user, content: content) }
    public static func assistant(_ content: String) -> ChatMessage { .init(role: .assistant, content: content) }
}

/// Decodes an `AnswerCard`, or nothing, without taking the message down with it.
///
/// Applied per element so one bad card in a message does not discard the
/// message's other cards alongside it.
private struct LossyCard: Decodable {
    let card: AnswerCard?

    init(from decoder: any Decoder) throws {
        card = try? AnswerCard(from: decoder)
    }
}

extension ChatMessage {
    private enum CodingKeys: String, CodingKey {
        case role, content, images, cards, sentAt, toolCalls, toolCallID
    }

    /// Written by hand only because a synthesised decoder throws on a key that
    /// is not there, and every conversation already on disk was written before
    /// `cards` existed. `ConversationStore.all()` skips a file it cannot
    /// decode, so the synthesised version would have made every transcript
    /// anyone has ever held disappear from the history list at the moment this
    /// field shipped. `images` is read the same way for the same reason.
    ///
    /// `cards` is written only when there are some. It is nearly always empty,
    /// and the store rewrites the whole conversation on every save. `sentAt`
    /// likewise only when there is one.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(Role.self, forKey: .role)
        content = try container.decode(String.self, forKey: .content)
        images = try container.decodeIfPresent([Data].self, forKey: .images) ?? []
        // Lossy on purpose, and the asymmetry with `AnswerCard`'s own strictness
        // is the point. Inside a card, a section of a known kind whose body will
        // not decode throws, so a bug surfaces in a test instead of hiding. At
        // this boundary the same throw costs the user their entire conversation:
        // `ConversationStore.all()` skips a file it cannot decode, so one
        // unreadable card took the whole transcript — messages, questions and
        // answers — out of the history list. Measured, not theorised: a probe
        // with a single malformed section recovered zero conversations.
        //
        // The messages are the part that cannot be recreated. A card is a
        // rendering of a tool result and is worth exactly nothing by
        // comparison, so it is the card that gets dropped.
        cards = (try? container.decodeIfPresent([LossyCard].self, forKey: .cards))?
            .compactMap(\.card) ?? []
        // Lossy for the same reason as `cards`: a date this build cannot read
        // costs a separator, never the conversation.
        sentAt = (try? container.decodeIfPresent(Date.self, forKey: .sentAt)) ?? nil
        // Lossy for the same reason again; no transcript on disk has either.
        toolCalls = (try? container.decodeIfPresent([ToolCall].self, forKey: .toolCalls)) ?? []
        toolCallID = (try? container.decodeIfPresent(String.self, forKey: .toolCallID)) ?? nil
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        try container.encode(images, forKey: .images)
        if !cards.isEmpty {
            try container.encode(cards, forKey: .cards)
        }
        try container.encodeIfPresent(sentAt, forKey: .sentAt)
        if !toolCalls.isEmpty {
            try container.encode(toolCalls, forKey: .toolCalls)
        }
        try container.encodeIfPresent(toolCallID, forKey: .toolCallID)
    }
}

/// Sampling knobs. Every field is optional so that a caller can send only what
/// it cares about and the engine can fall back to the model's own defaults —
/// clamping an unset value to some invented default is how you get a model that
/// silently behaves differently through the API than it does in the chat tab.
public struct GenerationOptions: Sendable, Equatable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var repeatPenalty: Double?
    public var maxTokens: Int?
    public var stopSequences: [String]
    public var seed: UInt64?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        repeatPenalty: Double? = nil,
        maxTokens: Int? = nil,
        stopSequences: [String] = [],
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repeatPenalty = repeatPenalty
        self.maxTokens = maxTokens
        self.stopSequences = stopSequences
        self.seed = seed
    }

    public static let `default` = GenerationOptions()
}
