import Foundation

/// The shell lines that point Claude Code at this phone.
///
/// `ANTHROPIC_AUTH_TOKEN` rather than `ANTHROPIC_API_KEY`: Claude Code sends
/// the first as `Authorization: Bearer` without asking to confirm a key, and
/// the Anthropic route accepts either header. The timeout is ten minutes
/// because a phone reading a long prompt is slow, and the traffic switch keeps
/// Claude Code from sending this server requests meant for Anthropic's.
public enum ClaudeCodeSetup {
    /// - Parameter apiKey: nil when the server asks for no key. Claude Code
    ///   still wants a token to send, and any placeholder does.
    public static func exports(host: String, port: UInt16, apiKey: String?) -> String {
        """
        export ANTHROPIC_BASE_URL=http://\(host):\(port)
        export ANTHROPIC_AUTH_TOKEN=\(apiKey ?? "not-needed")
        export API_TIMEOUT_MS=600000
        export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
        """
    }
}
