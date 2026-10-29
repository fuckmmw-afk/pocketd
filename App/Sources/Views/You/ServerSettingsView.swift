import PocketdKit
import SwiftUI
import UIKit

/// How this iPhone answers other devices, and nothing else.
///
/// Every control here is about the server: who can reach it, the key they
/// need, how long and how many requests it takes on, what it does when the
/// phone is hot or flat, and what a request inherits when it does not say.
/// The phone's own conversations follow none of it except the context limit
/// (one engine, one window) — thinking in particular is off for the phone
/// whatever this page says, which is why its switch says "other devices".
///
/// What used to sit beside these and was about the phone — keeping the brain
/// in memory, letting it go after a while — is gone from the app's screens:
/// residency is automatic now.
///
/// Changes are staged and applied together, because half of them restart the
/// listener and applying each keystroke would tear the socket down under a
/// paired laptop mid-request.
struct ServerSettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(ToastCenter.self) private var toasts

    @State private var draft = ServerConfiguration()

    /// What ships, which is not `SamplingParameters.default`: the server has
    /// always applied a 1.1 repeat penalty (see `ServerConfiguration.sampling`),
    /// and "put the defaults back" restoring llama.cpp's 1.0 instead would
    /// change behaviour under a button promising to undo changes.
    private static let shippedSampling = ServerConfiguration().sampling
    @State private var portText = ""
    @State private var showKeyWarning = false
    @State private var showsKey = false
    @FocusState private var portFocused: Bool

    /// nil when the field is a usable port.
    private var portProblem: String? {
        guard !portText.isEmpty else { return "Enter a port." }
        guard let value = Int(portText) else { return "Ports are numbers." }
        guard (1...65535).contains(value) else { return "Port must be between 1 and 65535." }
        return nil
    }

    private var isUnchanged: Bool {
        draft == model.configuration && portText == String(model.configuration.port)
    }

    var body: some View {
        YouDetailScreen(
            title: "Server settings",
            subtitle: "How this iPhone answers other devices. Nothing changes until you tap Apply."
        ) {
            network
            access
            claudeCode
            SpeechModelsSection()
            answers
            device
            sampling
            apply
        }
        .onAppear {
            draft = model.configuration
            portText = String(model.configuration.port)
        }
        .confirmationDialog(
            "Make a new key?",
            isPresented: $showKeyWarning,
            titleVisibility: .visible
        ) {
            Button("Make a new key", role: .destructive) {
                draft.apiKey = ServerConfiguration.generateAPIKey()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Every paired device stops working until it pairs again with a new code.")
        }
    }

    // MARK: - Network

    private var network: some View {
        YouSection(
            "Network",
            footnote: "Most apps expect port 11434. Changing it disconnects paired devices."
        ) {
            MenuRow(
                title: "Reachable from",
                selection: $draft.binding,
                options: ServerConfiguration.Binding.allCases.map { ($0, $0.title) }
            )
            RowDivider(inset: 0)
            RowShell(title: "Port", detail: nil) {
                TextField("11434", text: $portText)
                    .font(Font.pd.monoValue)
                    .foregroundStyle(Color.pd.ink)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(minWidth: 72, maxWidth: 140)
                    .fixedSize(horizontal: true, vertical: false)
                    .focused($portFocused)
                    .accessibilityLabel("Port")
                    // The number pad has no Return key, so without somewhere to
                    // tap the keyboard covers the screen with no way out.
                    .toolbar {
                        ToolbarItemGroup(placement: .keyboard) {
                            Spacer()
                            Button("Done") { portFocused = false }
                        }
                    }
            }
            if let problem = portProblem {
                Text(problem)
                    .font(Font.pd.footnote)
                    .foregroundStyle(Color.pd.caution)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.bottom, PDSpacing.xs)
            }
        }
    }

    // MARK: - Access

    private var access: some View {
        YouSection(
            "Access",
            footnote: "Without a key, anything on your Wi-Fi can use Pocketd. Only turn it off on a network you trust."
        ) {
            ToggleRow(title: "Require a key", detail: nil, isOn: $draft.requiresAuth)
            if draft.requiresAuth {
                RowDivider(inset: 0)
                keyBlock
                RowDivider(inset: 0)
                ActionRow(symbol: "doc.on.doc", title: "Copy the key", detail: nil) {
                    // The live key, not the draft: a key that is not applied
                    // yet is one no request would be accepted with.
                    UIPasteboard.general.string = model.configuration.apiKey
                    Haptics.success()
                    toasts.show("Key copied", symbol: "doc.on.doc")
                }
                RowDivider()
                ActionRow(
                    symbol: "key",
                    title: "Make a new key",
                    detail: "Paired devices have to pair again",
                    isDestructive: true
                ) {
                    showKeyWarning = true
                }
            }
            RowDivider(inset: 0)
            ToggleRow(title: "Allow browser requests (CORS)", detail: nil, isOn: $draft.allowCORS)
        }
    }

    // MARK: - Claude Code

    /// The live address, port and key — what a request would be accepted
    /// with now, not what the draft would change them to. With the server off
    /// the address is this phone's Wi-Fi one, which is what it will listen on.
    private var claudeCodeCommands: String {
        let live = model.configuration
        let host: String
        let port: UInt16
        if case let .running(runningHost, runningPort) = model.serverState {
            host = runningHost
            port = runningPort
        } else {
            host = live.binding == .loopback ? "127.0.0.1" : (NetworkInterface.localIPv4Address() ?? "<iphone-ip>")
            port = live.port
        }
        return ClaudeCodeSetup.exports(host: host, port: port, apiKey: live.requiresAuth ? live.apiKey : nil)
    }

    private var claudeCode: some View {
        YouSection(
            "Claude Code",
            footnote: "Paste into a terminal on your computer, then run claude. These lines include the key."
        ) {
            MonoBlock(text: claudeCodeCommands)
                .padding(.vertical, PDSpacing.s)
            RowDivider(inset: 0)
            ActionRow(symbol: "doc.on.doc", title: "Copy commands", detail: nil) {
                UIPasteboard.general.string = claudeCodeCommands
                Haptics.success()
                toasts.show("Commands copied", symbol: "doc.on.doc")
            }
        }
    }

    /// The key, hidden until asked for: this page is the one place it can be
    /// read, and a screen someone opens beside a friend is the wrong place to
    /// print a working credential by default.
    ///
    /// Labelled when the draft differs from what is live, so nobody copies a
    /// key the server would refuse.
    private var keyBlock: some View {
        let applied = draft.apiKey == model.configuration.apiKey
        return VStack(alignment: .leading, spacing: PDSpacing.xs) {
            HStack(alignment: .firstTextBaseline) {
                Text(applied ? "Key" : "New key, not applied yet")
                    .font(Font.pd.rowTitle)
                    .foregroundStyle(applied ? Color.pd.ink : Color.pd.caution)
                Spacer(minLength: PDSpacing.xs)
                Button(showsKey ? "Hide" : "Show") { showsKey.toggle() }
                    .font(Font.pd.secondary.weight(.semibold))
                    .foregroundStyle(Color.pd.accent)
                    .pdHitTarget(alignment: .trailing)
                    .accessibilityHint("Shows the key other devices use to reach this iPhone")
            }
            if showsKey {
                MonoBlock(text: breakableIdentifier(draft.apiKey))
            } else {
                Text(redactedSecret(draft.apiKey))
                    .font(Font.pd.code)
                    .foregroundStyle(Color.pd.inkSecondary)
                    .accessibilityLabel("Hidden")
            }
        }
        .padding(.vertical, PDSpacing.s)
    }

    // MARK: - Answers

    private var answers: some View {
        YouSection(
            "Answers",
            footnote: "A longer limit uses more memory. One request at a time keeps every answer fastest."
        ) {
            ToggleRow(
                title: "Think before answering",
                detail: "Other devices only. Slower, sometimes better on hard questions.",
                isOn: $draft.reasoningEnabled
            )
            RowDivider(inset: 0)
            StepperRow(
                title: "Context limit",
                value: $draft.maxContextTokens,
                range: 512...32_768,
                step: 512,
                display: { "\(Format.count($0)) tokens" }
            )
            RowDivider(inset: 0)
            StepperRow(
                title: "Requests at once",
                value: $draft.maxConcurrentRequests,
                range: 1...4,
                step: 1,
                display: { "\($0)" }
            )
        }
    }

    // MARK: - Device

    private var device: some View {
        YouSection(
            "This iPhone",
            footnote: "Paused, other devices are asked to wait rather than refused. Charging lifts the battery floor."
        ) {
            ToggleRow(title: "Keep the screen awake while on", detail: nil, isOn: $draft.keepAwakeWhileServing)
            RowDivider(inset: 0)
            MenuRow(
                title: "Pause when hot",
                selection: $draft.thermalTolerance,
                options: ThermalTolerance.allCases.map { ($0, $0.title) }
            )
            RowDivider(inset: 0)
            RowShell(title: "Pause below battery", detail: nil) {
                HStack(spacing: PDSpacing.xs) {
                    Text(batteryFloorText)
                        .font(Font.pd.monoValue)
                        .foregroundStyle(Color.pd.ink)
                    Stepper("", value: $draft.pauseBelowBatteryLevel, in: 0...0.5, step: 0.05)
                        .labelsHidden()
                        .fixedSize()
                }
            }
            // One adjustable element saying what the screen says. Left to
            // itself the stepper read its raw value, "0.15", beside a row
            // that shows "15%".
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Pause below battery")
            .accessibilityValue(batteryFloorText)
            .accessibilityAdjustableAction { direction in
                let level = draft.pauseBelowBatteryLevel
                switch direction {
                case .increment: draft.pauseBelowBatteryLevel = min(0.5, level + 0.05)
                case .decrement: draft.pauseBelowBatteryLevel = max(0, level - 0.05)
                @unknown default: break
                }
            }
        }
    }

    private var batteryFloorText: String {
        draft.pauseBelowBatteryLevel <= 0 ? "Never" : "\(Int((draft.pauseBelowBatteryLevel * 100).rounded()))%"
    }

    // MARK: - Sampling

    private var sampling: some View {
        YouSection(
            "Sampling",
            footnote: "What a request gets when it doesn't set its own. A request that sends a value still wins."
        ) {
            SliderRow(title: "Temperature", value: $draft.sampling.temperature, range: 0...2, step: 0.05)
            RowDivider(inset: 0)
            SliderRow(title: "Top-p", value: $draft.sampling.topP, range: 0...1, step: 0.01)
            RowDivider(inset: 0)
            StepperRow(title: "Top-k", value: $draft.sampling.topK, range: 0...200, step: 1, display: { "\($0)" })
            RowDivider(inset: 0)
            SliderRow(title: "Repeat penalty", value: $draft.sampling.repeatPenalty, range: 1...2, step: 0.01)
            RowDivider(inset: 0)
            ToggleRow(
                title: "Fixed seed",
                detail: nil,
                isOn: Binding(
                    get: { draft.sampling.seed != nil },
                    set: { draft.sampling.seed = $0 ? 0 : nil }
                )
            )
            if draft.sampling.seed != nil {
                RowShell(title: "Seed", detail: nil) {
                    TextField("0", value: Binding(
                        get: { draft.sampling.seed ?? 0 },
                        set: { draft.sampling.seed = $0 }
                    ), format: .number)
                        .font(Font.pd.monoValue)
                        .foregroundStyle(Color.pd.ink)
                        .keyboardType(.numberPad)
                        .multilineTextAlignment(.trailing)
                        .frame(minWidth: 72, maxWidth: 160)
                        .fixedSize(horizontal: true, vertical: false)
                        .accessibilityLabel("Seed")
                }
            }
            RowDivider(inset: 0)
            ActionRow(symbol: "arrow.uturn.backward", title: "Put the defaults back", detail: nil) {
                draft.sampling = Self.shippedSampling
            }
            .disabled(draft.sampling == Self.shippedSampling)
        }
    }

    // MARK: - Apply

    private var apply: some View {
        PrimaryButton(
            title: "Apply",
            arrow: false,
            isEnabled: portProblem == nil && !isUnchanged
        ) {
            // Guarded by `portProblem`, so the silent fallback that used to
            // make Apply a permanent no-op cannot happen: an out-of-range
            // port left the button enabled forever with nothing changing
            // and no message.
            guard let port = UInt16(portText) else { return }
            draft.port = port
            Task {
                await model.applyConfiguration(draft)
                toasts.show("Applied", symbol: "checkmark.circle", tint: .pd.positive)
            }
        }
        .padding(.top, PDSpacing.xs)
    }
}

/// `YouRouteView` still says `SettingsView()`; the page's name is the one above.
typealias SettingsView = ServerSettingsView

// MARK: - Controls

/// Label, value, and a track underneath.
///
/// The number belongs next to the name it qualifies. Putting the slider and its
/// readout in a trailing slot wraps them under one another at this width and
/// leaves the label floating beside a two-line stack.
private struct SliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double

    var body: some View {
        VStack(alignment: .leading, spacing: PDSpacing.xxs) {
            HStack {
                Text(title)
                    .font(Font.pd.rowTitle)
                    .foregroundStyle(Color.pd.ink)
                Spacer(minLength: PDSpacing.xs)
                Text(value, format: .number.precision(.fractionLength(2)))
                    .font(Font.pd.monoValue)
                    .foregroundStyle(Color.pd.inkSecondary)
            }
            Slider(value: $value, in: range, step: step)
                .tint(Color.pd.accent)
                .accessibilityLabel(title)
        }
        .padding(.vertical, PDSpacing.s)
    }
}

/// A whole number with two buttons, in the row shape the rest of the card uses.
private struct StepperRow: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int
    let display: (Int) -> String

    var body: some View {
        RowShell(title: title, detail: nil) {
            HStack(spacing: PDSpacing.xs) {
                Text(display(value))
                    .font(Font.pd.monoValue)
                    .foregroundStyle(Color.pd.ink)
                Stepper("", value: $value, in: range, step: step)
                    .labelsHidden()
                    .fixedSize()
            }
        }
        // One adjustable element with the value as displayed ("4,096 tokens"),
        // not the stepper's bare number beside a separate label.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(display(value))
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: value = min(range.upperBound, value + step)
            case .decrement: value = max(range.lowerBound, value - step)
            @unknown default: break
            }
        }
    }
}

// MARK: - Previews

#Preview("Server settings light") {
    NavigationStack { ServerSettingsView() }
        .environment(AppModel())
        .environment(Navigator())
        .environment(ToastCenter())
}

#Preview("Server settings dark") {
    NavigationStack { ServerSettingsView() }
        .environment(AppModel())
        .environment(Navigator())
        .environment(ToastCenter())
        .preferredColorScheme(.dark)
}
