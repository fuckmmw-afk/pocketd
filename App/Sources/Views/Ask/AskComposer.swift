import PhotosUI
import PocketdKit
import SwiftUI

/// The bottom of the conversation: what is attached, the one thing above the
/// bar, and the bar.
///
/// Its own `View` for one specific reason, and it is the difference between a
/// chat that types smoothly and one that locks up. SwiftUI registers an
/// observable read against whichever body performed it. While the composer was a
/// computed property of the chat screen, reading `model.draft` registered
/// against the *screen's* body — so every keystroke invalidated the whole
/// transcript and re-applied the `ForEach` over every message. A sampled hang
/// showed 1371 of 1371 main-thread frames inside one `CATransaction.commit`,
/// recursing through `LazyStack.sizeThatFits`, with no inference on the stack at
/// all: the model was not slow, the layout was rebuilding itself per letter.
///
/// Keeping the draft in here means a keystroke invalidates a bar, not a
/// transcript. Nothing above this view may read `model.draft`.
///
/// It never asks whether a brain can answer. `model.send()` holds a question
/// until one can and the notice above says so; a composer that refused to send
/// left somebody staring at their own words with no reply and no reason.
struct AskComposer<Accessory: View>: View {
    /// The notice, the Stop pill or the follow-up chip — whichever the
    /// conversation needs above the bar. Drawn here only so the waveform can
    /// take its place while the microphone is open; it reads its own state.
    @ViewBuilder var accessory: Accessory

    @Environment(AppModel.self) private var model
    @Environment(Navigator.self) private var navigator
    @Environment(ToastCenter.self) private var toasts: ToastCenter?

    @FocusState private var isFocused: Bool
    @State private var voice = VoiceInput()
    @State private var picked: [PhotosPickerItem] = []
    @State private var isPickingPhoto = false
    @State private var levels: [Float] = []
    /// A question handed over while a reply was still being written. It waits
    /// in the box and goes when that reply ends; sending over it would cut the
    /// reply off, and dropping it would lose the question.
    @State private var queued: String?
    /// One line above the bar just before iOS asks for Speech Recognition for
    /// the first time. The system's prompt says speech "will be sent to
    /// Apple"; this says first what Pocketd actually does with it.
    @State private var primer: String?

    var body: some View {
        @Bindable var model = model

        VStack(alignment: .leading, spacing: PDSpacing.s) {
            if !model.attachments.isEmpty { attachments }
            if voice.isListening {
                VoiceWaveform(levels: levels)
            } else if let primer {
                Primer(primer)
                    .padding(.horizontal, PDSpacing.xxs)
                    .transition(.opacity)
            } else {
                accessory
            }

            AskBar(
                text: $model.draft,
                placeholder: "Ask anything…",
                isListening: voice.isListening,
                canSend: canSend,
                showsMic: voice.isSupported && !model.isGenerating,
                onMic: startListening,
                onMicEnd: finishListening,
                onSend: send,
                onAttach: model.loadedModelSeesImages ? { isPickingPhoto = true } : nil,
                focus: $isFocused
            )
        }
        .photosPicker(
            isPresented: $isPickingPhoto,
            selection: $picked,
            maxSelectionCount: 4,
            matching: .images,
            photoLibrary: .shared()
        )
        .task(id: picked) { await loadPicked() }
        // Live, so the words appear as they are said rather than all at once on
        // release — which is the difference between dictation you can correct
        // and dictation you have to trust.
        .onChange(of: voice.transcript) { _, text in
            guard voice.isListening else { return }
            model.draft = text
        }
        .onChange(of: voice.level) { _, level in
            guard voice.isListening else { return }
            levels = Array((levels + [level]).suffix(VoiceWaveform.barCount))
        }
        // Already a sentence for a person, with the reason in it — see
        // `VoiceInput.plainMessage`. The cursor goes back in the box, so the
        // question can be typed instead.
        .onChange(of: voice.error) { _, error in
            guard let error else { return }
            toasts?.show(error, symbol: "mic.slash", tint: .pd.caution)
            isFocused = true
        }
        .onAppear(perform: arrive)
        .onChange(of: navigator.askHandoff?.id) { _, _ in takeHandoff(arriving: false) }
        .onChange(of: model.askComposerFocusRequested) { _, _ in takeFocusRequest(arriving: false) }
        .onChange(of: model.isGenerating) { _, generating in
            guard !generating else { return }
            sendQueued()
        }
    }

    /// Byte-for-byte the predicate `AppModel.send()` guards on, minus the
    /// brain: a question with nothing to answer it yet is still sent, and held.
    ///
    /// Two deciders answering one question is how a button lights up over a
    /// `send()` that returns immediately: `.whitespaces` does not include
    /// newlines, so a draft of one Return once left this enabled and the tap
    /// landed on nothing.
    private var canSend: Bool {
        !model.isGenerating
            && (!model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !model.attachments.isEmpty)
    }

    private func send() {
        isFocused = false
        model.send()
    }

    // MARK: - Attachments

    private var attachments: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: PDSpacing.xs) {
                ForEach(Array(model.attachments.enumerated()), id: \.offset) { index, data in
                    thumbnail(data, at: index)
                }
            }
            .padding(.horizontal, 2)
        }
        .frame(height: 62)
    }

    /// Extracted purely so the body type-checks: SwiftUI builds one expression
    /// per view body, and this one grew past what the solver will attempt.
    @ViewBuilder
    private func thumbnail(_ data: Data, at index: Int) -> some View {
        if let image = UIImage(data: data) {
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: 56, height: 56)
                .clipShape(RoundedRectangle(cornerRadius: PDRadius.iconTileSmall, style: .continuous))
                .overlay(alignment: .topTrailing) {
                    Button {
                        model.removeAttachment(at: index)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(Font.pd.secondary)
                            .symbolRenderingMode(.palette)
                            .foregroundStyle(.white, .black.opacity(0.6))
                            .pdHitTarget(alignment: .topTrailing)
                    }
                    .padding(-PDSpacing.xs)
                    .accessibilityLabel("Remove photo \(index + 1)")
                }
        }
    }

    /// Reads the picked items into memory and hands them to the model.
    ///
    /// Downscaled first: a modern iPhone photo is several thousand pixels wide
    /// and the projector sees a few hundred, so sending the original spends
    /// memory and encode time to produce the same tokens. It also has to be
    /// data rather than a file URL — the wire format carries base64, and a
    /// paired laptop cannot open a path on this phone.
    private func loadPicked() async {
        guard !picked.isEmpty else { return }
        for item in picked {
            guard let raw = try? await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: raw),
                  let shrunk = image.downscaled(to: 896)?.jpegData(compressionQuality: 0.8)
            else { continue }
            model.attach(shrunk)
        }
        picked = []
    }

    // MARK: - Voice

    private func startListening() {
        guard !voice.isListening, !model.isGenerating else { return }
        levels = []
        Task {
            if voice.authorization != .granted {
                if VoiceInput.hasNeverAsked {
                    withAnimation(PDMotion.gentle) {
                        primer = "Pocketd turns your voice into text on this iPhone. Next, tap **Allow**."
                    }
                    // Long enough to be read before the system's alert dims
                    // the screen; it stays readable beneath it.
                    try? await Task.sleep(for: .milliseconds(900))
                }
                // iOS shows each prompt once; asking again after a decision
                // re-reads it rather than nagging.
                let granted = await voice.requestAuthorization()
                withAnimation(PDMotion.gentle) { primer = nil }
                guard granted else {
                    toasts?.show(VoiceInput.refusedSentence, symbol: "mic.slash", tint: .pd.caution)
                    isFocused = true
                    return
                }
            }
            voice.reset()
            voice.start()
        }
    }

    private func finishListening() {
        guard voice.isListening else { return }
        voice.stop()
        Task {
            // The recogniser's last callback arrives after `endAudio()`, and it
            // carries the final word. Sending before it lands is how "remind me
            // at four" becomes "remind me at".
            let final = await voice.finalTranscript()
            levels = []
            let said = final.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !said.isEmpty else {
                isFocused = true
                return
            }
            model.draft = said
            model.record(.voiceAskUsed)
            // Unconditionally. With no brain yet it is held, in the box, with
            // the notice above saying it will go.
            model.send()
        }
    }

    // MARK: - Arriving with something to say

    /// The sheet has just come up.
    ///
    /// Every way in wants the cursor in the box — the field on Today, a detail
    /// page's composer, `pocketd://ask`, the "Ask Pocketd" control — except a
    /// question that is being sent: its answer is about to appear, and the
    /// keyboard would cover it.
    private func arrive() {
        if navigator.askHandoff != nil {
            takeHandoff(arriving: true)
        } else if model.askComposerFocusRequested {
            takeFocusRequest(arriving: true)
        } else {
            focus(after: true)
        }
    }

    /// A question handed over from Today, a detail page, Siri or a Shortcut.
    ///
    /// `send: true` asks it — in a new conversation unless this one was used
    /// in the last half hour, so a question about sleep never lands at the
    /// bottom of yesterday's chat (`askFresh`). `send: false` starts the
    /// sentence and leaves the cursor after it.
    private func takeHandoff(arriving: Bool) {
        guard let handoff = navigator.takeAskHandoff() else { return }
        if handoff.send {
            guard !model.isGenerating else {
                queued = handoff.draft
                model.draft = handoff.draft
                return
            }
            Task { await model.askFresh(handoff.draft) }
        } else {
            Task {
                await startFreshIfStale()
                model.draft = handoff.draft
                focus(after: arriving)
            }
        }
    }

    /// The handed-over question, once the reply it arrived during has ended —
    /// if the box still holds it. Somebody who has since typed something else
    /// has changed their mind.
    private func sendQueued() {
        guard let question = queued else { return }
        queued = nil
        guard model.draft.trimmingCharacters(in: .whitespacesAndNewlines) == question
            .trimmingCharacters(in: .whitespacesAndNewlines) else { return }
        Task { await model.askFresh(question) }
    }

    /// `askFresh`'s rule for a sentence that is started rather than sent.
    private func startFreshIfStale() async {
        guard !model.conversation.isEmpty, !model.isGenerating,
              Date().timeIntervalSince(model.lastConversationActivity) > AppModel.freshConversationAfter
        else { return }
        await model.newConversation()
    }

    /// The "Ask Pocketd" control, or a page's mic: the cursor in the box, and
    /// the microphone open where this phone can hear on-device.
    ///
    /// Both, in that order. Listening alone left nowhere to type when it
    /// failed — a mic tap that ended in a toast and an empty, unfocused box.
    /// The box is greyed while the microphone is open and takes the cursor
    /// back when it closes without sending.
    private func takeFocusRequest(arriving: Bool) {
        guard model.askComposerFocusRequested else { return }
        model.askComposerFocusRequested = false
        focus(after: arriving)
        guard voice.isSupported, !model.isGenerating else { return }
        Task {
            if arriving { try? await Task.sleep(for: .milliseconds(500)) }
            startListening()
        }
    }

    /// Puts the cursor in the box. On arrival the sheet is still sliding up,
    /// and a field that is not in a window yet cannot take focus — so the
    /// request waits for the presentation to land.
    private func focus(after arriving: Bool) {
        Task {
            if arriving { try? await Task.sleep(for: .milliseconds(450)) }
            isFocused = true
        }
    }
}

/// What the microphone is hearing, drawn.
///
/// A rolling history of the smoothed level rather than a decorative animation,
/// because the one question somebody holding a mic button has is "can it hear
/// me" — and a waveform that moves whether or not anybody is speaking answers it
/// wrongly.
struct VoiceWaveform: View {
    static let barCount = 28
    let levels: [Float]

    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(0..<Self.barCount, id: \.self) { index in
                Capsule()
                    .fill(Color.pd.accent.opacity(index < levels.count ? 0.9 : 0.25))
                    .frame(width: 3, height: height(at: index))
            }
            Spacer(minLength: 0)
            Text("Listening…")
                .font(Font.pd.secondary)
                .foregroundStyle(Color.pd.inkSecondary)
        }
        .frame(minHeight: 24)
        .padding(.horizontal, PDSpacing.s)
        .accessibilityElement()
        .accessibilityLabel("Listening")
    }

    private func height(at index: Int) -> CGFloat {
        guard index < levels.count else { return 3 }
        return 3 + CGFloat(levels[index]) * 21
    }
}

private extension UIImage {
    /// Longest edge capped, aspect preserved. Vision projectors work from a
    /// fixed small grid, so anything larger is bytes the model never reads.
    func downscaled(to longestEdge: CGFloat) -> UIImage? {
        let longest = max(size.width, size.height)
        guard longest > longestEdge else { return self }
        let scale = longestEdge / longest
        let target = CGSize(width: size.width * scale, height: size.height * scale)
        return UIGraphicsImageRenderer(size: target).image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }
    }
}
