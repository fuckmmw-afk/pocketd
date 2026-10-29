import PocketdKit
import SwiftUI

/// One downloadable speech model: what it is, and a button for the state it
/// is in. Used in Server settings and when search names its repository.
struct SpeechModelRow: View {
    let model: SpeechModel
    @State private var store = SpeechModelStore.shared

    var body: some View {
        switch store.state(of: model) {
        case .notInstalled:
            RowShell(title: model.displayName, detail: "\(model.languages) · \(pdDownloadLine(model.approximateBytes))") {
                PillButton(title: "Download", symbol: "arrow.down") { store.download(model) }
                    .accessibilityLabel("Download \(model.displayName)")
            }
        case let .downloading(fraction):
            VStack(alignment: .leading, spacing: PDSpacing.xs) {
                Text("Downloading \(model.displayName) · \(Int((fraction * 100).rounded(.down)))%")
                    .font(Font.pd.secondary)
                    .foregroundStyle(Color.pd.inkSecondary)
                ProgressView(value: fraction)
                    .tint(Color.pd.accent)
                    .accessibilityHidden(true)
            }
            .padding(.vertical, PDSpacing.s)
        case .installed:
            RowShell(title: model.displayName, detail: "Installed · \(model.languages)") {
                Button("Remove", role: .destructive) { store.remove(model) }
                    .font(Font.pd.secondary.weight(.semibold))
                    .accessibilityLabel("Remove \(model.displayName)")
            }
        case let .failed(message):
            RowShell(title: model.displayName, detail: message) {
                PillButton(title: "Try again", symbol: "arrow.clockwise") { store.download(model) }
            }
        }
    }
}

/// Every speech model, for Server settings.
struct SpeechModelsSection: View {
    var body: some View {
        YouSection(
            "Speech recognition",
            footnote: "Runs on this iPhone's Neural Engine. Once one is downloaded, the microphone uses it, and other devices can send audio to /v1/audio/transcriptions."
        ) {
            ForEach(Array(SpeechModels.all.enumerated()), id: \.element.id) { index, model in
                if index > 0 { RowDivider(inset: 0) }
                SpeechModelRow(model: model)
            }
        }
        .onAppear { SpeechModelStore.shared.refresh() }
    }
}
