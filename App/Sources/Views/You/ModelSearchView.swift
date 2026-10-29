import Foundation
import Observation
import PocketdKit
import SwiftUI

/// More brains › Search Hugging Face: brains other people have published.
///
/// The catalogue is a handful known to load on a phone — the right default and
/// the wrong ceiling — and this is the difference between "some brains" and
/// "any brain anyone has published". Deliberately two levels down: nobody
/// arrives at a new assistant wanting to browse a model hub.
///
/// What the hub returns is ranked for this phone before it is shown
/// (`HuggingFaceSearch.ranked`): repos named for removing a model's safety
/// training are not listed at all, nor is one whose name says even its
/// smallest useful file cannot fit; the rest come comfortable first. Inside a
/// repo, a size that cannot fit says "Too big for this iPhone" instead of
/// offering a download (`HuggingFaceSearch.offers`).
///
/// The search box is in the page rather than `.searchable`, under the page's
/// own heading: a field in the bar above a heading that says "Search" reads
/// as two searches.
struct ModelSearchView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    @State private var finder = RepositoryFinder()
    @State private var query = ""
    /// Bumped by Try again, and part of the task id so a retry runs down the
    /// same cancellable path as a keystroke. A loose `Task { }` would not be
    /// cancelled by the next keystroke and could land on top of it.
    @State private var attempt = 0
    @State private var chosen: HuggingFaceSearch.Repository?
    @FocusState private var searching: Bool

    var body: some View {
        YouDetailScreen(
            title: "Search Hugging Face",
            subtitle: "Pocketd hasn't checked these, so they count as chat only."
        ) {
            searchField
            results
        }
        .task(id: SearchKey(query: query, attempt: attempt)) {
            await finder.search(query, budget: model.budget)
        }
        #if DEBUG
        .onAppear {
            // A screenshot run arrives with something already typed. See
            // `YouStage`; both this and the type are inside `#if DEBUG`.
            if let staged = YouStage.searchQuery, query.isEmpty { query = staged }
        }
        #endif
        .sheet(item: $chosen) { repository in
            RepositoryFilesView(repository: repository) { record in
                model.download(record)
                chosen = nil
                // Back to More brains, where the download is already drawing
                // its progress. There is nothing left for this page to say.
                dismiss()
            }
        }
    }

    // MARK: - The box

    private var searchField: some View {
        HStack(spacing: PDSpacing.xs) {
            Image(systemName: "magnifyingglass")
                .font(Font.pd.body)
                .foregroundStyle(Color.pd.inkSecondary)
                .accessibilityHidden(true)
            TextField("qwen3, llama, gemma…", text: $query)
                .font(Font.pd.body)
                .foregroundStyle(Color.pd.ink)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .submitLabel(.search)
                .focused($searching)
                .accessibilityLabel("Search Hugging Face")
            if !query.isEmpty {
                Button {
                    query = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(Font.pd.body)
                        .foregroundStyle(Color.pd.inkTertiary)
                        .pdHitTarget()
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear the search")
            }
        }
        .padding(.horizontal, PDSpacing.m)
        .padding(.vertical, PDSpacing.xxs)
        .frame(minHeight: PDMetric.askBarHeight)
        .background(Color.pd.chip, in: Capsule(style: .continuous))
        .overlay(Capsule(style: .continuous).strokeBorder(Color.pd.cardStroke, lineWidth: 1))
    }

    // MARK: - What came back

    @ViewBuilder
    private var results: some View {
        switch finder.phase {
        case .prompt:
            EmptyCard(symbol: "magnifyingglass", title: "Type a family name.", detail: "qwen3, llama, gemma or phi work best.")
        case .searching:
            EmptyCard(symbol: "ellipsis", title: "Looking.", detail: "Searching what has been published.")
        case let .noMatches(term):
            EmptyCard(
                symbol: "questionmark.circle",
                title: "Nothing for “\(term)”.",
                detail: "Only phone-sized files show up here. Try the family name on its own."
            )
        case let .unavailable(id, reason):
            EmptyCard(symbol: "exclamationmark.circle", title: "Pocketd can't run “\(id)”.", detail: reason)
        case let .failed(message):
            EmptyCard(symbol: "wifi.exclamationmark", title: "That didn't work.", detail: message) {
                PillButton(title: "Try again", symbol: "arrow.clockwise") { attempt += 1 }
            }
        case let .results(repositories):
            YouSection("Best for this iPhone first") {
                ForEach(Array(repositories.enumerated()), id: \.element.id) { index, repository in
                    if index > 0 { RowDivider(inset: 0) }
                    repositoryRow(repository)
                }
            }
        }
    }

    @ViewBuilder
    private func repositoryRow(_ repository: HuggingFaceSearch.Repository) -> some View {
        if repository.gated {
            // Listed and inert. Hiding it makes somebody searching for a
            // well-known name conclude the search is broken; linking it means
            // a refusal after the tap.
            RowShell(title: repository.name, detail: "Needs an account Pocketd doesn't have") {
                EmptyView()
            }
            .opacity(0.75)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("\(repository.name) by \(repository.owner). Needs an account Pocketd doesn't have.")
        } else {
            NavRow(
                title: repository.name,
                detail: "\(repository.owner) · \(repository.downloadsText) downloads"
            ) {
                searching = false
                chosen = repository
            }
            // Spelled out rather than "95K", which VoiceOver reads as a letter.
            .accessibilityLabel("\(repository.name) by \(repository.owner). \(repository.downloads.formatted()) downloads.")
        }
    }
}

// MARK: - Files in one repository

/// The phone-sized files in one repository, one of which becomes a download.
///
/// A sheet rather than a push: it is a choice made and finished in one go —
/// you pick a size, it starts, and you are back on More brains watching it.
///
/// Picture parts (projector files) are left out: one is not a brain, and it
/// comes down automatically with whichever size is picked, so it shows up as
/// part of that size — which is what it is.
private struct RepositoryFilesView: View {
    let repository: HuggingFaceSearch.Repository
    let onAdd: (ModelRecord) -> Void

    @Environment(AppModel.self) private var model
    @Environment(\.openURL) private var openURL
    @State private var loader = FileLoader()
    @State private var attempt = 0

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: PDSpacing.cardGap) {
                VStack(alignment: .leading, spacing: PDSpacing.xs) {
                    Text(repository.name)
                        .font(Font.pd.titleDetail)
                        .foregroundStyle(Color.pd.ink)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityAddTraits(.isHeader)
                    Text("Published by \(repository.owner)")
                        .pdSubtitle()
                }
                .padding(.top, PDSpacing.l)

                content
            }
            .padding(.horizontal, PDSpacing.gutter)
            .padding(.bottom, PDSpacing.xxl)
        }
        .scrollIndicators(.hidden)
        .pdCanvas()
        .presentationDragIndicator(.visible)
        .task(id: attempt) { await loader.load(repository: repository.id) }
    }

    @ViewBuilder
    private var content: some View {
        switch loader.phase {
        case .loading:
            EmptyCard(symbol: "ellipsis", title: "Reading it.", detail: "Listing the sizes it comes in.")
        case .empty:
            EmptyCard(
                symbol: "shippingbox",
                title: "Nothing to download here.",
                detail: "Look for the same name ending in GGUF, published by someone else."
            )
        case let .failed(message):
            EmptyCard(symbol: "wifi.exclamationmark", title: "Couldn't read it.", detail: message) {
                PillButton(title: "Try again", symbol: "arrow.clockwise") { attempt += 1 }
            }
        case let .ready(weights, projector):
            YouSection(
                "Sizes",
                footnote: projector == nil
                    ? "Bigger files answer a little better and need more memory."
                    : "Bigger files answer a little better and need more memory. Each size includes the part that reads pictures."
            ) {
                let offers = HuggingFaceSearch.offers(weights, for: model.budget, includingTooBig: true)
                ForEach(Array(offers.enumerated()), id: \.element.id) { index, offer in
                    if index > 0 { RowDivider(inset: 0) }
                    row(for: offer, projector: projector)
                }
            }
            if let url = URL(string: "https://huggingface.co/\(repository.id)") {
                PDCard(padding: PDSpacing.m) {
                    LinkRow(symbol: "safari", title: "See it on Hugging Face") { openURL(url) }
                }
            }
        }
    }

    @ViewBuilder
    private func row(for offer: HuggingFaceSearch.Offer, projector: HuggingFaceSearch.File?) -> some View {
        let file = offer.file
        let record = record(for: file, projector: projector)
        // Offered only when the file fits comfortably on its own and as the
        // whole download, which also counts the picture part.
        let isOffered = offer.isOffered && model.fit(for: record) == .comfortable
        let installed = model.isInstalled(record)
        let downloading = model.transfer(for: record.id)?.isActive == true

        RowShell(title: pdBytes(record.totalDownloadBytes), detail: detail(file: file)) {
            if installed {
                quiet("On this iPhone")
            } else if downloading {
                quiet("Getting it")
            } else if !isOffered {
                quiet(offer.tooBigNote ?? "Too big for this iPhone")
            } else {
                PillButton(title: "Get", symbol: "arrow.down") { onAdd(record) }
                    .accessibilityLabel("Download the \(pdBytes(record.totalDownloadBytes)) size")
            }
        }
    }

    /// The file's own label, kept because two sizes are otherwise told apart
    /// by nothing — plus a steer for the one most people want.
    private func detail(file: HuggingFaceSearch.File) -> String {
        var parts = [file.quantization == "unknown" ? "Unlabelled" : file.quantization]
        if file.quantization == "Q4_K_M" { parts.append("Usual choice") }
        return pdSeparated(parts.joined(separator: " · "))
    }

    private func quiet(_ text: String) -> some View {
        Text(text)
            .font(Font.pd.secondary.weight(.medium))
            .foregroundStyle(Color.pd.inkSecondary)
            .multilineTextAlignment(.trailing)
    }

    /// The context length is the server's cap rather than anything the hub
    /// said: it does not report a window, and the served context is capped
    /// there anyway.
    private func record(for file: HuggingFaceSearch.File, projector: HuggingFaceSearch.File?) -> ModelRecord {
        HuggingFaceSearch.record(
            repository: repository.id,
            file: file,
            projector: projector,
            contextLength: model.configuration.maxContextTokens
        )
    }
}

// MARK: - An empty state that says what to do

/// A card that says what happened and what to do about it.
///
/// Not `ContentUnavailableView`: that one centres a grey glyph in the middle of
/// an empty screen, which on a page with a search box at the top leaves the
/// answer a long way from the question.
struct EmptyCard<Actions: View>: View {
    let symbol: String
    let title: String
    let detail: String
    @ViewBuilder var actions: Actions

    var body: some View {
        PDCard {
            VStack(alignment: .leading, spacing: PDSpacing.s) {
                IconTile(symbol: symbol)
                Text(title)
                    .font(Font.pd.headline)
                    .foregroundStyle(Color.pd.ink)
                    .fixedSize(horizontal: false, vertical: true)
                Text(detail)
                    .pdBody()
                    .fixedSize(horizontal: false, vertical: true)
                actions
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityElement(children: .contain)
    }
}

extension EmptyCard where Actions == EmptyView {
    init(symbol: String, title: String, detail: String) {
        self.init(symbol: symbol, title: title, detail: detail) { EmptyView() }
    }
}

// MARK: - State

/// Repository search, and the reason results never arrive out of order.
@MainActor
@Observable
private final class RepositoryFinder {
    enum Phase {
        /// Nothing worth searching for has been typed yet.
        case prompt
        case searching
        case results([HuggingFaceSearch.Repository])
        case noMatches(String)
        /// A repository named exactly, which exists but is not offered, and
        /// the sentence that says why.
        case unavailable(id: String, reason: String)
        case failed(String)
    }

    private(set) var phase: Phase = .prompt

    func search(_ raw: String, budget: DeviceBudget) async {
        let query = HuggingFaceSearch.normalizedQuery(raw)
        // The same floor the hub's search enforces: one character matches half
        // of it and none of it usefully.
        guard query.count >= 2 else {
            phase = .prompt
            return
        }

        // Spelling "qwen3" is five requests without this, four of them thrown
        // away. The sleep is cancelled along with the task, so a fast typist
        // pays for nothing.
        do {
            try await Task.sleep(for: .milliseconds(300))
        } catch {
            return
        }

        phase = .searching
        do {
            // Fifty rather than the hub's twenty: ranking drops the repos this
            // phone cannot use, and "qwen3" left three of twenty standing.
            let found = HuggingFaceSearch.ranked(
                try await HuggingFaceClient.shared.repositories(matching: query, limit: 50),
                for: budget
            )
            // Named exactly and still missing: ask the hub about that one
            // repository, so a Core ML speech model or a repo with no GGUF in
            // it says so instead of looking like a typo.
            if found.isEmpty, let id = HuggingFaceSearch.repositoryID(in: query),
               let details = try await HuggingFaceClient.shared.details(of: id) {
                if let reason = HuggingFaceSearch.whyNotOffered(details, for: budget) {
                    phase = .unavailable(id: details.repository.id, reason: reason)
                } else {
                    phase = .results([details.repository])
                }
                return
            }
            phase = found.isEmpty ? .noMatches(query) : .results(found)
        } catch {
            guard !isCancellation(error) else { return }
            phase = .failed(searchMessage(for: error))
        }
    }
}

/// The phone-sized files in one repository.
@MainActor
@Observable
private final class FileLoader {
    enum Phase {
        case loading
        /// Sizes the user can pick, plus the picture part that comes with
        /// whichever one they pick.
        case ready(weights: [HuggingFaceSearch.File], projector: HuggingFaceSearch.File?)
        case empty
        case failed(String)
    }

    private(set) var phase: Phase = .loading

    func load(repository: String) async {
        phase = .loading
        do {
            let files = try await HuggingFaceClient.shared.files(in: repository)
            let projector = files.first { $0.isProjector }
            let weights = files.filter { !$0.isProjector }
            phase = weights.isEmpty ? .empty : .ready(weights: weights, projector: projector)
        } catch {
            guard !isCancellation(error) else { return }
            phase = .failed(searchMessage(for: error))
        }
    }
}

// MARK: - Shared plumbing

/// One client for the screen, with a request timeout short enough that a dead
/// network fails visibly. URLSession's default is sixty seconds, which on a
/// phone is indistinguishable from a spinner that never stops. Ephemeral
/// because there is no reason for what somebody searched for to outlive the
/// search on a device whose entire premise is that nothing leaves it.
private enum HuggingFaceClient {
    static let shared: HuggingFaceSearch = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 15
        return HuggingFaceSearch(session: URLSession(configuration: configuration))
    }()
}

/// Identity for `.task(id:)`: the query plus a retry counter, so Try again
/// re-runs the same cancellable task rather than starting an unmanaged one.
private struct SearchKey: Equatable {
    let query: String
    let attempt: Int
}

/// URLSession reports a cancelled request as a thrown `URLError`, not as
/// `CancellationError`, and a request cancelled because the user kept typing is
/// not a failure worth showing them.
private func isCancellation(_ error: any Error) -> Bool {
    if error is CancellationError { return true }
    if let error = error as? URLError, error.code == .cancelled { return true }
    return false
}

/// One plain sentence per failure, and deliberately fixed strings: a URL
/// error's `userInfo` carries a signed CDN URL and the whole resume blob, so
/// `String(describing:)` on one ends up on screen — that has happened before.
private func searchMessage(for error: any Error) -> String {
    if let error = error as? HuggingFaceSearch.SearchError {
        switch error {
        case .httpStatus(429):
            return "Hugging Face is busy. Wait a minute and try again."
        case let .httpStatus(code) where code == 401 || code == 403:
            return "That one needs an account Pocketd doesn't have."
        case .httpStatus, .malformed:
            return "Hugging Face didn't answer properly. Try again in a moment."
        }
    }
    if let error = error as? URLError {
        switch error.code {
        case .notConnectedToInternet, .networkConnectionLost:
            return "This iPhone is offline. Searching needs the internet; running a brain doesn't."
        case .timedOut:
            return "Hugging Face took too long to answer. Try again."
        default:
            return "Hugging Face can't be reached from this network."
        }
    }
    return "Hugging Face can't be reached from this network."
}

// MARK: - Previews

#Preview("Search light") {
    NavigationStack { ModelSearchView() }
        .environment(AppModel())
        .environment(Navigator())
}

#Preview("Search dark") {
    NavigationStack { ModelSearchView() }
        .environment(AppModel())
        .environment(Navigator())
        .preferredColorScheme(.dark)
}
