import Foundation
import Observation
import PocketdKit
import StoreKit
#if DEBUG
import os
#endif

/// Pocketd Pro on this phone: whether it is on, what it costs here, and the
/// two ways in — buying it, or restoring a purchase made before.
///
/// The only type that talks to the App Store. The rules — who is early, what
/// is free — are the Kit's (`ProAccess`); this brings them three facts: whether
/// a verified purchase is among the current entitlements, which App Store
/// signed the app transaction, and the original version it records. A fact
/// that cannot be read, or does not verify, is brought as nil, and nil never
/// grants anything.
///
/// Created once, by `PocketdApp`, because the transaction listener has to run
/// from launch: a purchase a parent approved later (Ask to Buy), one that
/// finished after Pocketd was killed with Apple's sheet up, and a refund all
/// arrive through `Transaction.updates`, and a listener started later misses
/// whatever was waiting at launch.
///
/// The price is fetched when the sheet opens, not at launch: the App Store
/// request carries nothing about the person, but a phone that never opens the
/// sheet has no reason to make it.
///
/// Nothing here is counted for analytics. See `ProAccess`.
@Observable
@MainActor
final class ProStore {

    /// What the price button can say.
    enum Offer: Equatable {
        /// Asking the App Store. "Unlock Pro", waiting.
        case loading
        /// StoreKit's own price string for this storefront — "$9.99", "9,99 €".
        case ready(displayPrice: String)
        /// The product did not load: offline, not sold here, or not set up.
        case unavailable
    }

    /// Where a purchase started from the sheet stands.
    enum Purchase: Equatable {
        case idle
        /// Apple's sheet is up, or the App Store is answering.
        case purchasing
        /// Ask to Buy: a parent has to approve it. The approval arrives through
        /// the listener, possibly long after the sheet has closed.
        case pending
        case failed(ProFailure)
    }

    enum RestoreOutcome: Equatable {
        case restored
        /// The App Store answered, and this Apple Account has no purchase.
        case nothingToRestore
        /// The person closed Apple's sign-in. Nothing to say.
        case cancelled
        case failed(ProFailure)
    }

    /// Whether Pro is on, and why. Locked until the App Store says otherwise.
    private(set) var entitlement: ProEntitlement = .locked
    var isPro: Bool { entitlement.isPro }
    /// Whether the first look at the App Store's records has finished.
    private(set) var hasChecked = false
    private(set) var offer: Offer = .loading
    private(set) var purchase: Purchase = .idle
    private(set) var isRestoring = false

    /// The facts the Kit decides from. Kept so a purchase can be added to
    /// them without reading the app transaction again.
    @ObservationIgnored private var ownsPurchase = false
    @ObservationIgnored private var app: AppFacts?
    @ObservationIgnored private var product: Product?

    /// The app transaction, reduced to what the early-supporter rule reads.
    private struct AppFacts {
        var environment: ProAccess.StoreEnvironment
        var originalAppVersion: String
    }

    /// How long the sheet waits for a price before it says Pro isn't available.
    private static let offerTimeout: Duration = .seconds(12)

    init() {
        // The store lives as long as the app does, so the listener is never
        // cancelled; `weak` only keeps a preview's throwaway store collectable.
        Task { [weak self] in
            for await update in Transaction.updates {
                await self?.receive(update)
            }
        }
        // Pocketd Pro is listed on the App Store's page for Pocketd, and a tap
        // there arrives here. Listened for from launch, beside the transaction
        // listener and not in the sheet: when the tap is what opened the app,
        // the intent arrives before any view exists.
        Task { [weak self] in
            for await intent in PurchaseIntent.intents {
                await self?.receive(intent)
            }
        }
        Task { [weak self] in await self?.refresh() }
    }

    // MARK: - Is it on

    /// Reads the current entitlements and the app transaction, and decides.
    func refresh() async {
        #if DEBUG
        if let facts = ProStage.facts {
            ownsPurchase = facts.purchased || stagedPurchase
            app = facts.environment.map { AppFacts(environment: $0, originalAppVersion: facts.originalAppVersion ?? "") }
            decide()
            hasChecked = true
            return
        }
        #endif
        let purchased = await Self.ownsPro()
        let app = await Self.readAppTransaction()
        #if DEBUG
        ownsPurchase = purchased || stagedPurchase
        #else
        ownsPurchase = purchased
        #endif
        self.app = app
        decide()
        hasChecked = true
    }

    /// Reads again only when the app transaction has not been read yet — it
    /// may have failed offline at launch, and an early supporter should not
    /// meet the sheet because of it.
    func refreshIfNeeded() async {
        guard app == nil, !isPro else { return }
        await refresh()
    }

    /// Waits, briefly, for the first look at the App Store's records, so a
    /// gate tapped in the first moments after launch does not offer Pro to
    /// someone who has it. Two seconds at most: past that, the sheet is the
    /// better answer than a button that does nothing.
    func settle() async {
        var waited = 0
        while !hasChecked, waited < 20 {
            try? await Task.sleep(for: .milliseconds(100))
            waited += 1
        }
    }

    private func decide() {
        #if POCKETD_SIDELOAD
        // A build signed by its owner (ESign, AltStore) has no App Store to
        // buy from, so it is built with Pro on. Only the unsigned-IPA workflow
        // sets this condition; the App Store build never does.
        entitlement = .purchased
        return
        #endif
        entitlement = ProAccess.entitlement(
            purchased: ownsPurchase,
            environment: app?.environment,
            originalAppVersion: app?.originalAppVersion
        )
        Self.log("entitlement \(entitlement.rawValue): purchased \(ownsPurchase), \(app?.environment.rawValue ?? "no app transaction") \(app?.originalAppVersion ?? "")")
    }

    /// A verified, unrevoked purchase of Pro among this Apple Account's
    /// current entitlements. Works offline: StoreKit keeps them on the phone.
    private static func ownsPro() async -> Bool {
        for await result in Transaction.currentEntitlements {
            if case let .verified(transaction) = result,
               transaction.productID == ProAccess.productID,
               transaction.revocationDate == nil {
                return true
            }
        }
        return false
    }

    /// The app transaction, only when it verifies. An unverified one, or a
    /// read that throws, is nil — and nil is never an early supporter.
    private static func readAppTransaction() async -> AppFacts? {
        do {
            guard case let .verified(transaction) = try await AppTransaction.shared else {
                log("app transaction did not verify")
                return nil
            }
            log("app transaction: \(transaction.environment.rawValue), original \(transaction.originalAppVersion)")
            return AppFacts(
                environment: ProAccess.StoreEnvironment(transaction.environment),
                originalAppVersion: transaction.originalAppVersion
            )
        } catch {
            log("app transaction unreadable: \(error)")
            return nil
        }
    }

    /// Everything the listener hears: a purchase approved by a parent, one
    /// finished elsewhere, a restore, a refund.
    private func receive(_ update: VerificationResult<Transaction>) async {
        guard case let .verified(transaction) = update else { return }
        await transaction.finish()
        guard transaction.productID == ProAccess.productID else { return }
        if purchase == .pending { purchase = .idle }
        await refresh()
    }

    /// A purchase somebody started on the App Store's page for Pocketd.
    ///
    /// The same `buy()` the sheet's button runs, with the product the App Store
    /// handed over; the entitlement it produces is decided by `decide()` like
    /// any other purchase. The App Store's records are read first, and waited
    /// for rather than given `settle()`'s two seconds: on a cold start they are
    /// still loading, and deciding without them would charge an early
    /// supporter, or somebody who bought Pro on another phone, for what they
    /// already have.
    private func receive(_ intent: PurchaseIntent) async {
        await refresh()
        let response = ProAccess.response(toStoreIntentFor: intent.product.id, entitlement: entitlement)
        Self.log("purchase intent \(intent.product.id): \(response.rawValue)")
        guard response == .purchase else { return }
        product = intent.product
        offer = .ready(displayPrice: intent.product.displayPrice)
        await buy(offer: intent.offer)
    }

    // MARK: - The price

    /// Asks the App Store for the product and its price here. A price already
    /// on screen stays there while it asks again.
    func loadOffer() async {
        #if DEBUG
        if let staged = ProStage.offer {
            offer = staged
            return
        }
        #endif
        if product == nil { offer = .loading }
        // A request that neither answers nor fails would leave the button
        // spinning for as long as the sheet is open.
        let timeout = Task { [weak self] in
            try? await Task.sleep(for: Self.offerTimeout)
            guard !Task.isCancelled, let self, self.product == nil, self.offer == .loading else { return }
            self.offer = .unavailable
        }
        defer { timeout.cancel() }
        do {
            let products = try await Product.products(for: [ProAccess.productID])
            Self.log("products: \(products.map(\.id))")
            if let found = products.first(where: { $0.id == ProAccess.productID }) {
                product = found
                offer = .ready(displayPrice: found.displayPrice)
            } else if product == nil {
                offer = .unavailable
            }
        } catch {
            Self.log("products failed: \(error)")
            if product == nil { offer = .unavailable }
        }
    }

    // MARK: - Buying

    /// Buys Pro. True once it is on.
    ///
    /// `offer` is the one an App Store purchase intent carried. Pro is a
    /// one-time purchase and has none today; a win-back offer is the only kind
    /// an intent can carry, and it is passed on rather than dropped.
    @discardableResult
    func buy(offer intentOffer: Product.SubscriptionOffer? = nil) async -> Bool {
        guard purchase != .purchasing else { return false }
        #if DEBUG
        if ProStage.offer != nil { return await stagedBuy() }
        #endif
        guard let product else {
            purchase = .failed(.unavailable)
            return false
        }
        var options: Set<Product.PurchaseOption> = []
        if let intentOffer, intentOffer.type == .winBack {
            options.insert(.winBackOffer(intentOffer))
        }
        purchase = .purchasing
        do {
            switch try await product.purchase(options: options) {
            case let .success(.verified(transaction)):
                await transaction.finish()
                ownsPurchase = true
                decide()
                purchase = .idle
            case .success(.unverified):
                purchase = .failed(.unverified)
            case .pending:
                purchase = .pending
            case .userCancelled:
                purchase = .idle
            @unknown default:
                purchase = .idle
            }
        } catch {
            purchase = Self.failure(for: error).map(Purchase.failed) ?? .idle
        }
        return isPro
    }

    /// Asks the App Store for this Apple Account's purchases again — the
    /// "Restore purchase" App Review requires. Apple may ask the person to
    /// sign in. It also refreshes the app transaction, which is what puts an
    /// early supporter right if the first read failed.
    func restore() async -> RestoreOutcome {
        guard !isRestoring else { return .cancelled }
        isRestoring = true
        defer { isRestoring = false }
        #if DEBUG
        if ProStage.isStaged {
            try? await Task.sleep(for: .seconds(1))
            await refresh()
            return isPro ? .restored : .nothingToRestore
        }
        #endif
        do {
            try await AppStore.sync()
        } catch {
            guard let failure = Self.failure(for: error) else { return .cancelled }
            return .failed(failure)
        }
        await refresh()
        return isPro ? .restored : .nothingToRestore
    }

    /// Forgets the last failure, so a sheet opened again starts clean. A
    /// purchase waiting for approval is still waiting, and stays said.
    func clearFailure() {
        if case .failed = purchase { purchase = .idle }
    }

    /// The sentence for what went wrong, or nil when the person cancelled.
    private static func failure(for error: any Error) -> ProFailure? {
        if let error = error as? StoreKitError {
            switch error {
            case .userCancelled: return nil
            case .networkError: return .offline
            case .notAvailableInStorefront: return .unavailable
            default: return .other
            }
        }
        if let error = error as? Product.PurchaseError {
            switch error {
            case .purchaseNotAllowed: return .notAllowed
            case .productUnavailable: return .unavailable
            default: return .other
            }
        }
        return .other
    }

    // MARK: - Debug

    #if DEBUG
    /// A staged purchase that went through, kept for the rest of this run.
    @ObservationIgnored private var stagedPurchase = false

    /// What a tap on a staged price does — see `ProStage`.
    private func stagedBuy() async -> Bool {
        guard case .ready = offer else {
            purchase = .failed(.unavailable)
            return false
        }
        purchase = .purchasing
        try? await Task.sleep(for: .seconds(1.2))
        switch ProStage.buy {
        case .success:
            stagedPurchase = true
            purchase = .idle
            await refresh()
        case .pending:
            purchase = .pending
        case .fail:
            purchase = .failed(.other)
        case .offline:
            purchase = .failed(.offline)
        case .cancel:
            purchase = .idle
        }
        return isPro
    }

    private static let logger = Logger(subsystem: "dev.pocketd.app", category: "pro")
    #endif

    /// What StoreKit answered, in a Debug build's log. Nothing in Release.
    private static func log(_ message: @autoclosure () -> String) {
        #if DEBUG
        let text = message()
        logger.debug("\(text, privacy: .public)")
        #endif
    }
}

private extension ProAccess.StoreEnvironment {
    init(_ environment: AppStore.Environment) {
        switch environment {
        case .production: self = .production
        case .sandbox: self = .sandbox
        case .xcode: self = .xcode
        default: self = .other
        }
    }
}

// MARK: - Staging on a simulator

#if DEBUG
/// Stands in for the App Store on a simulator, which has no App Store account
/// and no StoreKit configuration unless Xcode launched the app.
///
///     xcrun simctl launch <udid> dev.pocketd.app -pocketd.proOverride off      # what App Review's phone reports: sandbox, "1.0"
///     xcrun simctl launch <udid> dev.pocketd.app -pocketd.proOverride early    # an App Store install of 1.0.1 (1789638888)
///     xcrun simctl launch <udid> dev.pocketd.app -pocketd.proOverride on       # bought
///     xcrun simctl launch <udid> dev.pocketd.app -pocketd.proOverride sandbox:1789638888   # any environment:originalAppVersion
///         -pocketd.proOffer '$9.99'        # the price button shows this; also: loading, unavailable
///         -pocketd.proPurchase pending     # what a tap on a staged price does: success (default), pending, fail, offline, cancel
///         -pocketd.proSheet                # the sheet, open at launch
///
/// `-pocketd.proOverride` feeds the same `ProAccess.entitlement` the App
/// Store's facts would, so `off` and `sandbox:…` exercise the rule that keeps
/// a reviewer from being early, and `early` the one that keeps v1 users whole.
///
/// Entirely inside `#if DEBUG`, so none of these strings ships in a Release
/// build — and every flag is longer than 15 bytes on purpose. Swift keeps a
/// literal that short inside the instruction stream rather than as bytes in
/// the binary, where `strings` cannot see it, and a Release check that greps
/// for `pocketd.pro` would pass whether the code shipped or not.
enum ProStage {
    struct Facts {
        var purchased: Bool
        var environment: ProAccess.StoreEnvironment?
        var originalAppVersion: String?
    }

    enum Buy: String {
        case success, pending, fail, offline, cancel
    }

    private static var arguments: [String] { ProcessInfo.processInfo.arguments }

    private static func value(after flag: String) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }

    /// The facts `-pocketd.proOverride` asks for, or nil to ask StoreKit.
    static var facts: Facts? {
        guard let raw = value(after: "-pocketd.proOverride") else { return nil }
        switch raw {
        case "on":
            return Facts(purchased: true, environment: .production, originalAppVersion: "1792600000")
        case "off":
            return Facts(purchased: false, environment: .sandbox, originalAppVersion: "1.0")
        case "early":
            return Facts(purchased: false, environment: .production, originalAppVersion: "1789638888")
        default:
            let parts = raw.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2, let environment = ProAccess.StoreEnvironment(rawValue: parts[0]) else { return nil }
            return Facts(purchased: false, environment: environment, originalAppVersion: parts[1])
        }
    }

    /// The price button `-pocketd.proOffer` asks for, or nil to ask StoreKit.
    static var offer: ProStore.Offer? {
        guard let raw = value(after: "-pocketd.proOffer") else { return nil }
        switch raw {
        case "loading": return .loading
        case "unavailable": return .unavailable
        default: return .ready(displayPrice: raw)
        }
    }

    /// What `-pocketd.proPurchase` says a tap on a staged price does.
    static var buy: Buy {
        value(after: "-pocketd.proPurchase").flatMap(Buy.init(rawValue:)) ?? .success
    }

    static var opensSheet: Bool { arguments.contains("-pocketd.proSheet") }

    /// Whether anything here replaces StoreKit for this run.
    static var isStaged: Bool { facts != nil || offer != nil }
}
#endif
