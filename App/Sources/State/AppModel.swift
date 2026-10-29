import Foundation
import LocalLLMClient
import Observation
import PocketdKit
import SwiftUI

/// The app's state, and the composition root that builds it.
///
/// One `@Observable` class, and only its stored properties and its lifecycle
/// live in this file: `init`, `bootstrap` and the scene phase. Everything it
/// does is in an extension named for the area it belongs to — `+Brain`,
/// `+Chat`, `+Abilities`, `+Server`, `+Schedule`, `+Analytics`, `+Briefing`,
/// `+Reminders`, `+Routines` — and the stored properties below are grouped the
/// same way.
///
/// Most setters are module-internal rather than `private(set)`, and that is a
/// limit of the language rather than an invitation: Swift scopes a `private`
/// setter to one file, and the methods that write these properties live in the
/// extension files. Views read this state and call methods; nothing outside
/// `State/` assigns to it.
@MainActor
@Observable
final class AppModel {

    // MARK: - Server

    private(set) var serverState: InferenceServer.State = .stopped
    private(set) var log: [RequestLogEntry] = []
    private(set) var pairing = PairingSession.Snapshot()
    private(set) var condition: ServeCondition = .ok
    /// A dimmed, burn-in-safe screen for a phone left serving on a desk. The
    /// display draws from the same thermal and power budget as the GPU, so a
    /// bright screen literally costs tokens per second.
    /// Deliberately NOT persisted. Desk mode hides the tab bar, the status bar
    /// and the home indicator, so restoring it on launch means the app opens on
    /// a black screen with no visible way out — and force-quitting, the one
    /// recovery every user knows, lands you straight back in it.
    ///
    /// It is also the switch that arms scheduled prompt tasks. See
    /// `restartDeskSweep`: this is the app's only unattended state where a model
    /// may legally run, so turning it on starts a sweep and leaving it on keeps
    /// one on a timer.
    var deskMode = false {
        didSet {
            guard deskMode != oldValue else { return }
            restartDeskSweep()
        }
    }
    var configuration: ServerConfiguration {
        didSet { persistConfiguration() }
    }

    /// What the user asked for, as distinct from what the socket is doing.
    /// iOS tears the listener down on suspend, so the two diverge constantly and
    /// only this flag knows whether to put it back on the next foreground.
    var serverShouldRun = false
    var lastServerError: String?

    // MARK: - Brain: what is on this phone and what is in memory

    var installed: [ModelRecord] = []

    /// Every model arriving on this phone, keyed by id, with the record.
    ///
    /// One dictionary rather than the three it replaced — progress, error and
    /// paused, each keyed by an id and none of them holding the model. That
    /// shape worked only because the eight curated records are compiled into
    /// the binary and could be looked up by id later. Anything from Hugging
    /// Face has no such entry, so a search download was an id with no record,
    /// and every screen here builds its list out of records. The bytes moved
    /// for minutes and not one pixel changed.
    var transfers: [String: ModelTransfer] = [:] {
        // One hook for the download's whole state machine. The seven sites that
        // mutate this dictionary are start, progress, finish, pause, fail and
        // two cancels; driving the Live Activity from each is how one of them
        // gets forgotten and an activity outlives its download.
        didSet { DownloadActivityController.sync(transfers) }
    }

    /// A model that is on disk and would not load.
    ///
    /// Separate from a transfer's `.failed`, because they are separate
    /// failures with separate fixes: one means download it again, the other
    /// means this phone cannot hold it. They shared a dictionary before and
    /// the row said "download failed" about a download that had succeeded.
    var loadErrors: [String: String] = [:]

    var loadedModelID: String?
    var isLoadingModel = false
    /// Varies with the context limit: what fits depends on how much KV cache
    /// the served context will demand, so this is not a constant of the device.
    var budget: DeviceBudget

    /// The id currently being loaded, so the row can say so. `isLoadingModel`
    /// alone greyed out every Load button in the list with no indication of
    /// which one was working, or that anything was happening at all.
    var loadingModelID: String?

    /// Shown after an offload, because otherwise the row simply reverts to
    /// "Load" and nothing says the memory actually came back.
    var offloadNotice: String?

    /// A short history of (when, how many bytes) per download, used to quote a
    /// rate and a time remaining.
    ///
    /// Instantaneous rate — bytes since the last callback over the time since
    /// the last callback — is useless to read: it swings by an order of
    /// magnitude between callbacks and the ETA flickers between "2 minutes"
    /// and "40 minutes". Averaging over a window instead means the number
    /// moves slowly enough to be worth looking at, at the cost of lagging a
    /// genuine change in speed by a few seconds. That trade is the right way
    /// round for something a person is watching.
    ///
    /// Not observed: it changes on every progress report, and the screens that
    /// quote a rate redraw when `transfers` does, which is throttled — see
    /// `publishProgress`.
    @ObservationIgnored var downloadSamples: [String: [(at: Date, bytes: Int64)]] = [:]

    /// Release the model when the app is backgrounded. On by default: a
    /// suspended process holding gigabytes is the first thing the kernel
    /// reclaims, and the socket is closed then anyway, so nothing is lost.
    var autoOffloadInBackground = true {
        didSet { UserDefaults.standard.set(autoOffloadInBackground, forKey: Keys.autoOffloadInBackground) }
    }

    /// Seconds of quiet before the model is released, or 0 to keep it resident.
    ///
    /// This is `keep_alive`, and it exists here for the reason it exists in
    /// Ollama: a server is idle most of the time, and a phone has other things
    /// to do with two gigabytes. It defaults to off rather than to Ollama's
    /// five minutes because the reload is not free here — a desktop refills
    /// from page cache in about a second, a phone takes appreciably longer, and
    /// silently adding that to someone's first request after lunch is a worse
    /// surprise than the memory.
    var idleOffloadSeconds = 0 {
        didSet {
            UserDefaults.standard.set(idleOffloadSeconds, forKey: Keys.idleOffloadSeconds)
            restartIdleWatch()
        }
    }

    var idleWatch: Task<Void, Never>?
    var lastLocalActivity = Date()

    /// Why the model left memory, when it was not the user's doing.
    ///
    /// The distinction decides whether it comes back on its own. A model the
    /// user offloaded stays offloaded; one the app released to survive
    /// backgrounding, or to stop holding a gigabyte hostage while idle, is
    /// restored the moment it is wanted again.
    enum ReleaseReason: String, Sendable {
        case background
        case idle
    }

    var autoReleased: (id: String, reason: ReleaseReason)?

    /// Reasons the app is currently forbidden from auto-releasing.
    ///
    /// A set rather than a flag because the suppressions nest: the photo
    /// picker can be open while a download completes. Clearing a boolean at
    /// the end of one of those would re-arm release while the other was still
    /// in progress.
    var releaseSuppressions: Set<String> = []

    let store: ModelStore

    var downloadTasks: [String: Task<Void, Never>] = [:]
    /// One per finished transfer, counting down its completion notice.
    var noticeTimers: [String: Task<Void, Never>] = [:]

    /// The brain whose load failed last, and why in a sentence a person can
    /// act on, so the status can say so until a later load succeeds.
    /// `loadErrors` keeps every failure, in the engine's words, for the Brain
    /// page; this is the one that decides what everyone else is told.
    var lastLoadFailure: (id: String, sentence: String)?

    /// When each download last published its progress, and at what whole
    /// percentage. See `publishProgress`.
    @ObservationIgnored var lastPublishedProgress: [String: (at: Date, percent: Int)] = [:]

    // MARK: - Chat

    var conversation: [ChatMessage] = []
    var draft = ""
    /// Left by `AskLaunch` for `AskComposer` to focus (and, where supported, start listening on) once, then clear.
    var askComposerFocusRequested = false

    /// An ability the last message would have used, if it were switched on.
    ///
    /// This is the answer to the app's biggest product problem, which is not
    /// that the personal-data tools are off by default — that is correct, and
    /// defaulting HealthKit on would be wrong — but that being off has meant
    /// being invisible. A switch nobody finds is a feature nobody has.
    ///
    /// So the assistant says so at the only moment the offer is welcome: the
    /// turn where the user just asked something it could have answered. Not a
    /// message bubble and not an `AnswerCard` — the model did not say this, and
    /// cards are written into the saved transcript, where an offer would still
    /// be sitting months later next to an ability that has since been enabled.
    var pendingToolOffer: ToolOffer?

    var isGenerating = false

    /// The tool the reply in flight is running, or `nil`.
    ///
    /// A tool turn is two complete prefills the user cannot see — seconds of
    /// nothing that read as a hang, which is the whole reason
    /// `GenerationEvent.toolCallStarted` exists. Held here so the Ask screen can
    /// say "Checking your calendar…" instead of showing an orb that looks stuck.
    /// Cleared on the next token and again when the generation ends, so it can
    /// never outlive the thing it describes.
    var activeToolName: String?

    var systemPrompt: String = "You are a helpful assistant." {
        didSet { UserDefaults.standard.set(systemPrompt, forKey: Keys.systemPrompt) }
    }

    /// Shown in Chat when the resident model changed underneath the user.
    var modelSwitchNotice: String?
    /// Transport failures live here rather than in the transcript, so they
    /// are never sent back to the model as prior context.
    var generationError: String?

    /// Images waiting to go with the next message.
    ///
    /// Held here rather than in the view so that switching tabs mid-compose
    /// does not silently drop a photo someone has already chosen — the same
    /// reason the text draft lives here.
    var attachments: [Data] = []

    let conversations: ConversationStore
    /// Every saved conversation, newest first. Kept in memory so the history
    /// list is instant; the store is the source of truth on disk.
    var history: [Conversation] = []
    var currentConversationID = UUID()
    var conversationStartedAt = Date()

    /// Writes are debounced rather than done per token: a reply arrives at
    /// dozens of tokens a second and each one would otherwise re-encode and
    /// re-write the entire transcript.
    var saveTask: Task<Void, Never>?

    var lastCheckpoint = Date.distantPast

    var generationTask: Task<Void, Never>?

    /// The number of the reply being written, so that a stopped reply's last
    /// tokens — cancellation is cooperative — cannot land on the one after it.
    @ObservationIgnored var generationSerial = 0

    /// Where the reply being written goes, for the "Stopped." marker. See
    /// `stopGenerating()` and `snapshotConversation()`.
    @ObservationIgnored var inFlightTarget: GenerationTarget?

    /// A question that arrived before a brain could answer it — from Siri, a
    /// Shortcut, the Control Center button, or the composer — sent the moment
    /// one can. See `askWhenReady(_:)`.
    var waitingQuestion: String?

    /// Notices put away for good, by `AskNotice.Dismissal` key. Persisted:
    /// the old silencing lived in view state and came back on every visit.
    var dismissedNotices: Set<String> = Set(UserDefaults.standard.stringArray(forKey: Keys.dismissedNotices) ?? []) {
        didSet { UserDefaults.standard.set(Array(dismissedNotices).sorted(), forKey: Keys.dismissedNotices) }
    }

    // MARK: - Abilities: calendar, reminders, health

    /// Whether the calendar tool is registered with the engine.
    ///
    /// Off by default, and it has to be a decision rather than a nicety.
    /// Registering a tool does not wait to be useful: the library appends every
    /// schema plus a fixed instruction preamble to the system message of
    /// *every* prompt, so a question about pasta is charged for a calendar the
    /// model was never going to open. `LlamaEngine.promptOverheadTokens`
    /// measures that and `ContextGuard` reserves it, which on a 4K window is a
    /// few hundred tokens of conversation the user no longer has.
    ///
    /// Split from `reminderToolsEnabled`, which it used to share a switch with.
    /// Both halves of that were wrong: the cost above is charged per tool, so
    /// somebody who only wanted their week read paid for a shopping list they
    /// never asked about, and iOS asked them for both permissions to get one.
    var calendarToolsEnabled = false {
        didSet {
            guard calendarToolsEnabled != oldValue else { return }
            UserDefaults.standard.set(calendarToolsEnabled, forKey: Keys.calendarTools)
            Task { await applyPersonalDataTools() }
        }
    }

    /// Whether the reminders tool is registered with the engine.
    ///
    /// Its own switch, its own key and its own iOS grant; the schema cost that
    /// makes this a decision rather than a default is described on
    /// `calendarToolsEnabled` and is charged separately here.
    var reminderToolsEnabled = false {
        didSet {
            guard reminderToolsEnabled != oldValue else { return }
            UserDefaults.standard.set(reminderToolsEnabled, forKey: Keys.reminderTools)
            Task { await applyPersonalDataTools() }
        }
    }

    /// Health is a separate switch from the calendar, deliberately.
    ///
    /// It is a separate grant, a separate iOS sheet, and its own schema cost on
    /// every prompt — and someone who is relaxed about the app reading their
    /// week is often not relaxed about it reading their heart.
    var healthToolsEnabled = false {
        didSet {
            guard healthToolsEnabled != oldValue else { return }
            UserDefaults.standard.set(healthToolsEnabled, forKey: Keys.healthTools)
            Task { await applyPersonalDataTools() }
        }
    }

    /// What the engine did with the switches, and why.
    ///
    /// Settings needs this because a switch is one thing and the effect is
    /// another: the switches above say what the user asked for, and this says
    /// what the model currently in memory will actually be given. The two
    /// disagreed silently before, which is how a user ends up with a switch
    /// that is on, a calendar that is never read, and no way to find out why.
    var personalDataToolGate: ToolGate.Decision = .noModelLoaded

    /// What the budget did with what the gate allowed.
    ///
    /// The second half of the same answer, and it was the half nobody could
    /// see. The gate decides whether this model gets tools at all; the context
    /// window then decides how many of them fit, and at a 1,024-token limit it
    /// quietly keeps the calendar and refuses reminders — while the only
    /// sentence on the screen was the gate's, which says both are registered.
    /// The plan holds the honest version, priced, and `CapabilityNotice` is
    /// where the two are stopped from contradicting each other.
    var capabilityPlan: CapabilityBudget.Plan?

    /// What iOS said when the switch was flipped — never a claim about whether
    /// a read was granted. HealthKit does not report that, and any UI implying
    /// otherwise is a lie the app cannot substantiate.
    ///
    /// Shown under the switch through `HealthAvailability.explanation`, because
    /// the cases that are not `.available` are the ones where the switch is on
    /// and nothing will ever be read: `registrations` skips the tool outright on
    /// a device with no Health store, so without a sentence here the user is
    /// left with a switch that is on, a heart rate that is never read, and no
    /// way to find out why — the exact failure `personalDataToolGate` exists to
    /// end, one switch further down.
    var healthAvailability: HealthAvailability = .available

    /// What iOS says about each entity right now, for Settings to show.
    ///
    /// Kept rather than read on demand because `EKEventStore.authorizationStatus`
    /// is a synchronous TCC lookup that a SwiftUI body would run on every
    /// redraw, and because the interesting transitions — the prompt being
    /// answered, the user coming back from iOS Settings — are events, not
    /// polling.
    var personalDataAuthorization: [PersonalDataEntity: PersonalDataAuthorization] = [:]

    /// Trips through the connect door that are still waiting on iOS, so a
    /// second tap shares the first one's answer. See `connect(_:from:)`.
    @ObservationIgnored var connectTrips: [Ability: Task<ConnectOutcome, Never>] = [:]

    // MARK: - Routines

    let tasks: ScheduledTaskStore

    /// Runs what is due. Never nil after `init`; optional only because its
    /// `Host` closures read this object, and Swift will not let them be
    /// captured until every stored property has a value.
    var scheduleRunner: ScheduleRunner?

    /// Every scheduled task, newest first. Kept in memory the way `history` is,
    /// so a list is instant; the store is the source of truth on disk.
    ///
    /// Refreshed after every sweep, because a sweep is the one thing that
    /// changes a task without anybody touching the screen — that is the whole
    /// point of it — and a schedule view reading a snapshot taken at launch
    /// would show this morning's briefing as never having run.
    var scheduledTasks: [ScheduledTask] = []

    /// One sweep at a time, in flight.
    ///
    /// Held rather than fired and forgotten because everything that tears the
    /// llama context down has to be able to stop it: a scheduled generation
    /// that outlives a model load is decoding against weights that are being
    /// unmapped underneath it.
    var scheduleSweep: Task<Void, Never>?

    /// The repeating sweep Desk Mode runs on.
    ///
    /// Managed exactly like `idleWatch`: cancelled and rebuilt from one place,
    /// so there is never a second loop nobody is holding a handle to.
    var deskSweep: Task<Void, Never>?

    // MARK: - Usage counts and the review prompt

    /// Whether anonymous usage counts may be sent, as decided by the person
    /// holding the phone.
    ///
    /// Starts `.undecided` and stays there until someone is actually shown the
    /// sentence explaining it, which happens on the third onboarding screen.
    /// Nothing is sent while undecided — so a user who taps Skip before
    /// reaching that screen produces no analytics at all, which is the correct
    /// outcome even though it costs us the data.
    var analyticsConsent: AnalyticsConsent = {
        let stored = UserDefaults.standard.string(forKey: Keys.analyticsConsent)
        // Absent means on. Usage events are on by default and stay on unless
        // someone goes and turns them off, so "never answered" has to resolve
        // to the default rather than to silence — an absent key used to mean
        // .undecided, which sent nothing, and would now mean a fresh install
        // reports nothing at all.
        return stored.flatMap(AnalyticsConsent.init(rawValue:)) ?? .granted
    }()

    /// Where events go.
    ///
    /// Constructed here rather than injected because there is nothing left to
    /// choose between: this is the app, and usage counts are part of it rather
    /// than a setting inside it. `NoAnalytics` is still what every test sees,
    /// and it is still a type with no code capable of transmitting rather than
    /// a flag that says not to — which is why the package's suite cannot reach
    /// a real project even by accident.
    private(set) var analytics: any AnalyticsSink = AnalyticsForBuild.make()

    /// Set when the app has just done its job and has earned the question.
    ///
    /// A flag rather than a call, because asking is a SwiftUI environment
    /// action that only a view can perform. `RootView` watches this and clears
    /// it through `reviewPromptShown()`.
    var wantsReviewPrompt = false

    // MARK: - The person, the brief, first run

    /// What to call them, and nothing else. See `Profile`.
    ///
    /// A `let` holding its own observable object rather than three properties
    /// here: a view that only greets somebody then observes one small thing and
    /// does not redraw on every byte a download reports.
    @ObservationIgnored let profile = Profile()

    /// The daily brief and the day it is about. See `AppModel+Briefing.swift`,
    /// which is the whole of the engine — a Swift extension cannot add a stored
    /// property, so this one line is the engine's entire footprint here.
    private(set) var briefing = BriefingState()

    /// Whether the intro has been seen.
    ///
    /// Read once at launch rather than observed: flipping it mid-session is
    /// what dismisses the intro, and a stored value that changed underneath
    /// would put it back.
    private(set) var hasCompletedOnboarding = UserDefaults.standard.bool(forKey: Keys.completedOnboarding)

    /// Watches the calendar database for changes made anywhere — another app,
    /// a sync, a tool's own write — so Today reads again rather than waiting
    /// for its reading to age. See `watchCalendarDatabase()`.
    @ObservationIgnored var calendarDatabaseObserver: (any NSObjectProtocol)?

    // MARK: - Plumbing

    let configurationStore: ServerConfigurationStore

    /// Module-internal rather than private only so that `AppModel+Briefing`,
    /// which is an extension in another file, can have the brief written by the
    /// same resident model the chat uses. Nothing outside this module sees it.
    let engine: LlamaEngine

    let server: InferenceServer

    private var observers: [Task<Void, Never>] = []
    let bonjour = BonjourAdvertiser()
    var governor: DeviceGovernor?

    /// When the listener came up, for the served seconds `serverStopped`
    /// carries. Nil whenever this app is not the one serving.
    ///
    /// Deliberately not "when the app was opened". iOS closes the socket on
    /// suspend without telling anyone, so the two clocks differ by every minute
    /// the phone spent in a pocket, and only one of them is about the product.
    var servingSince: Date?

    enum Keys {
        static let loadedModel = "pocketd.loadedModel"
        static let systemPrompt = "pocketd.systemPrompt"
        static let reviewState = "pocketd.reviewState"
        static let serverShouldRun = "pocketd.serverShouldRun"
        static let autoOffloadInBackground = "pocketd.autoOffloadInBackground"
        static let idleOffloadSeconds = "pocketd.idleOffloadSeconds"
        static let calendarTools = "pocketd.calendarTools"
        static let reminderTools = "pocketd.reminderTools"
        /// The switch that governed both of the two above. Read once, at the
        /// first launch of a build that has them, and removed on the spot —
        /// nothing writes it any more. See `PersonalDataToolsMigration`.
        static let legacyPersonalDataTools = "pocketd.personalDataTools"
        static let healthTools = "pocketd.healthTools"
        static let completedOnboarding = "pocketd.completedOnboarding"
        static let analyticsConsent = "pocketd.analyticsConsent"
        static let dismissedNotices = "pd.ask.dismissedNotices"
    }

    // MARK: - Lifecycle

    init() {
        // The configuration comes first because the memory budget depends on
        // it: how much KV cache a model needs is set by the context this
        // server creates, so "will it fit" cannot be answered until the served
        // context is known.
        //
        // load() persists a freshly generated configuration on the spot.
        // Relying on the didSet below would not: property observers do not run
        // during initialisation, so the API key would be new on every launch.
        let configurationStore = ServerConfigurationStore()
        self.configurationStore = configurationStore
        let configuration = configurationStore.load()
        self.configuration = configuration

        let budget = DeviceBudget.current(
            hasIncreasedMemoryLimit: Entitlements.hasIncreasedMemoryLimit,
            servedContextTokens: configuration.maxContextTokens
        )
        self.budget = budget

        let directory = (try? ModelStore.defaultDirectory())
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("Models")
        let store = ModelStore(directory: directory, budget: budget)
        self.store = store

        self.conversations = ConversationStore(
            directory: (try? ConversationStore.defaultDirectory())
                ?? directory.appendingPathComponent("Conversations")
        )

        // The process's one task store, shared with the background refresh
        // rather than built again here. Two stores over one folder made the
        // actor's read-modify-write last-writer-wins again: a background
        // refresh settling a firing while the app saved an edit lost one of
        // them. See `BackgroundWake.store`, which also creates the folder and
        // keeps it out of iCloud Backup.
        self.tasks = BackgroundWake.store

        // Read before the engine exists rather than from the stored properties,
        // because property observers do not run during initialisation: reading
        // `calendarToolsEnabled` here would see its declared default and launch
        // with the tools off for someone who turned them on.
        //
        // `object(forKey:)` and not `bool(forKey:)` for the three that feed the
        // migration: it needs "never written" and `false` to be different
        // answers, and `bool(forKey:)` collapses them into one. Health is not
        // in the migration and keeps the simpler read.
        let defaults = UserDefaults.standard
        let personalData = PersonalDataToolsMigration.resolve(
            .init(
                legacy: defaults.object(forKey: Keys.legacyPersonalDataTools) as? Bool,
                calendar: defaults.object(forKey: Keys.calendarTools) as? Bool,
                reminders: defaults.object(forKey: Keys.reminderTools) as? Bool
            )
        )
        if personalData.shouldPersist {
            // The removal last. Either write failing on its own leaves the old
            // key as the answer, which is recoverable; losing it first leaves
            // an upgraded user with no answer at all.
            defaults.set(personalData.calendar, forKey: Keys.calendarTools)
            defaults.set(personalData.reminders, forKey: Keys.reminderTools)
            defaults.removeObject(forKey: Keys.legacyPersonalDataTools)
        }
        let healthEnabled = defaults.bool(forKey: Keys.healthTools)
        let engine = LlamaEngine(
            fileURL: { store.fileURL(for: $0) },
            projectorURL: { store.projectorURL(for: $0) },
            tools: Self.registrations(
                calendar: personalData.calendar,
                reminders: personalData.reminders,
                health: healthEnabled
            ),
            exempt: Self.selfTestTools()
        )
        self.engine = engine
        // The phone's brain, which is Apple's model or this same LlamaEngine.
        // The server below is built on `engine` alone, so other devices are
        // only ever served a downloaded brain.
        let phoneBrain = PhoneBrain(downloaded: engine)
        self.phoneBrain = phoneBrain
        self.brainChoice = BrainChoice.load()
        self.builtInAvailability = Self.rememberedBuiltInAvailability()

        self.server = InferenceServer(
            configuration: configuration,
            engine: engine,
            models: { await store.installed() },
            puller: { record in
                AsyncThrowingStream { continuation in
                    let task = Task {
                        do {
                            for try await progress in await store.download(record) {
                                continuation.yield(progress)
                            }
                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                    continuation.onTermination = { _ in task.cancel() }
                }
            },
            deviceBudget: budget,
            // Speech to text for other devices, with the same model the
            // microphone uses once one is downloaded.
            transcriber: ParakeetTranscriber.shared
        )

        systemPrompt = UserDefaults.standard.string(forKey: Keys.systemPrompt) ?? systemPrompt
        serverShouldRun = UserDefaults.standard.bool(forKey: Keys.serverShouldRun)
        // `object(forKey:)` rather than `bool(forKey:)`: an absent key reads as
        // false, which would turn the default off for everyone who has never
        // touched the switch.
        autoOffloadInBackground = UserDefaults.standard.object(forKey: Keys.autoOffloadInBackground) as? Bool ?? true
        idleOffloadSeconds = UserDefaults.standard.integer(forKey: Keys.idleOffloadSeconds)
        calendarToolsEnabled = personalData.calendar
        reminderToolsEnabled = personalData.reminders
        healthToolsEnabled = healthEnabled

        // Last, because it is the only thing here that needs `self`: the four
        // answers in `Host` are all reads of this object, and Swift only lets
        // them be captured once every stored property has a value.
        scheduleRunner = ScheduleRunner(
            store: tasks,
            // The phone's brain, not the server's: a routine is the phone
            // thinking for its owner, so it uses whichever brain they chose.
            engine: phoneBrain,
            // Through `authorization(for:)` first, and never straight into
            // `requestReadAccess`. That call is what puts iOS's modal on
            // screen, and a sweep happens with nobody expecting one — on the
            // return to the foreground, or once a minute on a phone lying
            // face-down on a charger in Desk Mode. A permission alert nobody
            // asked for, with no visible connection to anything, is not a
            // decision the user gets to make properly, and it is the same
            // reasoning `applyPersonalDataTools` gives for asking at the switch
            // rather than at first tool use.
            //
            // The Abilities switches are deliberately NOT consulted. Those
            // decide what the *model* is handed on every prompt and what that
            // costs in context; a watcher is the user's own standing
            // instruction, written into a task they created, and iOS's grant is
            // the thing that says whether it may be honoured.
            readEvents: { window in
                let authorization = EventAccess.authorization(for: .calendar)
                guard authorization.canRead else { return .unauthorized(authorization) }
                return await EventAccess.shared.events(in: window)
            },
            readReminders: { window in
                let authorization = EventAccess.authorization(for: .reminders)
                guard authorization.canRead else { return .unauthorized(authorization) }
                return await EventAccess.shared.reminders(in: window)
            },
            host: ScheduleRunner.Host(
                isUserGenerating: { [weak self] in
                    // Fails closed. A model that has gone is not a model that
                    // is idle, and starting an unattended generation against
                    // whatever is left is the wrong way to be wrong.
                    self?.isGenerating ?? true
                },
                systemPrompt: { [weak self] in self?.systemPrompt ?? "" },
                maxTokens: { [weak self] in self?.configuration.maxContextTokens },
                record: { [weak self] event in self?.record(event) }
            )
        )

        // Before the first frame, which is drawn before `bootstrap` runs: the
        // tiles start from today's last reading and the permissions as they
        // stand, not from a placeholder that says everything is off.
        primeTodayFromDisk()
        profile.onNameChange = { [weak self] in self?.nameDidChange() }
        // Here rather than in `bootstrap`, because a Siri question can arrive
        // before the scene's first task has run.
        IntentInbox.onDeliver = { [weak self] question in self?.questionArrived(question) }
    }

    func bootstrap() async {
        #if DEBUG
        await DemoSeed.runIfRequested()
        // The seeder writes defaults that this object read at init, so on the
        // first seeded launch the intro still showed and the greeting had no
        // name. Re-read the two it wrote, and only when it ran.
        if DemoSeed.isRequested || ProcessInfo.processInfo.arguments.contains("-pocketd.skipIntro") {
            hasCompletedOnboarding = UserDefaults.standard.bool(forKey: Keys.completedOnboarding)
            if let seeded = UserDefaults.standard.string(forKey: Profile.nameKey), !seeded.isEmpty {
                profile.name = seeded
            }
            if UserDefaults.standard.bool(forKey: Keys.calendarTools) { calendarToolsEnabled = true }
            if UserDefaults.standard.bool(forKey: Keys.reminderTools) { reminderToolsEnabled = true }
        }
        #endif

        // Before anything else that could fail: the review gate needs to know
        // how long this install has existed, and an app that crashes on launch
        // is exactly the one that must not ask for a rating tomorrow.
        noteLaunchForReview()
        let governor = DeviceGovernor(
            batteryFloor: configuration.pauseBelowBatteryLevel,
            tolerance: configuration.thermalTolerance
        ) { [weak self] condition in
            guard let self else { return }
            self.condition = condition
            Task { await self.server.setCondition(condition) }
        }
        self.governor = governor
        governor.start()

        await store.load()
        await engine.updateDefaultSampling(configuration.sampling)
        installed = await store.installed()
        settleOnboardingForExistingInstall()

        // After the onboarding flag has been settled, not before. That flag is
        // the trace an earlier build leaves behind, and it is the only thing
        // separating a genuinely new install from someone updating into the
        // first version that had any analytics in it. Recorded a line earlier,
        // every existing user reports a first launch on release day.
        analytics.record(.appOpened(
            isFirstLaunch: FirstLaunch.claim(evidenceOfPriorUse: hasCompletedOnboarding)
        ))

        // Reopen what was on screen last time. Coming back to a blank Chat tab
        // after a force-quit reads as data loss even when the transcript is
        // safely on disk one tap away in the history.
        await loadHistory()
        if let latest = history.first {
            conversation = latest.messages
            draft = latest.draft
            currentConversationID = latest.id
            conversationStartedAt = latest.createdAt
        }

        // Driving the idle timer from the observer is what also catches the
        // transitions the app did not initiate — a failed bind, or iOS
        // reclaiming the socket.
        observers.append(Task { [server] in
            for await state in await server.stateStream() {
                await MainActor.run {
                    self.serverState = state
                    self.applyIdleTimer(running: state.isRunning)
                }
            }
        })
        observers.append(Task { [server] in
            for await snapshot in await server.pairing.stream() {
                await MainActor.run { self.pairing = snapshot }
            }
        })
        observers.append(Task { [server, engine, phoneBrain] in
            for await entries in await server.log.stream() {
                // The server loads models behind the app's back when a request
                // names one that is not resident, so the badge and the Chat tab
                // would otherwise keep pointing at a model that is gone.
                //
                // The phone's brain from `phoneBrain`, not `engine`: while the
                // phone uses Apple's model, a laptop loading a download must
                // not turn "Built into iPhone" into that download's name.
                let resident = await phoneBrain.loadedModel()?.id
                let served = await engine.loadedModel()?.id
                await MainActor.run {
                    self.log = entries
                    // Not user-initiated: this fires because some client asked
                    // for a different model.
                    self.syncLoadedModel(resident)
                    self.downloadedResidentID = served
                }
            }
        })

        #if DEBUG
        // Picks Apple's model before the Brain page can, for a simulator run
        // that needs it chosen without tapping (A3's verification).
        if ProcessInfo.processInfo.arguments.contains("-pocketd.useBuiltInBrain") {
            brainChoice = .builtIn
            BrainChoice.builtIn.save()
        }
        #endif
        // Which brain answers the phone: the one chosen, or — on a fresh
        // install — Apple's model when this phone has it, else a download.
        // See `applyEffectiveBrain`, which also keeps the old fallback: a
        // remembered download that will not load gives way to the next one.
        await phoneBrain.builtIn.updateAbilities(
            calendar: calendarToolsEnabled,
            reminders: reminderToolsEnabled,
            health: healthToolsEnabled
        )
        // A chosen download that is on the phone is put in charge without
        // asking Apple's model anything: that answer cannot change the
        // outcome, and its first read took seconds on the simulator. Every
        // other launch needs it first, a fresh install above all — that is
        // how Apple's model becomes the default where the phone has it.
        if case let .downloaded(id)? = brainChoice, installed.contains(where: { $0.id == id }) {
            await applyEffectiveBrain()
            Task { await refreshBuiltInAvailability() }
        } else {
            await refreshBuiltInAvailability()
            await applyEffectiveBrain()
        }
        if serverShouldRun {
            await startServer()
        }

        await loadScheduledTasks()
        #if DEBUG
        await seedDemoRoutinesIfRequested()
        #endif
        // Swept here rather than left to the scene phase, because there is no
        // scene-phase change to hang a cold launch on: SwiftUI's
        // `onChange(of:)` does not fire for the value a scene already has, so an
        // app launched straight into `.active` never reaches
        // `handleScenePhase`. Without this line a task that came due while the
        // app was closed waits for the user to background it and come back.
        //
        // `.foreground` unconditionally: `deskMode` is deliberately not
        // persisted, so it is false at this point on every launch.
        sweepSchedule(in: .foreground)

        // Last, and not awaited. The Today tab is what the app opens to, and
        // it must not wait behind a calendar daemon: the brief publishes a
        // template the moment EventKit answers and upgrades itself if a model
        // turned out to be resident. Reads only — nothing here prompts.
        watchCalendarDatabase()
        // Through the gate: the Today tab's own refresh has usually read the
        // day already, and anything since then that changed what may be read
        // has marked it stale itself.
        Task { await refreshBrief() }
    }

    /// Scene-phase handling, and the reason it is not a two-state check.
    ///
    /// `.inactive` is not "leaving". It fires for a notification banner, the
    /// app switcher, Control Centre, an incoming call sheet, and every system
    /// permission prompt. Unloading a multi-gigabyte model there would mean a
    /// notification arriving mid-conversation costs a full reload — so
    /// `.inactive` is deliberately ignored and only `.background` releases.
    func handleScenePhase(_ phase: ScenePhase) async {
        switch phase {
        case .background:
            await releaseForBackground()
        case .active:
            await reconcileAfterForeground()
            await restoreAutoReleasedModel()
            // Apple Intelligence may have been switched on in Settings, or
            // Apple's model may have finished arriving, while the app was away
            // — and a question may be waiting for exactly that.
            if await refreshBuiltInAvailability() { await applyEffectiveBrain() }
            // After the model is back, and not before: a prompt task swept
            // first would find nothing resident and file a `.noModelLoaded` run
            // against a phone that had the weights on disk the whole time.
            //
            // Desk Mode wins wherever it is on. It is the stronger context —
            // the same GPU access, and nobody watching — and sweeping
            // `.foreground` here as well would run one firing twice under two
            // different names.
            if deskMode {
                restartDeskSweep()
            } else {
                sweepSchedule(in: .foreground)
            }
            // The day has moved while the phone was in a pocket: a meeting has
            // started, a reminder has come due, last night has been recorded —
            // and a permission may have changed in iOS Settings. Read again
            // whatever the age of the last reading. Not awaited, for the
            // reason `bootstrap` gives.
            refreshPersonalDataAuthorization()
            dayMayHaveChanged()
        case .inactive:
            break
        @unknown default:
            break
        }
    }

    // MARK: - First run

    var needsOnboarding: Bool { !hasCompletedOnboarding }

    /// Settles the flag for an install that predates the intro existing.
    ///
    /// Called once at launch. Someone updating into this build already knows
    /// what the app is, so they are marked done rather than shown it — but the
    /// test has to be *here*, not folded into `needsOnboarding`, because
    /// otherwise "Show the introduction again" would be dead for exactly the
    /// people who have used the app enough to want it. That is the shape of
    /// bug where a button exists, does nothing, and nobody notices for months.
    private func settleOnboardingForExistingInstall() {
        guard !hasCompletedOnboarding, !installed.isEmpty else { return }
        completeOnboarding()
    }

    func completeOnboarding() {
        hasCompletedOnboarding = true
        UserDefaults.standard.set(true, forKey: Keys.completedOnboarding)
        // The brief written during the intro was written before a name was
        // given or a calendar connected — "Good morning, I hope you're doing
        // well." — and it stayed on Today until somebody pulled to refresh.
        // Written again now, with everything the intro collected.
        Task { await refreshBrief(force: true) }
    }

    /// Replays the intro from Settings. Deliberately does not undo anything
    /// else — this is "show me that again", not "reset the app".
    func replayOnboarding() {
        hasCompletedOnboarding = false
        UserDefaults.standard.set(false, forKey: Keys.completedOnboarding)
    }

    // MARK: - A3 state: the built-in brain

    /// The engine the phone's own questions go to — the conversation, the
    /// brief, routines — which forwards to whichever brain is in use. `engine`
    /// stays the server's, always a downloaded brain. See `PhoneBrain`.
    let phoneBrain: PhoneBrain

    /// What the user chose, as they chose it; `nil` until they have. Read
    /// `effectiveBrain` for what is answering today. See `BrainChoice`.
    var brainChoice: BrainChoice?

    /// Whether Apple's model can answer on this phone. Read off the main
    /// thread at launch and on every return to the app — the first read on
    /// the simulator took about three seconds — and remembered between
    /// launches so the first frame is not wrong.
    var builtInAvailability: BuiltInBrainAvailability = .notEligible

    /// The downloaded brain in memory, whoever it is serving — the phone, or
    /// only other devices while the phone uses Apple's model. `loadedModelID`
    /// is the phone's brain; this is the server's.
    var downloadedResidentID: String?

    // MARK: - B2 state: the conversation

    /// What the reader can do about the reply that just failed — the one
    /// action beside `generationError`'s sentence. Set with it, cleared with
    /// it. See `ReplyRemedy`.
    var replyFailureRemedy: ReplyRemedy?

    /// A question about the day that was not sent, because the brain in use
    /// cannot read the day. See `DayRefusal`.
    var dayRefusal: DayRefusal?

    // MARK: - B5 state: the intro

    /// Whether the intro had been finished when this launch began. Read once,
    /// here, because `replayOnboarding()` clears the flag it comes from — and
    /// it is what tells a replay from You, which may be closed, from a first
    /// run, which has no way out but through. See `introIsReplay`.
    let introFinishedAtLaunch = UserDefaults.standard.bool(forKey: Keys.completedOnboarding)

    /// Whether the intro was finished during this launch, for a replay started
    /// before the app has been relaunched once.
    var introFinishedThisLaunch = false
}
