import Foundation

// MARK: - What the model may ask for

/// The verbs the reminders tool accepts.
///
/// `list` is the old `get_reminders` and does exactly what it always did. The
/// other two are why this tool stopped being called `get_`.
///
/// One tool with a verb rather than three tools, and the reason is measured
/// rather than aesthetic. Tool schemas are frozen into the client at model load
/// — rebuilding the set costs a reload of the weights, which is why the
/// per-question router in `CapabilityBudget` has never had a caller — so every
/// tool is charged against every prompt for as long as the model is resident.
/// Against the default 4,096-token window, a third of which may go on schemas,
/// and a tool-native template like Qwen3's that renders them twice, three read
/// tools cost 785 tokens of a 1,344 ceiling and three more write tools take the
/// total to 1,469. That does not fit, and the budget's response to not fitting
/// is to drop tools by priority — so the writes would have been registered,
/// charged for, and then silently discarded on the exact default configuration
/// most people run. Folding the verb in costs 581 characters where three
/// separate tools cost 948.
public enum ReminderAction: String, Sendable, Codable, CaseIterable {
    case list
    case create
    case complete
}

/// The verbs the calendar tool accepts. No `delete`, and no `complete` because
/// an event is not a to-do.
public enum CalendarAction: String, Sendable, Codable, CaseIterable {
    case list
    case create
}

/// A reminder on its way into EventKit.
public struct NewReminder: Sendable, Equatable {
    public var title: String
    public var due: Date?
    /// False when the phrase named a day and no clock time. `EKReminder` models
    /// this natively and turning it into midnight would invent a deadline.
    public var dueHasTime: Bool
    /// How often it comes back, when it does. Always `nil` where `due` is
    /// `nil`: a pattern with no first occurrence has nothing to repeat from.
    public var repeats: ReminderRepeat?

    public init(
        title: String,
        due: Date? = nil,
        dueHasTime: Bool = false,
        repeats: ReminderRepeat? = nil
    ) {
        self.title = title
        self.due = due
        self.dueHasTime = dueHasTime
        self.repeats = repeats
    }
}

/// An event on its way into EventKit.
public struct NewEvent: Sendable, Equatable {
    public var title: String
    public var start: Date
    public var end: Date
    public var repeats: ReminderRepeat?
    /// Free text for where it is. Optional and last, because nothing that
    /// reaches this type through chat sets it today — the assistant's own
    /// `create` never asks "and where" — but `CalendarEventRow` already reads
    /// a location back out, and the write side needs the same field to exist
    /// for anything to put one there at all.
    public var location: String?

    public init(title: String, start: Date, end: Date, repeats: ReminderRepeat? = nil, location: String? = nil) {
        self.title = title
        self.start = start
        self.end = end
        self.repeats = repeats
        self.location = location
    }
}

/// How a write ended. Never a thrown error — see `PersonalDataLookup`.
public enum WriteResult: Sendable, Equatable {
    case written
    case unauthorised(PersonalDataAuthorization)
    case failed
}

// MARK: - The decisions

/// Everything the write half of the tools does, minus EventKit.
///
/// Same split as `PersonalDataTools`, for the same reason: the parts worth
/// being sure about — that a network caller causes no write at all, that a
/// repeating phrase is refused rather than flattened, that the same reminder
/// asked for twice is filed once — need neither a device nor a granted
/// permission to exercise.
public enum PersonalDataWrites {

    /// How long an event lasts when nobody said.
    ///
    /// An hour, because that is what every calendar app defaults to and because
    /// the alternative — refusing until the model supplies a duration — turns
    /// the commonest request ("put lunch with Sam in for Friday at 1") into a
    /// round trip. "Nobody said" is decided from the person's words, not from
    /// whether the model filled the field: see `EventLength`. The length is
    /// echoed back in the confirmation whatever it is, so a user who wanted
    /// ninety minutes can see that they did not get them.
    public static let defaultEventMinutes = 60

    /// The longest an event may be asked to run, in minutes.
    ///
    /// Guards against a model reading "a couple of days" as 2 and then as
    /// minutes, or emitting a duration with an extra digit. A week is well past
    /// anything anyone types and well short of the year-long event that a
    /// misplaced zero produces.
    public static let maximumEventMinutes = 60 * 24 * 7

    // MARK: Reminders

    /// - Parameter existing: The open reminders, for the duplicate check. Read
    ///   before writing and never otherwise — a create that skipped this would
    ///   still be correct, just noisier.
    /// - Parameter write: The actual EventKit save.
    public static func createReminder(
        title rawTitle: String?,
        when: String?,
        now: Date = Date(),
        calendar: Calendar = .current,
        origin: RequestOrigin = ToolContext.origin,
        existing: @Sendable () async -> PersonalDataLookup<ReminderRow>,
        write: @Sendable (NewReminder) async -> WriteResult
    ) async -> [String: any Sendable] {
        // First, before anything is resolved and long before EventKit is
        // touched: a caller that may not change things must cause no change.
        guard origin.mayWritePersonalData else {
            return ["text": ToolContext.writeRefusal(for: origin)]
        }
        guard let title = usableTitle(rawTitle) else {
            return ["text": "I need to know what the reminder should say."]
        }

        var due: Date?
        var dueHasTime = false
        var repeats: ReminderRepeat?
        if let phrase = when, !phrase.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            switch MomentPhrase.resolve(phrase, now: now, calendar: calendar) {
            case let .moment(date, hasTime):
                due = date
                dueHasTime = hasTime
            case let .repeating(pattern, first, hasTime):
                due = first
                dueHasTime = hasTime
                repeats = pattern
            case .recurring:
                // A cue with no readable pattern ("every so often") or a
                // pattern with no hour ("every day"). Still refused, because
                // filing one reminder for something asked for daily is the
                // failure the user finds out about on the second morning.
                return ["text": "I can set a reminder that repeats, but I need to know how often and at what time — like \"every day at 7am\"."]
            case .unresolved:
                // No guessed hour, ever. A reminder at a time nobody asked for
                // is worse than none, because the user stops holding it in
                // their head.
                return ["text": "I could not work out when \"\(phrase)\" is. Tell me a time like \"2am today\" or \"tomorrow at 3pm\"."]
            }
        }

        // A reminder set in the past cannot fire, and somebody asking to be
        // reminded is asking for something in the future.
        //
        // This is not a hypothetical. Asked "remind me to take the bins out at
        // 2am" at ten past three in the afternoon, qwen3-1.7b filled `when`
        // with "2am today" — and `MomentPhrase` honours a named day even when
        // it is behind, on the reasoning that a person who names a day is owed
        // the day they named. That reasoning holds for a day the USER named.
        // The model added this one, and the result was an alarm thirteen hours
        // in the past that would never go off.
        //
        // Only a time of day that has already gone today is moved. An
        // explicitly past DATE — "remind me on the 15th", said on the 17th — is
        // left alone, because rolling that to next month would invent an
        // intention nobody had.
        // A day with no time has no time to have passed: "today" is due today,
        // not tomorrow because its midnight is behind us.
        let rolled = dueHasTime ? rollPastTimeForward(due, now: now, calendar: calendar) : (due: due, moved: false)
        due = rolled.due
        let movedForward = rolled.moved

        let candidate = NewReminder(title: title, due: due, dueHasTime: dueHasTime, repeats: repeats)

        if case let .rows(rows, _) = await existing(), let clash = duplicate(of: candidate, in: rows, calendar: calendar) {
            // Not an error and not a second reminder. The case this is for is a
            // model that calls the tool, does not recognise the result, and
            // calls it again — and the user finding the same thing twice in
            // their list. Reported as done because from where the user is
            // standing it is.
            return [
                "text": "That reminder is already on the list\(clash.due.map { " for \(describe($0, hasTime: clash.dueHasTime, calendar: calendar))" } ?? "").",
                "already_existed": true
            ]
        }

        switch await write(candidate) {
        case .written:
            // The resolved time is echoed back deliberately. No phrase parser
            // is right every time, and the cheapest defence against a wrong one
            // is for the answer to state what it actually set, so the user
            // catches it in the same breath.
            return [
                "text": confirmation(for: candidate, calendar: calendar, movedForward: movedForward),
                "created": true
            ]
        case let .unauthorised(authorization):
            return ["text": authorization.explanation(for: .reminders)]
        case .failed:
            return ["text": "I could not save that reminder."]
        }
    }

    /// Marks one open reminder done.
    ///
    /// Completing and not deleting, and that is the line for the whole feature.
    /// Ticking something off is recoverable — the reminder is still there, and
    /// the user can un-tick it — where deleting is not. An assistant driven by
    /// a 1.7B model, reading calendar titles a stranger wrote, should not be
    /// able to destroy anything it cannot put back.
    public static func completeReminder(
        title rawTitle: String?,
        origin: RequestOrigin = ToolContext.origin,
        existing: @Sendable () async -> PersonalDataLookup<ReminderRow>,
        complete: @Sendable (ReminderRow) async -> WriteResult
    ) async -> [String: any Sendable] {
        guard origin.mayWritePersonalData else {
            return ["text": ToolContext.writeRefusal(for: origin)]
        }
        guard let title = usableTitle(rawTitle) else {
            return ["text": "I need to know which reminder to tick off."]
        }

        let rows: [ReminderRow]
        switch await existing() {
        case let .rows(found, _): rows = found
        case let .unauthorized(authorization): return ["text": authorization.explanation(for: .reminders)]
        }

        let matches = rows.filter { Self.matches(title, $0.title.attackerControlledValue()) }
        guard !matches.isEmpty else {
            return ["text": "I could not find an open reminder matching \"\(title)\"."]
        }
        guard matches.count == 1, let target = matches.first else {
            // Never a guess. Picking one of three reminders that all match is
            // the kind of wrong that is only discovered later, by which time
            // the user has thrown away the thing they still needed.
            return ["text": "\(matches.count) open reminders match \"\(title)\". Which one?"]
        }

        switch await complete(target) {
        case .written:
            return ["text": "Ticked off \"\(target.title.sanitisedForPrompt())\".", "completed": true]
        case let .unauthorised(authorization):
            return ["text": authorization.explanation(for: .reminders)]
        case .failed:
            return ["text": "I could not tick that off."]
        }
    }

    // MARK: Calendar

    /// - Parameter durationMinutes: The model's length, believed only when
    ///   `userText` says how long — see `EventLength.decide`.
    /// - Parameter userText: The person's latest message, from the task local
    ///   both engines bind. `nil` when there is none to check against.
    public static func createEvent(
        title rawTitle: String?,
        start when: String?,
        durationMinutes: Int?,
        userText: String? = ToolContext.userText,
        now: Date = Date(),
        calendar: Calendar = .current,
        origin: RequestOrigin = ToolContext.origin,
        write: @Sendable (NewEvent) async -> WriteResult
    ) async -> [String: any Sendable] {
        guard origin.mayWritePersonalData else {
            return ["text": ToolContext.writeRefusal(for: origin)]
        }
        guard let title = usableTitle(rawTitle) else {
            return ["text": "I need to know what the event is called."]
        }
        guard let phrase = when, !phrase.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            // Unlike a reminder, an event with no time is not a thing. A
            // reminder with no due date is perfectly ordinary; an event has to
            // happen somewhen.
            return ["text": "I need to know when \"\(title)\" starts."]
        }

        let start: Date
        var repeats: ReminderRepeat?
        switch MomentPhrase.resolve(phrase, now: now, calendar: calendar) {
        case let .repeating(pattern, first, hasTime):
            guard hasTime else {
                return ["text": "What time does \"\(title)\" start?"]
            }
            start = first
            repeats = pattern
        case let .moment(date, hasTime):
            guard hasTime else {
                // A day with no clock time would become an all-day event, which
                // is a different thing from what someone asking for "Friday"
                // usually means. Asked rather than assumed.
                return ["text": "What time on \(describe(date, hasTime: false, calendar: calendar)) does \"\(title)\" start?"]
            }
            start = date
        case .recurring:
            return ["text": "I can add an event that repeats, but I need to know how often and at what time — like \"every Tuesday at 6pm\"."]
        case .unresolved:
            return ["text": "I could not work out when \"\(phrase)\" is. Tell me a time like \"tomorrow at 3pm\"."]
        }

        let minutes = EventLength.decide(stated: userText, modelMinutes: durationMinutes)
        guard let end = calendar.date(byAdding: .minute, value: minutes, to: start) else {
            return ["text": "I could not work out when that event ends."]
        }

        switch await write(NewEvent(title: title, start: start, end: end, repeats: repeats)) {
        case .written:
            // The length is said every time, the default included. Left out
            // when it was an hour, the card for "dinner at 8pm" said nothing
            // about how long dinner was — and the card, not the model's
            // sentence beside it, is what the person checks.
            return [
                "text": "Added \"\(title)\" to your calendar for \(describe(start, hasTime: true, calendar: calendar)), for \(EventLength.spoken(minutes))\(repeats.map { ", repeating \($0.sentence)" } ?? "").",
                "created": true
            ]
        case let .unauthorised(authorization):
            return ["text": authorization.explanation(for: .calendar)]
        case .failed:
            return ["text": "I could not add that to your calendar."]
        }
    }

    // MARK: - Shared judgement

    /// Moves a due time that has already gone today to the next day.
    ///
    /// A reminder set in the past cannot fire, and somebody asking to be
    /// reminded is asking for something in the future.
    ///
    /// This is not hypothetical. Asked "remind me to take the bins out at 2am"
    /// at ten past three in the afternoon, qwen3-1.7b filled `when` with "2am
    /// today" — and `MomentPhrase` honours a named day even when it is behind,
    /// on the reasoning that a person who names a day is owed the day they
    /// named. That reasoning holds for a day the USER named. The model added
    /// this one, and the result was an alarm thirteen hours in the past that
    /// would never go off.
    ///
    /// Only a time of day that has already gone TODAY is moved. An explicitly
    /// past date — "remind me on the 12th", said on the 14th — is left alone,
    /// because rolling that to next month would invent an intention nobody had.
    ///
    /// Its own function so the decision can be tested directly. Reaching it
    /// through `MomentPhrase.resolve` cannot be: `NSDataDetector` anchors "2am
    /// today" to the system clock and ignores any injected `now`, so a test
    /// that went the long way round would be testing the hour it ran at.
    static func rollPastTimeForward(
        _ due: Date?,
        now: Date,
        calendar: Calendar
    ) -> (due: Date?, moved: Bool) {
        guard let due, due < now, calendar.isDate(due, inSameDayAs: now),
              let nextDay = calendar.date(byAdding: .day, value: 1, to: due)
        else { return (due, false) }
        return (nextDay, true)
    }

    /// A title the user would recognise, or nothing.
    ///
    /// A model that calls the tool with an empty string, a lone full stop or
    /// the word "null" has not understood the request, and filing a reminder
    /// called "null" is a worse outcome than asking again.
    static func usableTitle(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else { return nil }
        let lowered = trimmed.lowercased()
        guard !["null", "none", "nil", "undefined", "n/a", "todo", "string"].contains(lowered) else { return nil }
        return trimmed
    }

    /// Whether an equivalent reminder is already open.
    ///
    /// Title compared case- and whitespace-insensitively, due date compared to
    /// the minute — two reminders a second apart are the same reminder asked
    /// for twice, and nobody sets two alarms 40 seconds apart on purpose.
    static func duplicate(of candidate: NewReminder, in rows: [ReminderRow], calendar: Calendar) -> ReminderRow? {
        rows.first { row in
            guard matches(candidate.title, row.title.attackerControlledValue()) else { return false }
            // A daily "take the pills" and a one-off "take the pills" at the
            // same hour are different things, and filing the second over the
            // first would silently drop the repetition.
            guard candidate.repeats == row.repeats else { return false }
            switch (candidate.due, row.due) {
            case (nil, nil): return true
            case let (mine?, theirs?): return calendar.isDate(mine, equalTo: theirs, toGranularity: .minute)
            default: return false
            }
        }
    }

    /// Loose title comparison, in one place so the duplicate check and the
    /// complete lookup cannot disagree about what "the same reminder" means.
    static func matches(_ left: String, _ right: String) -> Bool {
        normalise(left) == normalise(right)
    }

    static func normalise(_ text: String) -> String {
        text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .joined(separator: " ")
    }

    static func resolvedMinutes(_ requested: Int?) -> Int {
        guard let requested, requested > 0 else { return defaultEventMinutes }
        return min(requested, maximumEventMinutes)
    }

    /// - Parameter calendar: The caller's, threaded through rather than left to
    ///   `PersonalDataFormat`'s `.current` defaults. In the app the two are the
    ///   same object and nothing changes; in a test they are not, and a
    ///   confirmation rendered in the machine's zone while the reminder was
    ///   filed in the calendar's would read as a different hour — which is the
    ///   exact class of bug this sentence exists to let the user catch.
    static func describe(_ date: Date, hasTime: Bool, calendar: Calendar) -> String {
        hasTime
            ? PersonalDataFormat.moment(date, timeZone: calendar.timeZone, calendar: calendar)
            : PersonalDataFormat.day(date, timeZone: calendar.timeZone, calendar: calendar)
    }

    /// - Parameter movedForward: Whether the due date was rolled to tomorrow
    ///   because the time had already gone. Said out loud rather than done
    ///   quietly: the user asked for a time, and if this reading is wrong they
    ///   can only tell us so if they are told what happened.
    static func confirmation(
        for reminder: NewReminder,
        calendar: Calendar,
        movedForward: Bool = false
    ) -> String {
        guard let due = reminder.due else {
            return "Reminder added: \"\(reminder.title)\", with no due date."
        }
        let when = describe(due, hasTime: reminder.dueHasTime, calendar: calendar)
        if movedForward, reminder.repeats == nil {
            return "That time has already passed today, so I have set \"\(reminder.title)\" for \(when)."
        }
        guard let repeats = reminder.repeats else {
            return "Reminder set: \"\(reminder.title)\" for \(when)."
        }
        // The pattern is read back in words for the same reason the time is:
        // "every weekday" and "every week" are one misparse apart, and the user
        // is the only one who can tell which they meant.
        return "Reminder set: \"\(reminder.title)\", starting \(when) and repeating \(repeats.sentence)."
    }
}
