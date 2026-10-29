import Foundation
import FlyingFox
import FlyingSocks

/// The HTTP server that turns the phone into an inference endpoint.
///
/// Lifecycle is deliberately explicit rather than automatic. iOS closes a
/// listening socket when it suspends the app, so "running" is a state the app
/// has to re-establish on every foreground transition — a server that silently
/// believed it was still up would report a URL that refuses connections.
public actor InferenceServer {
    public enum State: Sendable, Equatable {
        case stopped
        case starting
        case running(host: String, port: UInt16)
        case failed(String)

        public var isRunning: Bool {
            if case .running = self { return true }
            return false
        }
    }

    public private(set) var configuration: ServerConfiguration
    public let log: RequestLog
    /// The six-digit handshake that trades itself for the API key, so nobody
    /// retypes 35 random characters off a phone screen.
    public let pairing = PairingSession()

    private let engine: any InferenceEngine
    private let modelsProvider: @Sendable () async -> [ModelRecord]
    /// Supplied by the app so the server can fetch a model on request. Nil in
    /// tests and in any build without downloading, where /api/pull answers 501.
    let puller: (@Sendable (ModelRecord) -> AsyncThrowingStream<DownloadProgress, any Error>)?
    /// Used to tell a searcher whether a file will fit before they spend
    /// gigabytes finding out.
    let deviceBudget: DeviceBudget

    private var server: HTTPServer?
    private var runTask: Task<Void, Never>?
    private var state: State = .stopped {
        didSet {
            guard state != oldValue else { return }
            for continuation in stateObservers.values { continuation.yield(state) }
        }
    }
    private var stateObservers: [UUID: AsyncStream<State>.Continuation] = [:]
    private var activeRequests = 0
    private var condition: ServeCondition = .ok

    public init(
        configuration: ServerConfiguration = ServerConfiguration(),
        engine: any InferenceEngine,
        models: @escaping @Sendable () async -> [ModelRecord],
        puller: (@Sendable (ModelRecord) -> AsyncThrowingStream<DownloadProgress, any Error>)? = nil,
        deviceBudget: DeviceBudget = .current(hasIncreasedMemoryLimit: false),
        log: RequestLog = RequestLog()
    ) {
        self.configuration = configuration
        self.engine = engine
        self.modelsProvider = models
        self.puller = puller
        self.deviceBudget = deviceBudget
        self.log = log
    }

    public func currentState() -> State { state }

    func currentStateValue() -> State { state }

    func resolvedPort() async -> UInt16 {
        if case let .running(_, port) = state { return port }
        return configuration.port
    }

    @discardableResult
    public func openPairing() async -> String { await pairing.open() }

    public func closePairing() async { await pairing.close() }

    public func stateStream() -> AsyncStream<State> {
        let id = UUID()
        return AsyncStream { continuation in
            stateObservers[id] = continuation
            continuation.yield(state)
            continuation.onTermination = { [weak self] _ in
                Task { await self?.removeStateObserver(id) }
            }
        }
    }

    private func removeStateObserver(_ id: UUID) { stateObservers[id] = nil }

    /// Applies a new configuration, restarting the listener if the change
    /// affects the socket. Changing the API key while running takes effect on
    /// the next request without a restart, which matters because rotating a
    /// leaked key should not drop in-flight work.
    public func apply(_ new: ServerConfiguration) async throws {
        let needsRestart = new.port != configuration.port || new.binding != configuration.binding
        configuration = new
        if needsRestart, await liveState().isRunning {
            await stop()
            try await start()
        }
    }

    public func start() async throws {
        // Reject only the states where starting again is wrong. `.failed` must
        // stay retryable: the commonest failure is a port already in use, and a
        // guard that excluded `.failed` would leave the Start button dead for
        // the life of the process — worse than the bug it prevents.
        switch state {
        case .running:
            // A cached `.running` can be a lie. iOS closes the listening socket
            // when the app suspends without telling anyone, so trust the socket
            // rather than the memo we wrote about it.
            if await isSocketListening() { return }
            await stop()
        case .starting:
            return
        case .stopped, .failed:
            break
        }
        state = .starting

        let server: HTTPServer
        switch configuration.binding {
        case .loopback:
            server = HTTPServer(
                address: try sockaddr_in.inet(ip4: "127.0.0.1", port: configuration.port),
                timeout: configuration.connectionTimeout
            )
        case .localNetwork:
            server = HTTPServer(
                address: sockaddr_in.inet(port: configuration.port),
                timeout: configuration.connectionTimeout
            )
        }
        await installRoutes(on: server)

        // The task notices the listener ending for any reason — a bind failure,
        // a cancellation, or iOS reclaiming the socket — which is the only
        // signal that `state` has gone stale.
        let task = Task { [weak self] in
            try? await server.run()
            await self?.listenerEnded(server)
        }
        self.server = server
        self.runTask = task

        do {
            try await server.waitUntilListening(timeout: 5)
        } catch {
            // Cancel THIS attempt's task, not whatever `runTask` happens to hold:
            // another start may have overtaken us at one of the awaits above, and
            // tearing down its listener would leave a socket bound with no
            // reference to it.
            task.cancel()
            if self.server === server {
                self.server = nil
                self.runTask = nil
            }
            if case .starting = state {
                state = .failed(String(describing: error))
            }
            throw error
        }

        let host = configuration.binding == .loopback
            ? "127.0.0.1"
            : (NetworkInterface.localIPv4Address() ?? "0.0.0.0")
        // Read the port back rather than trusting the configuration: port 0 asks
        // the kernel to pick one, and the UI has to display the number a client
        // should actually dial.
        state = .running(host: host, port: await server.resolvedPort() ?? configuration.port)
    }

    /// Called when a listener's run loop exits, however it exited.
    private func listenerEnded(_ ended: HTTPServer) {
        // A newer start() may already have taken over; without this check the
        // old listener's exit clobbers the new one's `.running` state.
        guard self.server === ended else { return }
        self.server = nil
        self.runTask = nil
        // A dead listener leaves in-flight requests counted forever, which would
        // permanently eat into the admission limit after a restart.
        activeRequests = 0
        state = .stopped
    }

    private func isSocketListening() async -> Bool {
        guard let server else { return false }
        return await server.isListening
    }

    /// `state`, reconciled against the socket. Prefer this to `currentState()`
    /// anywhere a stale `.running` would cause the caller to skip a repair.
    public func liveState() async -> State {
        if state.isRunning, await isSocketListening() == false {
            state = .stopped
        }
        return state
    }

    public func stop() async {
        let stopping = server
        // Clear first, so the run task's listenerEnded sees a different (nil)
        // server and declines to touch state we are about to set ourselves.
        server = nil
        let task = runTask
        runTask = nil

        await stopping?.stop(timeout: 1)
        task?.cancel()
        activeRequests = 0
        state = .stopped
    }

    // MARK: - Shared request plumbing

    func models() async -> [ModelRecord] { await modelsProvider() }

    func currentEngine() -> any InferenceEngine { engine }

    /// Returns nil when the request may proceed, or the rejection to send back.
    /// Wraps `authorize` and records the refusal. A rejected request is the one
    /// a user most wants to see in the log, and it was the only kind that never
    /// appeared there.
    func authorizeAndLog(_ request: HTTPRequest) async -> HTTPResponse? {
        guard let rejection = authorize(request) else { return nil }
        await logImmediate(request, status: rejection.statusCode.code)
        return rejection
    }

    func authorize(_ request: HTTPRequest) -> HTTPResponse? {
        guard configuration.requiresAuth else { return nil }
        let presented = request.headers[.authorization]?
            .replacingOccurrences(of: "Bearer ", with: "")
            ?? request.headers[HTTPHeader("X-API-Key")]
        guard let presented, constantTimeEquals(presented, configuration.apiKey) else {
            return errorResponse(
                status: .unauthorized,
                message: "Missing or invalid API key.",
                type: "invalid_request_error",
                style: request.path.hasPrefix("/api") ? .ollama : .openAI
            )
        }
        return nil
    }

    /// Compares without an early return so that a wrong key takes the same time
    /// to reject regardless of how much of the prefix was correct.
    func constantTimeEquals(_ lhs: String, _ rhs: String) -> Bool {
        let a = Array(lhs.utf8), b = Array(rhs.utf8)
        guard a.count == b.count else { return false }
        var difference: UInt8 = 0
        for index in a.indices { difference |= a[index] ^ b[index] }
        return difference == 0
    }

    /// Admission control. One GPU, one generation: a second concurrent request
    /// does not run twice as fast, it makes both slower and heats the device.
    func beginRequest() -> Bool {
        guard activeRequests < configuration.maxConcurrentRequests else { return false }
        activeRequests += 1
        return true
    }

    /// Admission for a client that would rather wait than be turned away.
    ///
    /// The Anthropic route's: Claude Code sends several requests at once, and
    /// the engine serialises them anyway, so up to `anthropicQueueDepth` of them
    /// are let through to wait their turn there. They count against the same
    /// total as everything else, so while one waits the other dialects still
    /// see a busy phone and answer 503, exactly as before.
    func beginQueuedRequest() -> Bool {
        guard activeRequests < configuration.maxConcurrentRequests + Self.anthropicQueueDepth else { return false }
        activeRequests += 1
        return true
    }

    func activeRequestCount() -> Int { activeRequests }

    public func currentCondition() -> ServeCondition { condition }

    /// Set by the app from thermal and battery notifications. The server keeps
    /// listening while throttled — a refused request with a reason is far more
    /// useful to a client than a refused connection.
    public func setCondition(_ new: ServeCondition) {
        condition = new
    }

    func serveCondition() -> ServeCondition { condition }

    func endRequest() {
        activeRequests = max(0, activeRequests - 1)
    }

    func corsHeaders() -> HTTPHeaders {
        guard configuration.allowCORS else { return [:] }
        return [
            HTTPHeader("Access-Control-Allow-Origin"): "*",
            HTTPHeader("Access-Control-Allow-Headers"): "Authorization, Content-Type, X-API-Key",
            HTTPHeader("Access-Control-Allow-Methods"): "GET, POST, OPTIONS"
        ]
    }

    func contextCap() -> Int { configuration.maxContextTokens }
}
