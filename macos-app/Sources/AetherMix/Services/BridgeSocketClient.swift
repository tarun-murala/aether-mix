import Darwin
import Foundation
import Observation

/// Direct POSIX AF_UNIX NDJSON client connecting SwiftUI directly to `midi_bridge_daemon`
/// (`~/.aethermix/bridge.sock`). Remains 100% live (< 2 ms) for virtual faders and Emergency
/// Bypass even when the Python AI engine is crashed, hung, or disconnected.
@MainActor
@Observable
public final class BridgeSocketClient {
    public private(set) var isConnected: Bool = false
    public private(set) var bridgeState: BridgeStatePayload = BridgeStatePayload()
    public private(set) var lastError: String?
    public private(set) var lastTouchEvent: BridgeHumanTouchEvent?
    public private(set) var lastCCEvent: BridgeHumanCCEvent?
    public private(set) var lastActionButton: BridgeActionButtonEvent?
    public private(set) var reconnectAttempts: Int = 0

    public let socketPath: String
    private let tokenProvider: @Sendable () -> String?
    private let transport: UnixDomainSocketTransport
    private var shouldStayConnected: Bool = false
    private var reconnectTask: Task<Void, Never>?

    public init(
        socketPath: String = AetherMixProtocol.defaultBridgeSocketPath.path,
        tokenPath: URL = AetherMixProtocol.defaultTokenPath,
        tokenOverride: String? = nil
    ) {
        self.socketPath = socketPath
        if let tokenOverride {
            self.tokenProvider = { tokenOverride }
        } else {
            self.tokenProvider = { SessionTokenLoader.loadToken(from: tokenPath) }
        }
        self.transport = UnixDomainSocketTransport(socketPath: socketPath)
    }

    public func start() {
        guard !shouldStayConnected else { return }
        shouldStayConnected = true
        connectOnce()
    }

    public func stop() {
        shouldStayConnected = false
        reconnectTask?.cancel()
        reconnectTask = nil
        transport.disconnect()
        isConnected = false
    }

    public func connectOnce() {
        let token = tokenProvider() ?? ""
        transport.connect(
            onConnected: { [weak self] in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.isConnected = true
                    self.reconnectAttempts = 0
                    self.lastError = nil
                    self.sendHello(token: token)
                    self.requestBridgeState()
                }
            },
            onLine: { [weak self] line in
                Task { @MainActor [weak self] in
                    self?.handleInboundLine(line)
                }
            },
            onDisconnected: { [weak self] errorMessage in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.isConnected = false
                    if let errorMessage {
                        self.lastError = errorMessage
                    }
                    self.scheduleReconnectIfNeeded()
                }
            }
        )
    }

    /// Attaches an already-connected POSIX `AF_UNIX` file descriptor (e.g. from `socketpair`)
    /// through the exact same non-blocking `DispatchSource` NDJSON read/write pipeline.
    public func attachConnectedDescriptor(_ fd: Int32) {
        let token = tokenProvider() ?? ""
        transport.attach(
            fileDescriptor: fd,
            onConnected: { [weak self] in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.isConnected = true
                    self.reconnectAttempts = 0
                    self.lastError = nil
                    self.sendHello(token: token)
                    self.requestBridgeState()
                }
            },
            onLine: { [weak self] line in
                Task { @MainActor [weak self] in
                    self?.handleInboundLine(line)
                }
            },
            onDisconnected: { [weak self] errorMessage in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.isConnected = false
                    if let errorMessage {
                        self.lastError = errorMessage
                    }
                }
            }
        )
    }

    /// Sends a direct manual CC write to `midi_bridge_daemon` (`ui_manual_cc`).
    /// Works even when the AI engine is completely dead!
    public func sendManualCC(channelId: String, control: ControlType, cc: Int) {
        let clamped = max(0, min(127, cc))
        bridgeState.lastSentCC["\(channelId):\(control.rawValue)"] = clamped
        guard let line = Self.encodeManualCC(channelId: channelId, control: control, cc: clamped) else {
            return
        }
        transport.sendLine(line)
    }

    /// Latches or releases Emergency Bypass directly inside `midi_bridge_daemon` (< 1 ms).
    public func setBypass(latched: Bool) {
        bridgeState.bypassLatched = latched
        guard let line = Self.encodeSetBypass(latched: latched) else { return }
        transport.sendLine(line)
    }

    /// Requests a fresh authoritative `bridge_state` snapshot from `midi_bridge_daemon`.
    public func requestBridgeState() {
        guard let line = Self.encodeGetBridgeState() else { return }
        transport.sendLine(line)
    }

    private func sendHello(token: String) {
        guard let line = Self.encodeHello(token: token) else { return }
        transport.sendLine(line)
    }

    public func handleInboundLine(_ line: String) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let message = Self.decodeInboundLine(trimmed)
        switch message {
        case .bridgeState(let payload):
            self.bridgeState = payload
        case .humanTouch(let event):
            self.lastTouchEvent = event
            let key = "\(event.channelId):\(event.control.rawValue)"
            self.bridgeState.humanSeq[key] = event.humanSeq
        case .humanCC(let event):
            self.lastCCEvent = event
            let key = "\(event.channelId):\(event.control.rawValue)"
            self.bridgeState.lastSentCC[key] = event.cc
            self.bridgeState.humanSeq[key] = event.humanSeq
        case .actionButton(let event):
            self.lastActionButton = event
            if event.action == "emergency_bypass" {
                self.bridgeState.bypassLatched = true
            }
        case .helloAck(let ok, let message):
            if !ok {
                self.lastError = message ?? "Bridge authentication rejected"
            }
        case .error(let message):
            self.lastError = message
        case .unknown:
            break
        }
    }

    private func scheduleReconnectIfNeeded() {
        guard shouldStayConnected else { return }
        reconnectTask?.cancel()
        reconnectAttempts += 1
        let delaySeconds = min(3.0, 0.5 * Double(min(reconnectAttempts, 6)))
        reconnectTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(delaySeconds))
            guard let self, self.shouldStayConnected, !self.isConnected else { return }
            self.connectOnce()
        }
    }

    // MARK: - Pure NDJSON Wire Codec (Testable Without Socket)

    nonisolated public static func encodeHello(
        token: String,
        role: String = "ui",
        protocolVersion: String = AetherMixProtocol.protocolVersion
    ) -> String? {
        encodeNDJSON(BridgeHelloMessage(token: token, role: role, protocolVersion: protocolVersion))
    }

    nonisolated public static func encodeManualCC(channelId: String, control: ControlType, cc: Int) -> String? {
        encodeNDJSON(BridgeManualCCMessage(channelId: channelId, control: control, cc: cc))
    }

    nonisolated public static func encodeSetBypass(latched: Bool) -> String? {
        encodeNDJSON(BridgeSetBypassMessage(latched: latched))
    }

    nonisolated public static func encodeGetBridgeState() -> String? {
        encodeNDJSON(BridgeGetStateMessage())
    }

    nonisolated public static func decodeInboundLine(_ line: String) -> BridgeInboundMessage {
        guard let data = line.data(using: .utf8) else {
            return .error(message: "Invalid UTF-8 frame")
        }
        guard let root = try? JSONDecoder().decode([String: JSONValue].self, from: data),
              let type = root["type"]?.stringValue else {
            return .error(message: "Malformed bridge JSON frame")
        }

        // Some bridge messages may nest fields under "payload" or place them flat at root (per SPEC §6.5).
        let targetData: Data
        if let nested = root["payload"]?.objectValue,
           let encodedNested = try? JSONEncoder().encode(nested) {
            targetData = encodedNested
        } else {
            targetData = data
        }

        let decoder = JSONDecoder()
        switch type {
        case "bridge_state":
            if let payload = try? decoder.decode(BridgeStatePayload.self, from: targetData) {
                return .bridgeState(payload)
            }
            return .error(message: "Failed to decode bridge_state")
        case "human_touch_event":
            if let event = try? decoder.decode(BridgeHumanTouchEvent.self, from: targetData) {
                return .humanTouch(event)
            }
            return .error(message: "Failed to decode human_touch_event")
        case "human_cc_event":
            if let event = try? decoder.decode(BridgeHumanCCEvent.self, from: targetData) {
                return .humanCC(event)
            }
            return .error(message: "Failed to decode human_cc_event")
        case "action_button_event":
            if let event = try? decoder.decode(BridgeActionButtonEvent.self, from: targetData) {
                return .actionButton(event)
            }
            return .error(message: "Failed to decode action_button_event")
        case "hello_ack", "ack":
            let ok = root["ok"]?.boolValue ?? root["payload"]?.objectValue?["ok"]?.boolValue ?? true
            let msg = root["message"]?.stringValue ?? root["payload"]?.objectValue?["message"]?.stringValue
            return .helloAck(ok: ok, message: msg)
        case "error":
            let msg = root["message"]?.stringValue
                ?? root["error"]?.stringValue
                ?? root["payload"]?.objectValue?["message"]?.stringValue
                ?? "Unknown bridge error"
            return .error(message: msg)
        default:
            return .unknown(type: type)
        }
    }

    nonisolated private static func encodeNDJSON<T: Encodable>(_ value: T) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(value),
              let str = String(data: data, encoding: .utf8) else {
            return nil
        }
        return str + "\n"
    }
}

// MARK: - Low-Level POSIX AF_UNIX Non-Blocking Stream Transport

final class UnixDomainSocketTransport: @unchecked Sendable {
    private let socketPath: String
    private let ioQueue = DispatchQueue(label: "com.aethermix.bridge.socket", qos: .userInteractive)
    private var fd: Int32 = -1
    private var readSource: DispatchSourceRead?
    private var readBuffer = Data()
    private var onLineCallback: (@Sendable (String) -> Void)?
    private var onDisconnectedCallback: (@Sendable (String?) -> Void)?

    init(socketPath: String) {
        self.socketPath = socketPath
    }

    deinit {
        disconnect()
    }

    func connect(
        onConnected: @escaping @Sendable () -> Void,
        onLine: @escaping @Sendable (String) -> Void,
        onDisconnected: @escaping @Sendable (String?) -> Void
    ) {
        ioQueue.async { [weak self] in
            guard let self else { return }
            self.teardownLocked(notify: false, errorMessage: nil)

            let sockFD = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
            guard sockFD >= 0 else {
                onDisconnected("Failed to create AF_UNIX socket (errno \(errno))")
                return
            }

            // Prevent SIGPIPE on broken pipe writes on macOS.
            var noSigPipe: Int32 = 1
            _ = Darwin.setsockopt(
                sockFD,
                SOL_SOCKET,
                SO_NOSIGPIPE,
                &noSigPipe,
                socklen_t(MemoryLayout<Int32>.size)
            )

            var addr = sockaddr_un()
            addr.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
            addr.sun_family = sa_family_t(AF_UNIX)
            let maxPathLen = MemoryLayout.size(ofValue: addr.sun_path)
            let pathBytes = Array(self.socketPath.utf8)
            guard pathBytes.count < maxPathLen else {
                Darwin.close(sockFD)
                onDisconnected("Bridge socket path exceeds sun_path limit (\(maxPathLen) bytes)")
                return
            }

            withUnsafeMutableBytes(of: &addr.sun_path) { rawBuffer in
                rawBuffer.initializeMemory(as: UInt8.self, repeating: 0)
                for (index, byte) in pathBytes.enumerated() {
                    rawBuffer[index] = byte
                }
            }

            let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
            let connectResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    Darwin.connect(sockFD, sockPtr, addrLen)
                }
            }

            guard connectResult == 0 else {
                let err = errno
                Darwin.close(sockFD)
                let errText = String(cString: strerror(err))
                onDisconnected("Bridge unavailable at \(self.socketPath): \(errText)")
                return
            }

            self.installConnectedSocketLocked(
                sockFD,
                onConnected: onConnected,
                onLine: onLine,
                onDisconnected: onDisconnected
            )
        }
    }

    func attach(
        fileDescriptor sockFD: Int32,
        onConnected: @escaping @Sendable () -> Void,
        onLine: @escaping @Sendable (String) -> Void,
        onDisconnected: @escaping @Sendable (String?) -> Void
    ) {
        ioQueue.async { [weak self] in
            guard let self else { return }
            self.teardownLocked(notify: false, errorMessage: nil)
            self.installConnectedSocketLocked(
                sockFD,
                onConnected: onConnected,
                onLine: onLine,
                onDisconnected: onDisconnected
            )
        }
    }

    private func installConnectedSocketLocked(
        _ sockFD: Int32,
        onConnected: @escaping @Sendable () -> Void,
        onLine: @escaping @Sendable (String) -> Void,
        onDisconnected: @escaping @Sendable (String?) -> Void
    ) {
        var noSigPipe: Int32 = 1
        _ = Darwin.setsockopt(
            sockFD,
            SOL_SOCKET,
            SO_NOSIGPIPE,
            &noSigPipe,
            socklen_t(MemoryLayout<Int32>.size)
        )

        // Switch connected socket to non-blocking mode for DispatchSource reads/writes.
        let flags = Darwin.fcntl(sockFD, F_GETFL, 0)
        if flags >= 0 {
            _ = Darwin.fcntl(sockFD, F_SETFL, flags | O_NONBLOCK)
        }

        self.fd = sockFD
        self.readBuffer.removeAll(keepingCapacity: true)
        self.onLineCallback = onLine
        self.onDisconnectedCallback = onDisconnected

        let source = DispatchSource.makeReadSource(fileDescriptor: sockFD, queue: self.ioQueue)
        source.setEventHandler { [weak self] in
            self?.drainReadableBytesLocked()
        }
        source.setCancelHandler {
            Darwin.close(sockFD)
        }
        self.readSource = source
        source.resume()

        onConnected()
    }

    func sendLine(_ line: String) {
        guard let data = line.data(using: .utf8) else { return }
        ioQueue.async { [weak self] in
            guard let self, self.fd >= 0 else { return }
            let sockFD = self.fd
            var totalWritten = 0
            let byteCount = data.count

            data.withUnsafeBytes { rawBuf in
                guard let base = rawBuf.baseAddress else { return }
                while totalWritten < byteCount {
                    let written = Darwin.write(sockFD, base.advanced(by: totalWritten), byteCount - totalWritten)
                    if written > 0 {
                        totalWritten += written
                    } else if written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                        continue
                    } else {
                        let err = errno
                        let msg = String(cString: strerror(err))
                        self.teardownLocked(notify: true, errorMessage: "Bridge write failed: \(msg)")
                        return
                    }
                }
            }
        }
    }

    func disconnect() {
        ioQueue.async { [weak self] in
            self?.teardownLocked(notify: false, errorMessage: nil)
        }
    }

    private func drainReadableBytesLocked() {
        guard fd >= 0 else { return }
        var chunk = [UInt8](repeating: 0, count: 4096)

        while true {
            let bytesRead = Darwin.read(fd, &chunk, chunk.count)
            if bytesRead > 0 {
                readBuffer.append(contentsOf: chunk[0..<bytesRead])
                emitBufferedLinesLocked()
            } else if bytesRead == 0 {
                teardownLocked(notify: true, errorMessage: "Bridge closed connection")
                return
            } else {
                let err = errno
                if err == EAGAIN || err == EWOULDBLOCK {
                    return
                }
                if err == EINTR {
                    continue
                }
                let msg = String(cString: strerror(err))
                teardownLocked(notify: true, errorMessage: "Bridge read error: \(msg)")
                return
            }
        }
    }

    private func emitBufferedLinesLocked() {
        let newline: UInt8 = 0x0A
        while let range = readBuffer.firstRange(of: Data([newline])) {
            let lineData = readBuffer.subdata(in: readBuffer.startIndex..<range.lowerBound)
            readBuffer.removeSubrange(readBuffer.startIndex..<range.upperBound)
            if let line = String(data: lineData, encoding: .utf8), !line.isEmpty {
                onLineCallback?(line)
            }
        }
    }

    private func teardownLocked(notify: Bool, errorMessage: String?) {
        let disconnectedHandler = onDisconnectedCallback
        onLineCallback = nil
        onDisconnectedCallback = nil
        readBuffer.removeAll(keepingCapacity: false)

        if let source = readSource {
            readSource = nil
            fd = -1
            source.cancel() // Cancel handler closes sockFD safely.
        } else if fd >= 0 {
            Darwin.close(fd)
            fd = -1
        }

        if notify {
            disconnectedHandler?(errorMessage)
        }
    }
}
