#if canImport(AetherMixCore)
import AetherMixCore
#endif
import AppKit
import SwiftUI

@main
struct AetherMixApp: App {
    @State private var bridgeClient: BridgeSocketClient
    @State private var wsClient: EngineWebSocketClient
    @State private var processManager: EngineProcessManager

    init() {
        NSApplication.shared.setActivationPolicy(.regular)
        let meterStore = MeterStore()
        let bridge = BridgeSocketClient()
        let ws = EngineWebSocketClient(meterStore: meterStore)
        let manager = EngineProcessManager()

        _bridgeClient = State(initialValue: bridge)
        _wsClient = State(initialValue: ws)
        _processManager = State(initialValue: manager)
    }

    var body: some Scene {
        WindowGroup("AetherMix Mission Control") {
            MainDashboardView(
                wsClient: wsClient,
                bridgeClient: bridgeClient,
                processManager: processManager
            )
            .onAppear {
                NSApplication.shared.activate(ignoringOtherApps: true)
                processManager.refreshBridgeLaunchAgentStatus()
                bridgeClient.start()
                processManager.ensureEngineRunning()
                wsClient.start()
            }
            .onDisappear {
                wsClient.stop()
                bridgeClient.stop()
                processManager.stopManagedEngine()
            }
        }
        .defaultSize(width: 1440, height: 880)
        .commands {
            CommandMenu("Mission Control") {
                Button("Emergency Bypass") {
                    bridgeClient.setBypass(latched: true)
                    Task { await wsClient.setMode(.emergencyBypass) }
                }
                .keyboardShortcut(".", modifiers: [.command])

                Button("Switch to Shadow Observe") {
                    bridgeClient.setBypass(latched: false)
                    Task { await wsClient.setMode(.shadowObserve) }
                }

                Divider()

                Button("Restore Previous Arming") {
                    Task { await wsClient.restorePreviousArming() }
                }

                Button("Check for Updates…") {
                    _ = processManager.requestUpdateAction(
                        "App/Model Update",
                        autonomyMode: wsClient.snapshot.autonomyMode
                    )
                }
                .disabled(wsClient.snapshot.autonomyMode == .active)
            }
        }
    }
}
