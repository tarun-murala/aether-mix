import SwiftUI

/// Drawer to inspect, enable/disable, or delete permanent & session mixing rules,
/// and reset learned offset regression weights per phase or globally.
public struct RulesManagerView: View {
    public let wsClient: EngineWebSocketClient
    @Environment(\.dismiss) private var dismiss

    @State private var selectedResetPhase: ServicePhase? = nil
    @State private var confirmResetAlert = false
    @State private var ruleToDelete: MixingRule?

    public init(wsClient: EngineWebSocketClient) {
        self.wsClient = wsClient
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Rules Manager & Learning Reset")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(MissionTheme.textPrimary)
                    Text("Toggle or delete active rules in mixing_rules.json, or reset Ridge offset weights.")
                        .font(.system(size: 11))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            Divider().overlay(MissionTheme.border)

            // Active Rules List
            HStack {
                Text("ACTIVE RULES (\(wsClient.snapshot.activeRules.count))")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.cyanTarget)
                Spacer()
                Text("Continuous cap: [-12, +6 dB] • Conditional cap: [-36, 0 dB]")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
            }

            if wsClient.snapshot.activeRules.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "slider.horizontal.below.rectangle")
                        .font(.system(size: 24))
                        .foregroundStyle(MissionTheme.textSecondary.opacity(0.6))
                    Text("No active permanent or session rules loaded.")
                        .font(.system(size: 12))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
                .frame(maxWidth: .infinity, minHeight: 160)
                .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(wsClient.snapshot.activeRules) { rule in
                            RuleRowView(
                                rule: rule,
                                onToggle: { enabled in
                                    Task { await wsClient.toggleRule(ruleId: rule.id, enabled: enabled) }
                                },
                                onDelete: {
                                    ruleToDelete = rule
                                }
                            )
                        }
                    }
                }
                .frame(minHeight: 220, maxHeight: 340)
            }

            Divider().overlay(MissionTheme.border)

            // Reset Phase Learning Section
            VStack(alignment: .leading, spacing: 8) {
                Text("RESET OFFSET LEARNER WEIGHTS")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.amber)

                HStack(spacing: 10) {
                    Picker("Target Phase", selection: $selectedResetPhase) {
                        Text("All Phases").tag(ServicePhase?.none)
                        ForEach(ServicePhase.macroPhases) { phase in
                            Text(phase.displayName).tag(ServicePhase?.some(phase))
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 220)

                    Button {
                        confirmResetAlert = true
                    } label: {
                        Label("Reset Phase Learning", systemImage: "arrow.counterclockwise")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(Color.black)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(MissionTheme.amber, in: RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                    .disabled(!wsClient.isConnected)

                    Spacer()
                }
            }
            .padding(12)
            .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(MissionTheme.border, lineWidth: 1))
        }
        .padding(18)
        .frame(minWidth: 660, maxWidth: 660, minHeight: 500)
        .background(MissionTheme.canvasBg)
        .preferredColorScheme(.dark)
        .alert("Reset Learned Offsets?", isPresented: $confirmResetAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Reset Weights", role: .destructive) {
                let target = selectedResetPhase
                Task { await wsClient.resetLearning(phase: target) }
            }
        } message: {
            Text(
                selectedResetPhase == nil
                    ? "This resets learned per-channel Ridge offsets for ALL macro phases back to calibrated baselines."
                    : "This resets learned per-channel Ridge offsets for \(selectedResetPhase!.displayName) back to calibrated baselines."
            )
        }
        .alert("Delete Rule?", isPresented: Binding(
            get: { ruleToDelete != nil },
            set: { if !$0 { ruleToDelete = nil } }
        )) {
            Button("Cancel", role: .cancel) { ruleToDelete = nil }
            Button("Delete", role: .destructive) {
                if let rule = ruleToDelete {
                    Task { await wsClient.deleteRule(ruleId: rule.id) }
                }
                ruleToDelete = nil
            }
        } message: {
            Text("Permanently remove \"\(ruleToDelete?.description ?? "")\"? An atomic backup is written before deletion.")
        }
    }
}

private struct RuleRowView: View {
    let rule: MixingRule
    let onToggle: (Bool) -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Toggle("", isOn: Binding(
                get: { rule.enabled },
                set: { onToggle($0) }
            ))
            .labelsHidden()
            .toggleStyle(.switch)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(rule.description)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(rule.enabled ? MissionTheme.textPrimary : MissionTheme.textSecondary)

                    Text(rule.scope == .ephemeralSession ? "SESSION" : "PERMANENT")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(rule.scope == .ephemeralSession ? MissionTheme.cyanTarget : MissionTheme.emerald)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(MissionTheme.canvasBg, in: Capsule())
                }

                HStack(spacing: 8) {
                    Text(ChannelCatalog.descriptor(for: rule.channelId).label)
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(MissionTheme.cyanTarget)
                    Text(rule.phase.rawValue)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(MissionTheme.textSecondary)
                    Text("\(rule.parameter.displayName): \(formattedValue)")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(MissionTheme.textPrimary)
                    if let cond = rule.condition {
                        Text("IF \(cond.whenChannel).\(cond.predicate.rawValue)")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(MissionTheme.amber)
                    }
                }
            }

            Spacer()

            Button(role: .destructive) {
                onDelete()
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 12))
                    .foregroundStyle(MissionTheme.crimson)
                    .padding(6)
                    .background(MissionTheme.crimson.opacity(0.14), in: RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(MissionTheme.border, lineWidth: 1))
    }

    private var formattedValue: String {
        if rule.parameter == .conditionalMute {
            return rule.value >= 0.5 ? "MUTE" : "UNMUTE"
        }
        return String(format: "%+.1f dB", rule.value)
    }
}
