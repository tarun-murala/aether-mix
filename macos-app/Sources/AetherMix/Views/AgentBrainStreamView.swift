import SwiftUI

public enum BrainFeedFilter: String, CaseIterable, Identifiable, Sendable {
    case all = "All"
    case transitions = "Transitions"
    case safety = "Safety"
    case llm = "LLM"
    case rlhf = "RLHF"

    public var id: String { rawValue }

    public func matches(_ category: BrainCategory) -> Bool {
        switch self {
        case .all: return true
        case .transitions: return category == .perception || category == .action
        case .safety: return category == .safety
        case .llm: return category == .llm
        case .rlhf: return category == .rlhf
        }
    }
}

/// Right-hand Mission Control panel combining:
/// 1. Pending / Preview Rule Cards with exact `[Apply Session]` / `[Save Permanent]` / `[Dismiss]`
///    buttons (NEVER using 👍 on rule cards per SPEC v5.1).
/// 2. Filterable real-time Agent Brain Event feed (All / Transitions / Safety / LLM / RLHF).
public struct AgentBrainStreamView: View {
    public let wsClient: EngineWebSocketClient
    @State private var selectedFilter: BrainFeedFilter = .all
    @State private var expandedEventIds: Set<String> = []

    public init(wsClient: EngineWebSocketClient) {
        self.wsClient = wsClient
    }

    private var filteredEvents: [BrainEventPayload] {
        wsClient.brainEvents.filter { selectedFilter.matches($0.category) }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header + Filter Bar
            HStack {
                Label("AGENT BRAIN STREAM", systemImage: "brain.head.profile")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textPrimary)

                Spacer()

                Text("\(filteredEvents.count)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(MissionTheme.canvasBg, in: Capsule())
            }

            HStack(spacing: 4) {
                ForEach(BrainFeedFilter.allCases) { filter in
                    Button {
                        selectedFilter = filter
                    } label: {
                        Text(filter.rawValue)
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(selectedFilter == filter ? Color.black : MissionTheme.textSecondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                selectedFilter == filter ? MissionTheme.cyanTarget : MissionTheme.canvasBg,
                                in: Capsule()
                            )
                    }
                    .buttonStyle(.plain)
                }
            }

            // Pending / Preview Rule Cards (SPEC §5.3 Tier 3 & §6.4)
            if !wsClient.snapshot.pendingRules.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("PROPOSED & PENDING RULES (\(wsClient.snapshot.pendingRules.count))")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(MissionTheme.amber)

                    ForEach(wsClient.snapshot.pendingRules) { card in
                        PendingRuleCardView(card: card) { action in
                            Task {
                                await wsClient.resolvePendingRule(ruleId: card.ruleId, action: action)
                            }
                        }
                    }
                }
            }

            Divider().overlay(MissionTheme.border)

            // Scrollable Brain Event Feed
            if filteredEvents.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "waveform")
                        .font(.system(size: 22))
                        .foregroundStyle(MissionTheme.textSecondary.opacity(0.6))
                    Text("Waiting for agent perception & reasoning events…")
                        .font(.system(size: 11))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        ForEach(filteredEvents) { event in
                            BrainEventRow(
                                event: event,
                                isExpanded: expandedEventIds.contains(event.id),
                                onToggleExpand: {
                                    if expandedEventIds.contains(event.id) {
                                        expandedEventIds.remove(event.id)
                                    } else {
                                        expandedEventIds.insert(event.id)
                                    }
                                }
                            )
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
        .padding(12)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(MissionTheme.border, lineWidth: 1))
    }
}

// MARK: - Pending Rule Card (Exact Buttons: [Apply Session], [Save Permanent], [Dismiss])

private struct PendingRuleCardView: View {
    let card: PendingRuleCard
    let onResolve: (PendingRuleAction) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                if card.requiresPreview && !card.appliedInSession {
                    Text("PREVIEW REQUIRED")
                        .font(.system(size: 9, weight: .black, design: .monospaced))
                        .foregroundStyle(Color.black)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(MissionTheme.amber, in: Capsule())
                } else {
                    Text("SESSION APPLIED")
                        .font(.system(size: 9, weight: .black, design: .monospaced))
                        .foregroundStyle(Color.black)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(MissionTheme.cyanTarget, in: Capsule())
                }

                Spacer()

                Text(card.phase.rawValue)
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
            }

            Text(card.description)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(MissionTheme.textPrimary)

            HStack(spacing: 6) {
                Text(ChannelCatalog.descriptor(for: card.channelId).label)
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(MissionTheme.cyanTarget)
                Text("•")
                    .foregroundStyle(MissionTheme.textSecondary)
                Text("\(card.parameter.displayName): \(formattedValue)")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(MissionTheme.textPrimary)
            }

            if let cond = card.condition {
                Text("When \(ChannelCatalog.descriptor(for: cond.whenChannel).label) is \(cond.predicate.rawValue)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(MissionTheme.amber)
            }

            // Explicit rule resolution buttons — NEVER 👍 on rule cards!
            HStack(spacing: 6) {
                if !card.appliedInSession {
                    Button {
                        onResolve(.applySession)
                    } label: {
                        Text("Apply Session")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(Color.black)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(MissionTheme.cyanTarget, in: RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                }

                Button {
                    onResolve(.promote)
                } label: {
                    Text("Save Permanent")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Color.black)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(MissionTheme.emerald, in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)

                Spacer()

                Button {
                    onResolve(.dismiss)
                } label: {
                    Text("Dismiss")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(MissionTheme.textSecondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(MissionTheme.canvasBg, in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(MissionTheme.border, lineWidth: 1))
                }
                .buttonStyle(.plain)
            }
            .padding(.top, 2)
        }
        .padding(10)
        .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    card.requiresPreview && !card.appliedInSession ? MissionTheme.amber : MissionTheme.cyanTarget.opacity(0.6),
                    lineWidth: 1.2
                )
        )
    }

    private var formattedValue: String {
        if card.parameter == .conditionalMute {
            return card.value >= 0.5 ? "MUTE" : "UNMUTE"
        }
        return String(format: "%+.1f dB", card.value)
    }
}

// MARK: - Single Brain Event Row

private struct BrainEventRow: View {
    let event: BrainEventPayload
    let isExpanded: Bool
    let onToggleExpand: () -> Void

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(event.category.rawValue)
                    .font(.system(size: 8, weight: .black, design: .monospaced))
                    .foregroundStyle(Color.black)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(MissionTheme.categoryColor(event.category), in: Capsule())

                Text(formattedTime)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)

                Spacer()

                Text(String(format: "%.0f%%", event.confidence * 100.0))
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
            }

            Text(event.summary)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MissionTheme.textPrimary)

            if isExpanded && !event.detail.isEmpty {
                Text(event.detail)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
                    .padding(.top, 2)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(MissionTheme.cardBg.opacity(0.75), in: RoundedRectangle(cornerRadius: 7))
        .overlay(
            RoundedRectangle(cornerRadius: 7)
                .stroke(MissionTheme.border.opacity(0.7), lineWidth: 0.8)
        )
        .onTapGesture {
            if !event.detail.isEmpty {
                onToggleExpand()
            }
        }
    }

    private var formattedTime: String {
        guard event.tsMs > 0 else { return "--:--:--" }
        let date = Date(timeIntervalSince1970: Double(event.tsMs) / 1000.0)
        return Self.timeFormatter.string(from: date)
    }
}
