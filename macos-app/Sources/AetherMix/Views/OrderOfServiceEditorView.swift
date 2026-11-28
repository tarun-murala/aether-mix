import SwiftUI

/// Editable Order of Service & "Ignore Order of Service" Policy Sheet.
/// Allows operators to run 100% on live audio while still editing/marking segments
/// for post-service telemetry & learning context whenever helpful.
public struct OrderOfServiceEditorView: View {
    public let wsClient: EngineWebSocketClient

    @Environment(\.dismiss) private var dismiss
    @State private var ignoreOrderOfService: Bool = false
    @State private var draftSegments: [EditableSegment] = []
    @State private var isSaving: Bool = false
    @State private var errorMessage: String?

    private struct EditableSegment: Identifiable, Equatable {
        let id: String
        var title: String
        var expectedPhase: ServicePhase
    }

    public init(wsClient: EngineWebSocketClient) {
        self.wsClient = wsClient
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Order of Service & Live Audio Mode")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(MissionTheme.textPrimary)
                    Text("Toggle pure live-audio sensing or customize segments to mark learnings & telemetry.")
                        .font(.system(size: 11))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            // Ignore Order of Service Toggle Card
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Image(systemName: ignoreOrderOfService ? "waveform" : "list.bullet.rectangle")
                            .foregroundStyle(ignoreOrderOfService ? MissionTheme.amber : MissionTheme.cyanTarget)
                        Text("Ignore Order of Service (100% Live Audio)")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(MissionTheme.textPrimary)
                    }
                    Text(
                        ignoreOrderOfService
                            ? "Live classifier runs 100% on real-time audio with zero schedule bias. Segments below remain clickable to mark moments for learning & post-service review."
                            : "Live classifier blends 80% live audio with a 20% soft hint from the active segment and pre-stages upcoming worship songs."
                    )
                    .font(.system(size: 11))
                    .foregroundStyle(MissionTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Toggle("", isOn: $ignoreOrderOfService)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
            .padding(12)
            .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(ignoreOrderOfService ? MissionTheme.amber.opacity(0.7) : MissionTheme.border, lineWidth: 1)
            )

            // Segment List Header
            HStack {
                Text("SEGMENTS (OPTIONAL MARKERS FOR LEARNINGS & TELEMETRY)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
                Spacer()
                Button {
                    let nextNum = draftSegments.count + 1
                    draftSegments.append(
                        EditableSegment(
                            id: "seg_\(UUID().uuidString.prefix(6).lowercased())",
                            title: "Segment \(nextNum)",
                            expectedPhase: .worshipHigh
                        )
                    )
                } label: {
                    Label("Add Segment", systemImage: "plus")
                        .font(.system(size: 11, weight: .bold))
                }
                .buttonStyle(.bordered)

                Button("Reset Defaults") {
                    draftSegments = ChannelCatalog.defaultOrderOfService.map {
                        EditableSegment(id: $0.id, title: $0.title, expectedPhase: $0.expectedPhase)
                    }
                }
                .buttonStyle(.bordered)

                Button("Clear All") {
                    draftSegments.removeAll()
                }
                .buttonStyle(.bordered)
            }

            // Editable Segment Rows
            ScrollView {
                if draftSegments.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "waveform.path.ecg")
                            .font(.system(size: 24))
                            .foregroundStyle(MissionTheme.textSecondary)
                        Text("No segments defined — AetherMix will run 100% on live audio.")
                            .font(.system(size: 12))
                            .foregroundStyle(MissionTheme.textSecondary)
                        Text("Click 'Add Segment' anytime if you want quick buttons to mark moments for learning.")
                            .font(.system(size: 11))
                            .foregroundStyle(MissionTheme.textSecondary.opacity(0.8))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 36)
                } else {
                    VStack(spacing: 8) {
                        ForEach(Array(draftSegments.enumerated()), id: \.element.id) { idx, _ in
                            segmentRow(index: idx)
                        }
                    }
                }
            }
            .frame(maxHeight: .infinity)

            if let errorMessage {
                Text(errorMessage)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(MissionTheme.crimson)
            }

            // Footer Actions
            HStack {
                Text("Saved automatically to service_plan.json")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
                Spacer()
                Button("Cancel") { dismiss() }
                    .buttonStyle(.bordered)
                Button {
                    saveChanges()
                } label: {
                    Text(isSaving ? "Saving..." : "Save Order of Service")
                        .font(.system(size: 12, weight: .bold))
                        .padding(.horizontal, 8)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isSaving || !wsClient.isConnected)
            }
        }
        .padding(18)
        .frame(width: 680, height: 500)
        .background(MissionTheme.panelBg)
        .preferredColorScheme(.dark)
        .onAppear {
            ignoreOrderOfService = !wsClient.snapshot.useServicePlan
            draftSegments = wsClient.snapshot.orderOfService.map {
                EditableSegment(id: $0.id, title: $0.title, expectedPhase: $0.expectedPhase)
            }
        }
    }

    @ViewBuilder
    private func segmentRow(index: Int) -> some View {
        let isMarkedNow = index == wsClient.snapshot.serviceStepIdx
        HStack(spacing: 8) {
            Text("\(index + 1)")
                .font(.system(size: 11, weight: .black, design: .monospaced))
                .foregroundStyle(isMarkedNow ? Color.black : MissionTheme.textSecondary)
                .frame(width: 22, height: 22)
                .background(
                    isMarkedNow ? MissionTheme.cyanTarget : MissionTheme.canvasBg,
                    in: Circle()
                )

            TextField("Segment title", text: $draftSegments[index].title)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))

            Picker("Phase", selection: $draftSegments[index].expectedPhase) {
                ForEach(ServicePhase.allCases) { phase in
                    Text(phase.displayName).tag(phase)
                }
            }
            .labelsHidden()
            .frame(width: 155)

            Button {
                guard index > 0 else { return }
                draftSegments.swapAt(index, index - 1)
            } label: {
                Image(systemName: "arrow.up")
            }
            .buttonStyle(.borderless)
            .disabled(index == 0)
            .help("Move up")

            Button {
                guard index < draftSegments.count - 1 else { return }
                draftSegments.swapAt(index, index + 1)
            } label: {
                Image(systemName: "arrow.down")
            }
            .buttonStyle(.borderless)
            .disabled(index >= draftSegments.count - 1)
            .help("Move down")

            Button {
                draftSegments.remove(at: index)
            } label: {
                Image(systemName: "trash")
                    .foregroundStyle(MissionTheme.crimson)
            }
            .buttonStyle(.borderless)
            .help("Remove segment")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isMarkedNow ? MissionTheme.cyanTarget.opacity(0.6) : MissionTheme.border, lineWidth: 1)
        )
    }

    private func saveChanges() {
        let cleaned = draftSegments.map {
            ServiceSegment(
                id: $0.id,
                title: $0.title.trimmingCharacters(in: .whitespacesAndNewlines),
                expectedPhase: $0.expectedPhase
            )
        }
        if cleaned.contains(where: { $0.title.isEmpty }) {
            errorMessage = "Every segment needs a non-empty title (or remove blank rows)."
            return
        }
        errorMessage = nil
        isSaving = true
        Task {
            await wsClient.updateOrderOfService(cleaned, useServicePlan: !ignoreOrderOfService)
            isSaving = false
            dismiss()
        }
    }
}
