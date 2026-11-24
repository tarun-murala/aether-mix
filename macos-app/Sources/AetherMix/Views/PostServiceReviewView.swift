import SwiftUI

/// Post-Service Review Drawer for inspecting and committing or discarding
/// learning samples queued while Service Lock (`freeze_learning == true`) was active.
public struct PostServiceReviewView: View {
    public let wsClient: EngineWebSocketClient
    @Environment(\.dismiss) private var dismiss
    @State private var selectedIds: Set<String> = []

    public init(wsClient: EngineWebSocketClient) {
        self.wsClient = wsClient
    }

    private var queuedCount: Int {
        max(wsClient.snapshot.queuedReviewCount, wsClient.queuedReviewSamples.count)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Post-Service Freeze Learning Review")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(MissionTheme.textPrimary)
                    Text("Review operator corrections & feedback queued while Service Lock (Freeze Learning) was enabled.")
                        .font(.system(size: 11))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            Divider().overlay(MissionTheme.border)

            // Summary Banner
            HStack {
                Image(systemName: wsClient.snapshot.freezeLearning ? "lock.fill" : "lock.open.fill")
                    .foregroundStyle(wsClient.snapshot.freezeLearning ? MissionTheme.amber : MissionTheme.emerald)
                Text(
                    wsClient.snapshot.freezeLearning
                        ? "Service Lock is currently ON — live corrections are being queued instead of mutating weights."
                        : "Service Lock is OFF — ready to commit or discard queued samples."
                )
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MissionTheme.textPrimary)
                Spacer()
                Text("\(queuedCount) queued")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(queuedCount > 0 ? MissionTheme.amber : MissionTheme.textSecondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(MissionTheme.canvasBg, in: Capsule())
            }
            .padding(10)
            .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(MissionTheme.border, lineWidth: 1))

            // Sample List
            if wsClient.queuedReviewSamples.isEmpty && queuedCount == 0 {
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(MissionTheme.emerald)
                    Text("No queued Freeze Learning samples pending review.")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
                .frame(maxWidth: .infinity, minHeight: 200)
                .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
            } else if wsClient.queuedReviewSamples.isEmpty && queuedCount > 0 {
                VStack(spacing: 8) {
                    Image(systemName: "tray.full.fill")
                        .font(.system(size: 26))
                        .foregroundStyle(MissionTheme.amber)
                    Text("\(queuedCount) learning sample(s) queued in SQLite telemetry store.")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(MissionTheme.textPrimary)
                    Text("Commit all samples to update per-(phase, channel, control) Ridge weights, or discard them to keep pre-service weights unchanged.")
                        .font(.system(size: 11))
                        .foregroundStyle(MissionTheme.textSecondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 440)
                }
                .frame(maxWidth: .infinity, minHeight: 200)
                .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
            } else {
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(wsClient.queuedReviewSamples) { sample in
                            HStack(spacing: 10) {
                                Toggle("", isOn: Binding(
                                    get: { selectedIds.contains(sample.id) },
                                    set: { isOn in
                                        if isOn { selectedIds.insert(sample.id) }
                                        else { selectedIds.remove(sample.id) }
                                    }
                                ))
                                .labelsHidden()

                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 6) {
                                        Text(ChannelCatalog.descriptor(for: sample.channelId).label)
                                            .font(.system(size: 12, weight: .bold))
                                            .foregroundStyle(MissionTheme.textPrimary)
                                        Text(sample.control.rawValue.uppercased())
                                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                                            .foregroundStyle(MissionTheme.cyanTarget)
                                        Text(sample.phase.shortLabel)
                                            .font(.system(size: 9, weight: .semibold, design: .monospaced))
                                            .foregroundStyle(MissionTheme.textSecondary)
                                    }
                                    Text("Source: \(sample.source)")
                                        .font(.system(size: 10))
                                        .foregroundStyle(MissionTheme.textSecondary)
                                }

                                Spacer()

                                Text(String(format: "%+.2f dB", sample.targetOffsetDb))
                                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                                    .foregroundStyle(sample.targetOffsetDb >= 0 ? MissionTheme.emerald : MissionTheme.amber)
                            }
                            .padding(10)
                            .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(MissionTheme.border, lineWidth: 1))
                        }
                    }
                }
                .frame(minHeight: 200, maxHeight: 320)
            }

            Divider().overlay(MissionTheme.border)

            // Action Buttons
            HStack(spacing: 10) {
                Button {
                    let ids = selectedIds.isEmpty ? nil : Array(selectedIds)
                    Task { await wsClient.commitQueuedReviews(commit: false, sampleIds: ids) }
                    selectedIds.removeAll()
                } label: {
                    Label(selectedIds.isEmpty ? "Discard All" : "Discard Selected (\(selectedIds.count))", systemImage: "trash")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(MissionTheme.crimson)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(MissionTheme.crimson.opacity(0.16), in: RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(.plain)
                .disabled(queuedCount == 0 || !wsClient.isConnected)

                Spacer()

                Button {
                    let ids = selectedIds.isEmpty ? nil : Array(selectedIds)
                    Task { await wsClient.commitQueuedReviews(commit: true, sampleIds: ids) }
                    selectedIds.removeAll()
                } label: {
                    Label(selectedIds.isEmpty ? "Commit All to Model" : "Commit Selected (\(selectedIds.count))", systemImage: "checkmark.circle.fill")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(Color.black)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .background(MissionTheme.emerald, in: RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(.plain)
                .disabled(queuedCount == 0 || !wsClient.isConnected)
            }
        }
        .padding(18)
        .frame(minWidth: 620, maxWidth: 620, minHeight: 440)
        .background(MissionTheme.canvasBg)
        .preferredColorScheme(.dark)
    }
}
