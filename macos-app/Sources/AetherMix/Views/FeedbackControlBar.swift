import AVFoundation
import Speech
import SwiftUI

/// Bottom Live Feedback & Critique Control Dock:
/// - Oversized 👍 Good Mix (`Cmd+Shift+Up`) and 👎 Undo & Penalize (`Cmd+Shift+Down`),
///   strictly suppressed while the critique `TextField` has focus.
/// - Quick-chip presets for 1-click volunteer critiques.
/// - Keyboard critique input (primary) + optional Apple Speech dictation hook (rehearsal use).
public struct FeedbackControlBar: View {
    public let wsClient: EngineWebSocketClient

    @State private var critiqueText: String = ""
    @State private var isDictating: Bool = false
    @State private var dictationStatus: String?
    @State private var speechHelper = OptionalSpeechDictationHelper()
    @FocusState private var isCritiqueFocused: Bool

    private let quickChips: [String] = [
        "Keys too loud",
        "More lead vocal",
        "Less reverb on pastor",
        "Cut BVs during speech",
        "Warm up guitars",
        "Drums slightly softer"
    ]

    public init(wsClient: EngineWebSocketClient) {
        self.wsClient = wsClient
    }

    public var body: some View {
        VStack(spacing: 8) {
            // Quick-chip presets row
            HStack(spacing: 6) {
                Text("QUICK CRITIQUE:")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(quickChips, id: \.self) { chip in
                            Button {
                                submitCritique(chip)
                            } label: {
                                Text(chip)
                                    .font(.system(size: 10, weight: .semibold))
                                    .foregroundStyle(MissionTheme.textPrimary)
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 4)
                                    .background(MissionTheme.cardBg, in: Capsule())
                                    .overlay(Capsule().stroke(MissionTheme.border, lineWidth: 1))
                            }
                            .buttonStyle(.plain)
                            .disabled(!wsClient.isConnected)
                        }
                    }
                }

                if let status = dictationStatus {
                    Text(status)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(MissionTheme.amber)
                }
            }

            // Main Dock Row: Oversized 👍 / 👎 + Text Critique + Optional Speech Dictation
            HStack(spacing: 12) {
                // 👍 Good Mix (Cmd+Shift+Up, suppressed when TextField focused)
                Button {
                    triggerThumb(.up)
                } label: {
                    HStack(spacing: 8) {
                        Text("👍")
                            .font(.system(size: 18))
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Good Mix")
                                .font(.system(size: 12, weight: .bold))
                            Text("⌘⇧↑")
                                .font(.system(size: 9, design: .monospaced))
                                .opacity(0.8)
                        }
                    }
                    .foregroundStyle(Color.black)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(MissionTheme.emerald, in: RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .disabled(!wsClient.isConnected)
                .keyboardShortcut(
                    isCritiqueFocused ? nil : KeyboardShortcut(.upArrow, modifiers: [.command, .shift])
                )
                .help("Reinforce current armed mix offsets (+1 sample; suppressed while typing)")

                // 👎 Undo & Penalize (Cmd+Shift+Down, suppressed when TextField focused)
                Button {
                    triggerThumb(.down)
                } label: {
                    HStack(spacing: 8) {
                        Text("👎")
                            .font(.system(size: 18))
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Undo & Penalize")
                                .font(.system(size: 12, weight: .bold))
                            Text("⌘⇧↓  (2× = T-15s)")
                                .font(.system(size: 9, design: .monospaced))
                                .opacity(0.85)
                        }
                    }
                    .foregroundStyle(Color.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(MissionTheme.crimson, in: RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .disabled(!wsClient.isConnected)
                .keyboardShortcut(
                    isCritiqueFocused ? nil : KeyboardShortcut(.downArrow, modifiers: [.command, .shift])
                )
                .help("Roll back non-touch-locked armed channels to T-5s (press twice within 5s for T-15s); suppressed while typing")

                // Primary Keyboard Critique Input
                HStack(spacing: 8) {
                    Image(systemName: "text.bubble.fill")
                        .foregroundStyle(MissionTheme.cyanTarget)

                    TextField(
                        "Type mix critique for supervisory LLM (e.g. 'keys -2 dB during prayer')…",
                        text: $critiqueText
                    )
                    .textFieldStyle(.plain)
                    .font(.system(size: 12))
                    .foregroundStyle(MissionTheme.textPrimary)
                    .focused($isCritiqueFocused)
                    .onSubmit {
                        submitCritique(critiqueText)
                    }

                    // Optional Apple Speech Dictation Hook (Rehearsal use)
                    Button {
                        toggleDictation()
                    } label: {
                        Image(systemName: isDictating ? "mic.fill" : "mic")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(isDictating ? MissionTheme.crimson : MissionTheme.textSecondary)
                            .padding(6)
                            .background(
                                isDictating ? MissionTheme.crimson.opacity(0.18) : MissionTheme.panelBg,
                                in: Circle()
                            )
                    }
                    .buttonStyle(.plain)
                    .help("Optional rehearsal dictation via Apple Speech (keyboard typing is primary)")

                    Button {
                        submitCritique(critiqueText)
                    } label: {
                        Text("Send")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(Color.black)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(MissionTheme.cyanTarget, in: RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                    .disabled(!wsClient.isConnected || critiqueText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(MissionTheme.canvasBg, in: RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isCritiqueFocused ? MissionTheme.cyanTarget : MissionTheme.border, lineWidth: 1.2)
                )
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(MissionTheme.panelBg)
        .overlay(alignment: .top) {
            Divider().overlay(MissionTheme.border)
        }
    }

    private func triggerThumb(_ thumb: FeedbackThumb) {
        guard !isCritiqueFocused else { return }
        Task { await wsClient.sendFeedbackThumb(thumb) }
    }

    private func submitCritique(_ raw: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        critiqueText = ""
        isCritiqueFocused = false
        Task { await wsClient.submitCritique(trimmed) }
    }

    private func toggleDictation() {
        if isDictating {
            speechHelper.stop()
            isDictating = false
            dictationStatus = nil
        } else {
            speechHelper.start(
                onTranscript: { text in
                    self.critiqueText = text
                },
                onError: { message in
                    self.isDictating = false
                    self.dictationStatus = message
                }
            )
            isDictating = true
            dictationStatus = "Listening (rehearsal dictation)…"
        }
    }
}

// MARK: - Optional Apple Speech Dictation Helper (Rehearsal Use)

@MainActor
private final class OptionalSpeechDictationHelper {
    private var audioEngine: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    func start(
        onTranscript: @escaping @MainActor (String) -> Void,
        onError: @escaping @MainActor (String) -> Void
    ) {
        SFSpeechRecognizer.requestAuthorization { status in
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard status == .authorized else {
                    onError("Speech permission unavailable — use keyboard input")
                    return
                }
                guard let recognizer = SFSpeechRecognizer(), recognizer.isAvailable else {
                    onError("On-device Speech recognizer unavailable")
                    return
                }
                do {
                    let engine = AVAudioEngine()
                    let req = SFSpeechAudioBufferRecognitionRequest()
                    req.shouldReportPartialResults = true
                    req.requiresOnDeviceRecognition = true

                    let inputNode = engine.inputNode
                    let format = inputNode.outputFormat(forBus: 0)
                    inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
                        req.append(buffer)
                    }
                    engine.prepare()
                    try engine.start()

                    self.audioEngine = engine
                    self.request = req
                    self.task = recognizer.recognitionTask(with: req) { result, error in
                        if let text = result?.bestTranscription.formattedString {
                            Task { @MainActor in onTranscript(text) }
                        }
                        if let error {
                            Task { @MainActor in onError(error.localizedDescription) }
                        }
                    }
                } catch {
                    onError("Dictation error: \(error.localizedDescription)")
                }
            }
        }
    }

    func stop() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        audioEngine = nil
        request = nil
        task = nil
    }
}
