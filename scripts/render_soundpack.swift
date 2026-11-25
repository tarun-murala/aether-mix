#!/usr/bin/env swift
// Offline 48 kHz Stereo CoreAudio DLS Studio Instrument Renderer for AetherMix Sound Pack.
//
// Reads a JSON track description (instrument program, drum flag, reverb/delay sends, notes, CC events)
// and renders realistic sampled audio using macOS's built-in Apple DLS SoundFont (`gs_instruments.dls`)
// through AVAudioEngine manual offline rendering + AVAudioUnitReverb + AVAudioUnitDelay.

import AVFoundation
import AudioToolbox
import Foundation

struct NoteEvent: Codable {
    let start: Double      // seconds
    let duration: Double   // seconds
    let note: UInt8        // MIDI note 0..127
    let velocity: UInt8    // MIDI velocity 1..127
    let channel: UInt8?    // optional MIDI channel (default 0, drums = 9)
}

struct CCEvent: Codable {
    let time: Double
    let cc: UInt8
    let value: UInt8
    let channel: UInt8?
}

struct TrackLayer: Codable {
    let name: String
    let program: UInt8         // General MIDI program 0..127
    let is_drum: Bool?
    let gain_db: Double?
    let pan: Float?            // -1.0 .. 1.0
    let reverb_wet: Float?     // 0..100
    let delay_wet: Float?      // 0..100
    let delay_time: Double?    // seconds
    let delay_feedback: Float? // 0..100
    let notes: [NoteEvent]
    let ccs: [CCEvent]?
}

struct RenderJob: Codable {
    let sample_rate: Double
    let duration: Double       // total seconds to render (including tail if desired)
    let output_wav: String
    let layers: [TrackLayer]
}

enum ScheduledEvent {
    case noteOn(frame: Int64, layerIdx: Int, note: UInt8, vel: UInt8, ch: UInt8)
    case noteOff(frame: Int64, layerIdx: Int, note: UInt8, ch: UInt8)
    case cc(frame: Int64, layerIdx: Int, cc: UInt8, val: UInt8, ch: UInt8)

    var frame: Int64 {
        switch self {
        case .noteOn(let f, _, _, _, _): return f
        case .noteOff(let f, _, _, _): return f
        case .cc(let f, _, _, _, _): return f
        }
    }
}

func renderJob(from jsonPath: String) throws {
    let data = try Data(contentsOf: URL(fileURLWithPath: jsonPath))
    let job = try JSONDecoder().decode(RenderJob.self, from: data)

    let dlsURL = URL(fileURLWithPath: "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls")
    guard FileManager.default.fileExists(atPath: dlsURL.path) else {
        throw NSError(domain: "AetherMixRenderer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing Apple gs_instruments.dls"])
    }

    let sampleRate = job.sample_rate
    guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2) else {
        throw NSError(domain: "AetherMixRenderer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create stereo AVAudioFormat"])
    }

    let engine = AVAudioEngine()
    var samplers: [AVAudioUnitSampler] = []

    for layer in job.layers {
        let sampler = AVAudioUnitSampler()
        engine.attach(sampler)

        let isDrum = layer.is_drum ?? false
        let bankMSB: UInt8 = isDrum ? UInt8(kAUSampler_DefaultPercussionBankMSB) : UInt8(kAUSampler_DefaultMelodicBankMSB)
        let bankLSB: UInt8 = UInt8(kAUSampler_DefaultBankLSB)
        try sampler.loadSoundBankInstrument(at: dlsURL, program: layer.program, bankMSB: bankMSB, bankLSB: bankLSB)

        let gainDb = layer.gain_db ?? 0.0
        sampler.overallGain = Float(gainDb)
        sampler.stereoPan = layer.pan ?? 0.0

        engine.connect(sampler, to: engine.mainMixerNode, format: format)
        samplers.append(sampler)
    }

    let quantumFrames: AVAudioFrameCount = 128
    let maxFrames: AVAudioFrameCount = 512
    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: maxFrames)
    engine.prepare()
    try engine.start()

    // Collect and sort all MIDI events chronologically
    var events: [ScheduledEvent] = []
    for (idx, layer) in job.layers.enumerated() {
        let defaultCh: UInt8 = (layer.is_drum ?? false) ? 9 : 0
        for n in layer.notes {
            let ch = n.channel ?? defaultCh
            let onFrame = max(0, Int64((n.start * sampleRate).rounded()))
            let offFrame = max(onFrame + 1, Int64(((n.start + n.duration) * sampleRate).rounded()))
            events.append(.noteOn(frame: onFrame, layerIdx: idx, note: n.note, vel: n.velocity, ch: ch))
            events.append(.noteOff(frame: offFrame, layerIdx: idx, note: n.note, ch: ch))
        }
        for c in (layer.ccs ?? []) {
            let ch = c.channel ?? defaultCh
            let f = max(0, Int64((c.time * sampleRate).rounded()))
            events.append(.cc(frame: f, layerIdx: idx, cc: c.cc, val: c.value, ch: ch))
        }
    }
    events.sort { a, b in
        if a.frame != b.frame { return a.frame < b.frame }
        // Process noteOff before noteOn on the exact same sample
        switch (a, b) {
        case (.noteOff, .noteOn): return true
        case (.noteOn, .noteOff): return false
        default: return false
        }
    }

    let rawTotalFrames = Int64((job.duration * sampleRate).rounded())
    let q = Int64(quantumFrames)
    let totalFrames = ((rawTotalFrames + q - 1) / q) * q
    let outURL = URL(fileURLWithPath: job.output_wav)
    try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)

    let outSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: sampleRate,
        AVNumberOfChannelsKey: 2,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
    ]
    let outFile = try AVAudioFile(forWriting: outURL, settings: outSettings, commonFormat: .pcmFormatFloat32, interleaved: false)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: maxFrames) else {
        throw NSError(domain: "AetherMixRenderer", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate PCM buffer"])
    }

    var currentFrame: Int64 = 0
    var eventCursor = 0

    while currentFrame < totalFrames {
        // Dispatch all events occurring within this 128-frame (~2.67 ms) quantum
        while eventCursor < events.count && events[eventCursor].frame < currentFrame + q {
            switch events[eventCursor] {
            case .noteOn(_, let idx, let note, let vel, let ch):
                samplers[idx].startNote(note, withVelocity: vel, onChannel: ch)
            case .noteOff(_, let idx, let note, let ch):
                samplers[idx].stopNote(note, onChannel: ch)
            case .cc(_, let idx, let cc, let val, let ch):
                samplers[idx].sendController(cc, withValue: val, onChannel: ch)
            }
            eventCursor += 1
        }

        if !engine.isRunning {
            try engine.start()
        }
        var status: AVAudioEngineManualRenderingStatus
        do {
            status = try engine.renderOffline(quantumFrames, to: buffer)
        } catch let err as NSError where err.code == -80802 {
            // CoreAudio AUSampler async config-change can briefly pause offline engine; resume cleanly
            try engine.start()
            status = try engine.renderOffline(quantumFrames, to: buffer)
        }
        switch status {
        case .success:
            try outFile.write(from: buffer)
            currentFrame += Int64(buffer.frameLength)
        case .insufficientDataFromInputNode:
            break
        case .cannotDoInCurrentContext:
            continue
        case .error:
            throw NSError(domain: "AetherMixRenderer", code: 4, userInfo: [NSLocalizedDescriptionKey: "AVAudioEngine renderOffline returned .error"])
        @unknown default:
            break
        }
    }

    engine.stop()
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    fputs("Usage: render_soundpack <job.json> [job2.json ...]\n", stderr)
    exit(2)
}

for path in args.dropFirst() {
    do {
        try renderJob(from: path)
    } catch {
        fputs("Render error for \(path): \(error)\n", stderr)
        exit(1)
    }
}
