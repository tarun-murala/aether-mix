#!/usr/bin/env bash
# Build release AetherMix.app (native Swift 6 / SwiftUI + bundled runtime + 5-tune Sound Pack)
# and package a drag-to-Applications macOS DMG installer in dist/AetherMix-<version>-arm64.dmg.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${AETHERMIX_VERSION:-0.5.1}"
ARCH="$(uname -m)"
DIST_DIR="${REPO_ROOT}/dist"
STAGE_DIR="${DIST_DIR}/dmg_stage"
APP_DIR="${DIST_DIR}/AetherMix.app"
DMG_PATH="${DIST_DIR}/AetherMix-${VERSION}-${ARCH}.dmg"
SWIFT_PKG="${REPO_ROOT}/macos-app"
CACHE_DIR="${CLANG_MODULE_CACHE_PATH:-/tmp/clang-module-cache}"

echo "====================================================================="
echo "  Building AetherMix v${VERSION} (${ARCH}) macOS App Bundle & DMG"
echo "====================================================================="

# 1. Build release native SwiftUI binary
echo "==> [1/6] Compiling native Swift 6 / SwiftUI release binary..."
CLANG_MODULE_CACHE_PATH="${CACHE_DIR}" swift build \
  -c release \
  --disable-sandbox \
  --package-path "${SWIFT_PKG}"

BIN_PATH="${SWIFT_PKG}/.build/release/AetherMix"
if [[ ! -x "${BIN_PATH}" ]]; then
  BIN_PATH="$(CLANG_MODULE_CACHE_PATH="${CACHE_DIR}" swift build -c release --disable-sandbox --package-path "${SWIFT_PKG}" --show-bin-path)/AetherMix"
fi
if [[ ! -x "${BIN_PATH}" ]]; then
  echo "ERROR: Built AetherMix release binary not found at ${BIN_PATH}" >&2
  exit 1
fi
echo "    [ok] Release binary: ${BIN_PATH} ($(du -h "${BIN_PATH}" | awk '{print $1}'))"

# 2. Generate clean native macOS App Icon (.icns) using Swift CoreGraphics + iconutil
echo "==> [2/6] Generating AetherMix.icns app icon..."
rm -rf "${APP_DIR}" "${STAGE_DIR}"
mkdir -p "${APP_DIR}/Contents/MacOS" "${APP_DIR}/Contents/Resources" "${STAGE_DIR}"

ICNS_OUT="${APP_DIR}/Contents/Resources/AetherMix.icns"

CLANG_MODULE_CACHE_PATH="${CACHE_DIR}" swift - "${ICNS_OUT}" <<'SWIFT_EOF'
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let outICNS = CommandLine.arguments[1]
let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

func renderPNG(px: Int) -> Data? {
    guard let ctx = CGContext(
        data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    let scale = CGFloat(px) / 1024.0
    ctx.scaleBy(x: scale, y: scale)

    let rect = CGRect(x: 40, y: 40, width: 944, height: 944)
    let bgPath = CGPath(roundedRect: rect, cornerWidth: 210, cornerHeight: 210, transform: nil)
    ctx.saveGState()
    ctx.addPath(bgPath)
    ctx.clip()

    let colors = [
        CGColor(red: 0.06, green: 0.09, blue: 0.18, alpha: 1.0),
        CGColor(red: 0.10, green: 0.17, blue: 0.34, alpha: 1.0),
        CGColor(red: 0.05, green: 0.24, blue: 0.30, alpha: 1.0),
    ] as CFArray
    if let grad = CGGradient(colorsSpace: colorSpace, colors: colors, locations: [0.0, 0.55, 1.0]) {
        ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: 1024), end: CGPoint(x: 1024, y: 0), options: [])
    }

    let faderX: [CGFloat] = [232, 372, 512, 652, 792]
    let faderLevel: [CGFloat] = [0.68, 0.82, 0.48, 0.74, 0.58]
    let trackBottom: CGFloat = 210
    let trackHeight: CGFloat = 604

    for i in 0..<5 {
        let x = faderX[i]
        let levelY = trackBottom + trackHeight * faderLevel[i]

        ctx.setFillColor(CGColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.14))
        ctx.addPath(CGPath(roundedRect: CGRect(x: x - 14, y: trackBottom, width: 28, height: trackHeight), cornerWidth: 14, cornerHeight: 14, transform: nil))
        ctx.fillPath()

        let fillColor = (i == 1)
            ? CGColor(red: 0.22, green: 0.92, blue: 0.72, alpha: 0.95)
            : CGColor(red: 0.24, green: 0.76, blue: 0.98, alpha: 0.90)
        ctx.setFillColor(fillColor)
        ctx.addPath(CGPath(roundedRect: CGRect(x: x - 14, y: trackBottom, width: 28, height: levelY - trackBottom), cornerWidth: 14, cornerHeight: 14, transform: nil))
        ctx.fillPath()

        ctx.setFillColor(CGColor(red: 0.96, green: 0.98, blue: 1.0, alpha: 1.0))
        ctx.addPath(CGPath(roundedRect: CGRect(x: x - 44, y: levelY - 26, width: 88, height: 52), cornerWidth: 16, cornerHeight: 16, transform: nil))
        ctx.fillPath()

        ctx.setFillColor(CGColor(red: 0.08, green: 0.14, blue: 0.25, alpha: 0.85))
        ctx.fill(CGRect(x: x - 28, y: levelY - 4, width: 56, height: 8))
    }

    ctx.restoreGState()
    ctx.addPath(bgPath)
    ctx.setStrokeColor(CGColor(red: 0.45, green: 0.86, blue: 1.0, alpha: 0.35))
    ctx.setLineWidth(8)
    ctx.strokePath()

    guard let img = ctx.makeImage() else { return nil }
    let mutable = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(mutable as CFMutableData, UTType.png.identifier as CFString, 1, nil) else {
        return nil
    }
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else { return nil }
    return mutable as Data
}

// Standard Apple ICNS PNG chunk types (16px .. 1024px)
let chunks: [(String, Int)] = [
    ("icp4", 16), ("ic11", 32), ("icp5", 32), ("ic12", 64),
    ("ic07", 128), ("ic13", 256), ("ic08", 256), ("ic14", 512),
    ("ic09", 512), ("ic10", 1024),
]

var body = Data()
for (fourCC, px) in chunks {
    guard let png = renderPNG(px: px), let tag = fourCC.data(using: .ascii) else { exit(2) }
    var lenBE = UInt32(8 + png.count).bigEndian
    body.append(tag)
    body.append(Data(bytes: &lenBE, count: 4))
    body.append(png)
}

var icns = Data("icns".utf8)
var totalBE = UInt32(8 + body.count).bigEndian
icns.append(Data(bytes: &totalBE, count: 4))
icns.append(body)
try icns.write(to: URL(fileURLWithPath: outICNS))
SWIFT_EOF
echo "    [ok] Generated AetherMix.icns ($(du -h "${ICNS_OUT}" | awk '{print $1}'))"

# 3. Assemble AetherMix.app bundle + Info.plist + runtime payload
echo "==> [3/6] Bundling AetherMix.app + Python engine + Bridge + 5-Tune Sound Pack..."
cp "${BIN_PATH}" "${APP_DIR}/Contents/MacOS/AetherMix"
chmod +x "${APP_DIR}/Contents/MacOS/AetherMix"

cat > "${APP_DIR}/Contents/Info.plist" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>AetherMix</string>
    <key>CFBundleExecutable</key>
    <string>AetherMix</string>
    <key>CFBundleIconFile</key>
    <string>AetherMix</string>
    <key>CFBundleIdentifier</key>
    <string>com.aethermix.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>AetherMix</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>AetherMix reads multi-channel pre-fader live audio from BlackHole 16ch / your audio interface to run real-time mixing, ducking, and anti-feedback protection.</string>
</dict>
</plist>
PLIST_EOF

RUNTIME_DEST="${APP_DIR}/Contents/Resources/runtime"
mkdir -p \
  "${RUNTIME_DEST}/bridge" \
  "${RUNTIME_DEST}/engine/core" \
  "${RUNTIME_DEST}/engine/schemas" \
  "${RUNTIME_DEST}/scripts" \
  "${RUNTIME_DEST}/assets" \
  "${RUNTIME_DEST}/docs" \
  "${RUNTIME_DEST}/models"

# Copy clean Python runtime (no __pycache__, .venv, or telemetry DBs)
cp "${REPO_ROOT}/bridge/__init__.py" "${REPO_ROOT}/bridge/midi_bridge_daemon.py" "${RUNTIME_DEST}/bridge/"
cp "${REPO_ROOT}/engine/__init__.py" "${REPO_ROOT}/engine/main.py" "${REPO_ROOT}/engine/pyproject.toml" "${REPO_ROOT}/engine/uv.lock" "${RUNTIME_DEST}/engine/"
cp "${REPO_ROOT}/engine/core/"*.py "${RUNTIME_DEST}/engine/core/"
cp "${REPO_ROOT}/engine/schemas/"*.json "${RUNTIME_DEST}/engine/schemas/"

# Copy operational & verification scripts
for s in \
  install_prereqs.sh \
  check_setup.py \
  play_mainstage_pack.py \
  run_live_mainstage_test.py \
  verify_audio_midi.py \
  phase0_spike.py \
  bootstrap_mac.sh \
  com.aethermix.midibridge.plist; do
  if [[ -f "${REPO_ROOT}/scripts/${s}" ]]; then
    cp "${REPO_ROOT}/scripts/${s}" "${RUNTIME_DEST}/scripts/"
  fi
done
chmod +x "${RUNTIME_DEST}/scripts/"*.sh "${RUNTIME_DEST}/scripts/"*.py

# Copy packaged 5-tune Sound Pack
if [[ -d "${REPO_ROOT}/assets/soundpack" ]]; then
  cp -R "${REPO_ROOT}/assets/soundpack" "${RUNTIME_DEST}/assets/soundpack"
else
  echo "WARNING: assets/soundpack not found; run 'make soundpack' first to bundle tunes." >&2
fi

# Bundle Silero VAD ONNX if cached locally (only ~2 MB, enables offline VAD on target Mac)
if [[ -f "${HOME}/.aethermix/models/silero_vad.onnx" ]]; then
  cp "${HOME}/.aethermix/models/silero_vad.onnx" "${RUNTIME_DEST}/models/silero_vad.onnx"
  echo "    [ok] Bundled silero_vad.onnx into app runtime"
fi

# Copy documentation
cp "${REPO_ROOT}/docs/INSTALL_AND_VERIFY.md" "${REPO_ROOT}/docs/MAINSTAGE_LIVE_TEST.md" "${RUNTIME_DEST}/docs/"

# 4. Ad-hoc codesign AetherMix.app
echo "==> [4/6] Ad-hoc signing AetherMix.app..."
codesign --force --deep --sign - "${APP_DIR}"
codesign --verify --deep --strict "${APP_DIR}"
echo "    [ok] Code signature verified"

# 5. Populate drag-and-drop DMG staging folder
echo "==> [5/6] Populating DMG staging directory with installer helpers & setup guide..."
cp -R "${APP_DIR}" "${STAGE_DIR}/AetherMix.app"
ln -s /Applications "${STAGE_DIR}/Applications"
cp "${REPO_ROOT}/docs/INSTALL_AND_VERIFY.md" "${STAGE_DIR}/SETUP_GUIDE.md"

cat > "${STAGE_DIR}/1 - Install Prerequisites & Ollama.command" <<'CMD1_EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "/Applications/AetherMix.app/Contents/Resources/runtime/scripts/install_prereqs.sh" ]]; then
  exec "/Applications/AetherMix.app/Contents/Resources/runtime/scripts/install_prereqs.sh"
else
  echo "Tip: Drag AetherMix.app into Applications first for the best experience."
  exec "${SCRIPT_DIR}/AetherMix.app/Contents/Resources/runtime/scripts/install_prereqs.sh"
fi
CMD1_EOF

cat > "${STAGE_DIR}/2 - Run Setup Verification Doctor.command" <<'CMD2_EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCTOR="/Applications/AetherMix.app/Contents/Resources/runtime/scripts/check_setup.py"
if [[ ! -f "${DOCTOR}" ]]; then
  DOCTOR="${SCRIPT_DIR}/AetherMix.app/Contents/Resources/runtime/scripts/check_setup.py"
fi
python3 "${DOCTOR}" --test-ollama || true
echo ""
read -r -p "Press Enter to close..."
CMD2_EOF

cat > "${STAGE_DIR}/3 - Start Live MainStage Sound Pack.command" <<'CMD3_EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${PATH}"
PYTHON_BIN="${HOME}/.aethermix/venv/bin/python"
PLAYER="/Applications/AetherMix.app/Contents/Resources/runtime/scripts/play_mainstage_pack.py"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: AetherMix Python runtime not installed yet."
  echo "Double-click '1 - Install Prerequisites & Ollama.command' first!"
  read -r -p "Press Enter to close..."
  exit 1
fi
exec "${PYTHON_BIN}" "${PLAYER}"
CMD3_EOF

chmod +x "${STAGE_DIR}/"*.command

# 6. Build compressed macOS DMG
echo "==> [6/6] Creating compressed DMG: ${DMG_PATH}..."
rm -f "${DMG_PATH}"
hdiutil create \
  -volname "AetherMix ${VERSION}" \
  -srcfolder "${STAGE_DIR}" \
  -ov \
  -format UDZO \
  -imagekey zlib-level=9 \
  "${DMG_PATH}"

rm -rf "${STAGE_DIR}"

echo ""
echo "====================================================================="
echo "  DMG Build Complete!"
echo "  Path:   ${DMG_PATH}"
echo "  Size:   $(du -h "${DMG_PATH}" | awk '{print $1}')"
echo "  SHA256: $(shasum -a 256 "${DMG_PATH}" | awk '{print $1}')"
echo "====================================================================="
