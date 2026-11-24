// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AetherMix",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "AetherMixCore",
            targets: ["AetherMixCore"]
        ),
        .executable(
            name: "AetherMix",
            targets: ["AetherMix"]
        )
    ],
    targets: [
        .target(
            name: "AetherMixCore",
            path: "Sources/AetherMix",
            exclude: ["App"],
            sources: ["Models", "Services", "Views"]
        ),
        .executableTarget(
            name: "AetherMix",
            dependencies: ["AetherMixCore"],
            path: "Sources/AetherMix/App",
            sources: ["AetherMixApp.swift"]
        ),
        .testTarget(
            name: "AetherMixTests",
            dependencies: ["AetherMixCore"],
            path: "Tests/AetherMixTests"
        )
    ]
)
