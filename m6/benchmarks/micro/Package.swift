// swift-tools-version: 5.9
import PackageDescription

// Benchmark harness for "Bandwidth, Not FLOPS: FFT Kernels, Matrix Units and SAR Imaging on
// Apple M6". Metal kernels are compiled at run time from ../../src (the harness locates the m6/
// directory by walking up to src/common/fft_device.h, or from $FFT_REPO_ROOT).
let package = Package(
    name: "M6Bench",
    platforms: [.macOS(.v13)],
    targets: [
        // SME2 batched FFT. Canonical source: ../../src/cpu/sme_fft.{c,h}.
        // -mcpu=apple-m4, not -march=armv9: Apple cores have no non-streaming SVE.
        // -fno-modules: SwiftPM's dependency scan otherwise rejects arm_sme.h.
        .target(
            name: "CSMEFFT",
            path: "CSMEFFT",
            sources: ["shim.c"],
            publicHeadersPath: "include",
            cSettings: [.unsafeFlags(["-mcpu=apple-m4", "-O3", "-fno-modules"])]
        ),
        .executableTarget(
            name: "M6Bench",
            dependencies: ["CSMEFFT"],
            path: "Sources",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShadersGraph"),
                .linkedFramework("Accelerate"),
            ]
        ),
    ]
)
