import Foundation

// =============================================================================
// Portable source resolution for the device-side FFT primitive.
//
// The dx benchmarks compile fft_device.h + the kernels at runtime via
// MTLDevice.makeLibrary(source:), which cannot resolve relative #includes, so we
// concatenate the files. To stay machine-independent we locate the repo root at
// runtime (NO hardcoded absolute paths):
//   1. $FFT_REPO_ROOT if set and valid;
//   2. walk up from the executable's directory and the CWD looking for the
//      marker file src/common/fft_device.h.
// =============================================================================

enum FFTSourceError: Error, CustomStringConvertible {
    case repoNotFound
    case fileNotFound(String)
    var description: String {
        switch self {
        case .repoNotFound:
            return "Could not locate the fft repo root (marker '\(FFTDeviceSource.marker)'). " +
                   "Set FFT_REPO_ROOT=/path/to/fft or run from within the repository."
        case .fileNotFound(let p):
            return "Required source file not found: \(p)"
        }
    }
}

enum FFTDeviceSource {
    static let marker = "src/common/fft_device.h"

    /// Locate the repository root, or nil if it cannot be found.
    static func repoRoot() -> String? {
        let fm = FileManager.default
        func hasMarker(_ dir: String) -> Bool {
            fm.fileExists(atPath: (dir as NSString).appendingPathComponent(marker))
        }
        if let env = ProcessInfo.processInfo.environment["FFT_REPO_ROOT"], hasMarker(env) { return env }
        var starts = [fm.currentDirectoryPath]
        let exec = CommandLine.arguments.first ?? ""
        if exec.contains("/") { starts.append((exec as NSString).deletingLastPathComponent) }
        for start in starts {
            var dir = (start as NSString).standardizingPath
            for _ in 0..<16 {
                if hasMarker(dir) { return dir }
                let parent = (dir as NSString).deletingLastPathComponent
                if parent == dir || parent.isEmpty { break }
                dir = parent
            }
        }
        return nil
    }

    /// Absolute path to a repo-relative file, validated to exist.
    static func path(_ rel: String) throws -> String {
        guard let root = repoRoot() else { throw FFTSourceError.repoNotFound }
        let p = (root as NSString).appendingPathComponent(rel)
        guard FileManager.default.fileExists(atPath: p) else { throw FFTSourceError.fileNotFound(p) }
        return p
    }

    /// fft_device.h + the radix-8 baseline + the kernels, concatenated (with
    /// local #includes stripped) for runtime makeLibrary(source:).
    static func combinedSource() throws -> String {
        let header   = try String(contentsOfFile: path("src/common/fft_device.h"), encoding: .utf8)
        let baseline = try String(contentsOfFile: path("src/metal/fft_4096_batched.metal"), encoding: .utf8)
        let kernels  = try String(contentsOfFile: path("src/metal/fft_device_kernels.metal"), encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.contains("#include \"") }
            .joined(separator: "\n")
        return header + "\n" + baseline + "\n" + kernels
    }
}

/// Convenience for the CLI benchmarks: resolve+concatenate, or print a clear
/// diagnostic and exit(2) (a missing repo is an operator error, not a crash).
func loadCombinedSource() -> String {
    do { return try FFTDeviceSource.combinedSource() }
    catch { FileHandle.standardError.write(Data("FATAL: \(error)\n".utf8)); exit(2) }
}
