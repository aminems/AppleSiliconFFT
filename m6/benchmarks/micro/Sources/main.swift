import Foundation

// M6Bench <mode>: one mode per measurement in the paper (see ../../README.md).
let args = CommandLine.arguments
let mode = args.count >= 2 ? args[1] : ""
switch mode {
case "m6-bw": runM6Bandwidth()                  // Table II, GPU peak (Table I)
case "m6-batch": runM6BatchSweep()              // Table III
case "m6-cpu": runM6CPU()                       // vDSP per-thread placement (Sec. IV-B)
case "m6-cpu-scale": runM6CPUScale()            // Table IV
case "m6-landscape":                            // Fig. 1, Sec. V
    if #available(macOS 14.0, *) { runM6Landscape() }
case "m6-onepass":                              // Table V
    if #available(macOS 14.0, *) { runM6OnePass() }
case "m6-onepass-check": runM6OnePassCheck()    // one-pass correctness incl. two-window path
case "m6-fp16":                                 // Table VI, Fig. 2
    if #available(macOS 14.0, *) { runM6FP16() }
case "m6-mma-peak":                             // Table VII (top)
    if #available(macOS 26.0, *) { runM6MMAPeak() }
case "m6-tensor-fft":                           // Table VII (bottom)
    if #available(macOS 26.0, *) { runM6TensorFFT() }
case "m6-sar": runM6SAR()                       // Sec. X, Table IX
default:
    print("usage: M6Bench m6-bw | m6-batch | m6-cpu | m6-cpu-scale | m6-landscape | m6-onepass | " +
          "m6-onepass-check | m6-fp16 | m6-mma-peak | m6-tensor-fft | m6-sar")
    exit(1)
}
