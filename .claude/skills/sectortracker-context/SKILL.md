---
name: sectortracker-context
description: Orientation packet for the three sector-tracker GPU kernels we focus most of our optimization work on — TrackletConstructor, TrackletSelector, ExtrapolationTracking. Source file locations, the relevant sections of David Rohr's thesis (with page numbers), launch-config/register-pressure pointers, and where accumulated findings live. Load this before diagnosing a profiler finding or designing a fix for anything in GPU/GPUTracking/SectorTracker, instead of re-locating these each time.
---

Scope: `GPU/GPUTracking/SectorTracker/`, specifically `GPUTPCTrackletConstructor`, `GPUTPCTrackletSelector`, `GPUTPCExtrapolationTracking` (branch name/kernel family: `ExtrapolationTracking`). This is not the whole codebase — just the recurring focus area, so this packet is worth keeping current rather than rediscovering per question.

## Source files

- `GPU/GPUTracking/SectorTracker/GPUTPCTrackletConstructor.{h,cxx}`
- `GPU/GPUTracking/SectorTracker/GPUTPCTrackletSelector.{h,cxx}`
- `GPU/GPUTracking/SectorTracker/GPUTPCExtrapolationTracking.{h,cxx}` — the baseline/`ngt-ci-pipeline` version. The `ExtrapolationTracking` feature branch additionally adds `GPUTPCExtrapolationCandidate.h` (dense sorted-candidate arrays), not present on `ngt-ci-pipeline`/`dev`.
- `GPU/GPUTracking/SectorTracker/GPUTPCTracker.{h,cxx}` — shared per-sector state all three kernels read/write.
- `GPU/GPUTracking/SectorTracker/GPUTPCTracklet.h`
- `GPU/GPUTracking/Global/GPUChainTrackingSectorTracker.cxx` — where these kernels get scheduled/chained.
- `GPU/Common/GPUCommonAlgorithm.h`, `GPU/Common/GPUCommonAlgorithmThrust.h` — shared sort primitives (used by TrackletSelector's pre-sort and the ExtrapolationTracking candidate sort).
- `GPU/GPUTracking/Base/cuda/GPUReconstructionCUDAKernelsSpecialize.inc` — per-kernel CUDA specialization hooks.
- `GPU/GPUTracking/Global/GPUErrorCodes.h`

### Launch config / register pressure

- `GPU/GPUTracking/Definitions/Parameters/GPUParameters.csv` — per-architecture `[blockSize, minBlocksPerCU]` launch config, one row per kernel (rows for all three kernels above), plus a dedicated `AMD_EUS_PER_CU` row. Columns (header row): `Architecture,default,default_cpu,MI100,VEGA,TAHITI,TESLA,FERMI,PASCAL,KEPLER,AMPERE,TURING,HOPPER,ADA,OPENCL,RDNA,MI210,BLACKWELL,MI300,METAL`. Re-read fresh each time rather than trusting a cached copy of the values — this file changes as we tune launch configs.
- `GPU/GPUTracking/Base/GPUReconstructionKernelMacros.h` — turns the CSV's `[blockSize, minBlocksPerCU]` into the actual `__launch_bounds__(...)` on each kernel. NVIDIA passes the CSV value straight through; AMD/HIP converts it via `GPUCA_MIN_WARPS_PER_EU` (HIP's second `__launch_bounds__` argument means "min warps per EU", not "min blocks per SM").
- `GPU/GPUTracking/Base/hip/GPUReconstructionHIPGenRTC.cxx` (and the CUDA sibling `GPUReconstructionCUDAGenRTC.cxx`) — where `GPUCA_WARP_SIZE`/`par_AMD_EUS_PER_CU` get injected at HIP RTC time.

## Thesis — David Rohr, `/shared/backupO2/Diplomarbeit_David_Rohr.pdf` (127 PDF pages)

**Don't re-render the PDF for the sections below — read [thesis-excerpts.md](/shared/gpu-notes/thesis-excerpts.md) instead**, a one-time distillation of exactly these sections in plain text, with notes on which parts are timeless theory vs. obsolete GT200-era hardware numbers. Only go back to the raw PDF for a section not covered there, or to double-check a direct quote.

**Printed page numbers run 8 behind PDF page numbers** throughout (verified against multiple anchors) — add 8 to a printed-TOC page number to get the PDF page to `Read` with `pages:`. The thesis calls sectors **"slices"** throughout (stated explicitly on PDF p. 14) — same concept as our "sector", different name; expect that term when reading it.

| Section | Topic | Printed pp. | PDF pp. |
|---|---|---|---|
| §2.3.4 | Tracklet Constructor [IV] — algorithm/theory | 25–26 | **33–34** |
| §2.3.5 | Tracklet Selector [V] — algorithm/theory | 27–28 | **35–36** |
| §3.7 (+.1/.2) | Coalescing Rules (global + shared memory) | 35–38 | **43–46** |
| §4.2.3 | Row Synchronicity | 42 | **50** |
| §6.1.2 | Parallel Threads / Register Usage | 55–56 | **63–64** |
| §6.2.2 | Tracklet Constructor — optimizations (the big one) | 62–79 | **70–87** |
| §6.2.3 | Tracklet Selector — optimizations | 80 | **88** |
| §6.3.1 | Memory Optimizations | 81–83 | **89–91** |

**Gap**: the thesis does not cover `ExtrapolationTracking` as we know it — closest relatives are the *intra-tracklet* forward/backward extrapolation inside §2.3.4 (PDF p. 33–34) and a one-paragraph forward-looking "Track Merger" note in §10.7 (PDF p. 118). The actual cross-sector candidate-extrapolation algorithm postdates this thesis; for it, rely on current source + our own accumulated notes, not the thesis.

## Accumulated findings

Running, per-kernel notes live in `/shared/gpu-notes/`: [TrackletConstructor.md](/shared/gpu-notes/TrackletConstructor.md), [TrackletSelector.md](/shared/gpu-notes/TrackletSelector.md), [ExtrapolationTracking.md](/shared/gpu-notes/ExtrapolationTracking.md). Check the relevant file first before re-deriving a diagnosis from scratch, and append a new dated entry (don't overwrite history) after reaching a durable finding. The original combined doc, `/shared/ci-artifacts/tracklet-kernel-optimization-summary.md`, stays as the archival source these were split from.

Known pitfalls already in memory (check these before proposing a design): persistent-lane row-sweep tracklet packing was tried and rejected; single-thread GPU kernels doing dependent-chain bookkeeping are catastrophically slow (no warp-level latency hiding).
