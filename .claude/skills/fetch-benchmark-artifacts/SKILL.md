---
name: fetch-benchmark-artifacts
description: Fetch and interpret the CI artifacts produced by .github/workflows/standalone-benchmark.yml (the GPU/CPU standalone track-reconstruction benchmark matrix: cpu, nvidia-h100, nvidia-rtx6000, nvidia-l40s, amd-mi300x, amd-w7900). Use this whenever the user wants to check a benchmark run's results, look at profiler/occupancy/PMC data from CI, compare a run against a baseline, or asks something like "what happened in the last benchmark run", "pull the ncu/rocprofv3 results from CI", or "did the GPU occupancy change in that run" — even if they don't name the workflow file directly.
---

Fetch the artifacts GitHub Actions produced for a run of `standalone-benchmark.yml`, then read the *profiler* data inside them — not just the overall runtime. The workflow already prints a timing-only summary table to the GitHub Actions step summary; the reason to reach for this skill is everything that table doesn't show: per-kernel occupancy, register pressure, warp/wavefront activity, memory-unit stalls, and the raw `.ncu-rep`/`.nsys-rep`/`.pftrace`/`.perf.data` traces.

**Steps 1-3 (find/download) are owned by the `ci-artifact-fetcher` subagent** — it downloads into `/shared/ci-artifacts/<short-sha>/` instead of an arbitrary dest-dir, and caches by commit. Delegate to that agent to get artifacts onto disk; don't run steps 1-3 yourself from the main thread or another agent. Steps 4-5 (knowing the file shapes, interpreting them) are for whoever reads the files afterward — that part isn't the fetcher agent's job.

## 1. Check `gh` is available and authenticated

```
gh auth status
```

If `gh` itself is missing, don't fail confusingly on the next step — tell the user directly and point them at the fix: on most Linux package managers it's `dnf install gh` / `apt install gh`, or see https://cli.github.com. If it's installed but not authenticated, `gh auth login` needs to happen before anything else here will work.

## 2. Find the run

```
gh run list --workflow=standalone-benchmark.yml --limit 10
gh run list --workflow=standalone-benchmark.yml --branch <branch> --limit 5
```

Default to the most recent run unless the user names a specific run ID, branch, or PR. If the user says something vague like "the last run" or "the one that just finished," use the newest entry — but say which run ID you picked, since artifacts are matrix-job-specific and there are 6 of them per workflow run.

## 3. Download the artifacts

```
gh run download <run-id> -D <dest-dir>
```

This creates one subdirectory per artifact bundle inside `<dest-dir>`, named `<matrix-name>-artifact/` (e.g. `nvidia-h100-artifact/`, `amd-mi300x-artifact/`). If the user only cares about one platform, you can still download everything (it's cheap) and just read the one subdirectory you need — `gh run download` doesn't offer fine-grained filtering by artifact name before downloading.

## 4. Know what's inside each bundle, because the shape differs by platform

| file | present for | what it is |
|---|---|---|
| `standalone_<name>.csv` | all | raw per-run debug dump: one row per kernel per benchmark run, not yet averaged |
| `summary_standalone_<name>.csv` | all | the already-averaged timing table (`name,mean,stdev,count`) — this is what the CI step summary table is built from; read this first if the question is just "did the runtime regress" |
| `<name>_detail.csv` | GPU rows only | the profiler detail data — **format depends on platform, see below** |
| `<name>.ncu-rep` | h100, rtx6000 | binary Nsight Compute report, covers every sector-launch of the profiled kernels in one event |
| `<name>.nsys-rep` | h100, rtx6000, l40s | binary Nsight Systems report |
| `<name>_kernel_stats.csv` | mi300x, w7900 | per-kernel timing/count from `rocprofv3 --stats` (not PMC detail — just durations) |
| `<name>_results.pftrace` | mi300x, w7900 | Perfetto trace — open at ui.perfetto.dev, or in a local Perfetto viewer |
| `<name>.perf.data` | cpu | `perf record` call-graph sample data |

### `<name>_detail.csv` is the same per-kernel-averaged shape on both platforms

**NVIDIA (h100/rtx6000)**: `ncu --print-summary per-kernel --csv` output. Columns include `Kernel Name`, `Section Name`, `Metric Name`, `Metric Unit`, `Minimum`, `Maximum`, `Average`, `Invocations`.

**AMD (mi300x/w7900)**: `.github/scripts/rocprofv3_detail_summary.py` runs in CI right after `rocprofv3`, averaging its raw per-dispatch output into the same shape: `Kernel Name`, `Block Size`, `Grid Size`, `VGPR Count`, `SGPR Count`, `LDS Block Size`, `Scratch Size`, `Counter Name`, `Minimum`, `Maximum`, `Average`, `Stdev`, `Invocations`. The register/resource columns are recorded once per kernel (constant across its dispatches), matching what `ncu`'s sections give for free on the NVIDIA side.

The raw, unaggregated rocprofv3 output (`<name>_detail_counter_collection.csv`) is no longer kept in CI artifacts — `_detail.csv` is the only AMD detail file now. Dead/uncollected counters still show up in it directly: a counter flat at `Minimum == Maximum == 0.0` or `100.0` across all its dispatches means it wasn't actually collected on that GPU/driver/capability combination, not that the kernel was genuinely idle or saturated. A counter that's merely *constant* for some other reason (e.g. `Wavefronts`, a fixed launch-geometry number, not a measurement) isn't a concern — only those two known-dead sentinel values are. `scripts/summarize_rocprofv3_detail.py` still exists for the rare case of a raw `_detail_counter_collection.csv` obtained outside CI (e.g. local hardware testing), but it no longer applies to anything downloaded from a workflow run.

## 5. Interpreting what you find

- **`-rep`/`pftrace` files**: if the right CLI tool (`ncu`, `nsys`, a Perfetto reader) isn't installed locally, say so plainly rather than silently skipping them — these are binary formats, there's no fallback text parse. If `ncu` is available, `ncu --import <file>.ncu-rep --csv --print-summary per-kernel` re-derives the same kind of summary the `_detail.csv` already has, straight from the raw report, which is useful if you need a different `--section` breakdown than whatever this particular CI run happened to request (check `PROFILING_KERNELS_REGEX` and `matrix.metrics` in the workflow file for what was actually captured).
- **Comparing two runs** (e.g. "did this regress against the baseline"): don't re-derive the statistics from scratch — `.github/scripts/csv_to_md.py` already implements the Welch's-t-test comparison CI itself uses (same `mean`/`stdev`/`count` summary CSVs as input). Run it directly: `python3 .github/scripts/csv_to_md.py --runs <N> --baseline <old>/summary_standalone_<name>.csv --current <new>/summary_standalone_<name>.csv`.
- **Low occupancy / low active-lane numbers on `ExtrapolationTracking`** are expected, not a new finding — this repo has an known, already-diagnosed severe under-occupancy issue in that kernel (sparse candidates against a fixed large launch grid). Worth knowing before treating it as a surprise regression.
