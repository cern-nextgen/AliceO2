---
name: run-cleanup
description: Clean the build files of O2 (deterministic and non-deterministic). Use when you want a new clean build.
---

**Owned by the `build-verify` subagent** — delegate to that agent rather than running this skill directly from the main thread or another agent.

Clean the build files generated with the `run-standalone-verify` and `run-standalone-benchmark` skills. After that, you can use those skills to build and run the benchmark again from a clean directory.

## Clean up

[run this first, exactly as written, and use its output: `rm -rf /scratch/build /scratch/ca /scratch/include /scratch/lib* /scratch/makefile /scratch/share /scratch/GPU.out /scratch/CPU.out /scratch/run-benchmarks.log /scratch/run-deterministic.log /scratch/cmake-configure.log /scratch/cmake-build.log`]
