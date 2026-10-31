---
name: run-standalone-benchmark
description: Build the ALICE O2 standalone benchmark in non-deterministic mode and run the 50kHz benchmark. Use when asked to benchmark, measure performance, or run a non-deterministic build of the O2 standalone.
---

**Owned by the `build-verify` subagent** — delegate to that agent rather than running this skill directly from the main thread or another agent.

[Run these 4 commands first, exactly as written, and use their output where each is named below. Run them one per call, or all in one Bash call joined with &&.]
1. `module load ninja/fortran-v1.11.1.g9-15 Vc/1.4.5-10 boost/v1.83.0-alice2-57 fmt/11.1.2-14 CMake/v3.31.6-10 ms_gsl/4.2.1-3 TBB/v2022.3.0-3 ROOT/v6-36-04-alice9-15 ONNXRuntime/v1.22.0-71 GLFW/3.3.2-25`
2. `cmake -B /scratch/build -DENABLE_CUDA=1 -DENABLE_HIP=0 -DCUDA_COMPUTETARGET=90 -DCMAKE_INSTALL_PREFIX=/scratch /root/alice/O2/GPU/GPUTracking/Standalone > /scratch/cmake-configure.log`
3. `cmake --build /scratch/build --target install -j8 > /scratch/cmake-build.log`
4. `cd /scratch && /scratch/ca -e 50kHz -g --seed 0 --memSize 9000000000 --sync --debug 1 > /scratch/run-benchmarks.log`

Build and run the benchmark configuration for performance measurement. The logs of the cmake commands are written to `/scratch/cmake-configure.log` and `/scratch/cmake-build.log`. The benchmark times and logs are written to `/scratch/run-benchmarks.log`. It should end with `CUDA Uninitialized` and contain no `std::bad_alloc`, `Aborted`, or `CUDA Error`. If it crashes, don't retry blindly, but report the exact error and ask before re-running. The results are not verified, as the benchmark is non-deterministic. For verification, see the `run-standalone-verify` skill. Use the following defaults for the parameters: `memSize=9000000000`, `events=50kHz`, `debug=1`. You can override these defaults by providing arguments to the skill.

## Load environment modules

[output of command 1, `module load ninja/fortran-v1.11.1.g9-15 Vc/1.4.5-10 boost/v1.83.0-alice2-57 fmt/11.1.2-14 CMake/v3.31.6-10 ms_gsl/4.2.1-3 TBB/v2022.3.0-3 ROOT/v6-36-04-alice9-15 ONNXRuntime/v1.22.0-71 GLFW/3.3.2-25`]

## CMake configure (non-deterministic mode)

[output of command 2, `cmake -B /scratch/build -DENABLE_CUDA=1 -DENABLE_HIP=0 -DCUDA_COMPUTETARGET=90 -DCMAKE_INSTALL_PREFIX=/scratch /root/alice/O2/GPU/GPUTracking/Standalone > /scratch/cmake-configure.log`]

## CMake build

[output of command 3, `cmake --build /scratch/build --target install -j8 > /scratch/cmake-build.log`]

## Run (non-deterministic mode for benchmarks)

[output of command 4, `cd /scratch && /scratch/ca -e 50kHz -g --seed 0 --memSize 9000000000 --sync --debug 1 > /scratch/run-benchmarks.log`]
