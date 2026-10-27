---
name: run-standalone-verify
description: Build the ALICE O2 standalone benchmark in deterministic mode and verify results against the reference output. Use when asked to verify, check correctness, or run a deterministic build of the O2 standalone benchmark.
---

[Run these 5 commands first, exactly as written, and use their output where each is named below. Run them one per call, or all in one Bash call joined with &&.]
1. `module load ninja/fortran-v1.11.1.g9-15 Vc/1.4.5-10 boost/v1.83.0-alice2-57 fmt/11.1.2-14 CMake/v3.31.6-10 ms_gsl/4.2.1-3 TBB/v2022.3.0-3 ROOT/v6-36-04-alice9-15 ONNXRuntime/v1.22.0-71 GLFW/3.3.2-25`
2. `cmake -B /scratch/build -DENABLE_CUDA=1 -DENABLE_HIP=0 -DCUDA_COMPUTETARGET=90 -DGPUCA_DETERMINISTIC_MODE=GPU -DCMAKE_INSTALL_PREFIX=/scratch /root/alice/O2/GPU/GPUTracking/Standalone > /scratch/cmake-configure.log`
3. `cmake --build /scratch/build --target install -j8 > /scratch/cmake-build.log`
4. `cd /scratch && /scratch/ca -e o2-simple -g --seed 0 --memSize 6000000000 --sync --runs 1 --RTCenable --PROCdeterministicGPUReconstruction 1 --RTCoptSpecialCode 1 --debug 6 > /scratch/run-deterministic.log`
5. `cmp /scratch/GPU.out /root/alice/o2-simple-GPU.out || true`

Build and run the deterministic configuration, then compare output against the reference file to confirm correctness. The logs of the cmake commands are written to `/scratch/cmake-configure.log` and `/scratch/cmake-build.log`. The logs of the run are written to `/scratch/run-deterministic.log`. They should end with `CUDA Uninitialized` and contain no `std::bad_alloc`, `Aborted`, or `CUDA Error`. If it crashes, don't retry blindly, but report the exact error and ask before re-running. The reconstructed tracks are written to `/scratch/GPU.out`, and the reference output file can be set with the argument `reference` of this skill. The two files should be identical. Use the following defaults for the parameters: `memSize=6000000000`, `events=o2-simple`, `reference=/root/alice/o2-simple-GPU.out`. The argument `events` (input data) and `reference` (reference output for given input data) need to match each other. You can override these defaults by providing arguments to the skill. For benchmarking, see the `run-standalone-benchmark` skill.

## Load environment modules

[output of command 1, `module load ninja/fortran-v1.11.1.g9-15 Vc/1.4.5-10 boost/v1.83.0-alice2-57 fmt/11.1.2-14 CMake/v3.31.6-10 ms_gsl/4.2.1-3 TBB/v2022.3.0-3 ROOT/v6-36-04-alice9-15 ONNXRuntime/v1.22.0-71 GLFW/3.3.2-25`]

## CMake configure (deterministic mode)

[output of command 2, `cmake -B /scratch/build -DENABLE_CUDA=1 -DENABLE_HIP=0 -DCUDA_COMPUTETARGET=90 -DGPUCA_DETERMINISTIC_MODE=GPU -DCMAKE_INSTALL_PREFIX=/scratch /root/alice/O2/GPU/GPUTracking/Standalone > /scratch/cmake-configure.log`]

## CMake build

[output of command 3, `cmake --build /scratch/build --target install -j8 > /scratch/cmake-build.log`]

## Run (deterministic mode to verify results)

[output of command 4, `cd /scratch && /scratch/ca -e o2-simple -g --seed 0 --memSize 6000000000 --sync --runs 1 --RTCenable --PROCdeterministicGPUReconstruction 1 --RTCoptSpecialCode 1 --debug 6 > /scratch/run-deterministic.log`]

## Verify results (deterministic mode)

[output of command 5, `cmp /scratch/GPU.out /root/alice/o2-simple-GPU.out || true`]
