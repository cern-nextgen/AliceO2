// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCExtrapolationTracking.h
/// \author David Rohr

#ifndef GPUTPCEXTRAPOLATIONTRACKING_H
#define GPUTPCEXTRAPOLATIONTRACKING_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCExtrapolationCandidate.h"

namespace o2::gpu
{
class GPUTPCTracker;

// Named functor, not an inline lambda, deliberately -- matching GPUTPCGMMergerSortTracks_comp's
// own pattern (GPUTPCGMMerger.cxx). An inline lambda defined directly inside a
// runKernelBackendTimed<> specialization (GPUReconstructionCUDAKernelsSpecialize.inc) hit an NVCC
// extended-lambda compile error there; every comparator actually used from inside one of those
// specializations in this codebase is a named functor, never a raw lambda.
struct GPUTPCExtrapolationCandidateRowIndex_comp {
  const int32_t* const mRowIndex;
  GPUhd() GPUTPCExtrapolationCandidateRowIndex_comp(int32_t* rowIndex) : mRowIndex(rowIndex) {}
  GPUd() bool operator()(int32_t a, int32_t b) const { return mRowIndex[a] < mRowIndex[b]; }
};

class GPUTPCExtrapolationTracking : public GPUKernelTemplate
{
 public:
  // "sort" is a real dispatched kernel (needed so runKernel<> routes through runKernelBackendTimed,
  // which GPUReconstructionCUDAKernelsSpecialize.inc intercepts on CUDA/HIP to call sortOnDevice
  // instead -- see GPUChainTrackingSectorTracker.cxx for why that call can't live inline there).
  // Thread<sort>'s own body below is the fallback for backends where that specialization isn't
  // active (e.g. CPU-only builds): it still performs a real sort via sortDeviceDynamic, same
  // #ifndef GPUCA_SPECIALIZE_THRUST_SORTS pattern as GPUTPCGMMerger::SortTracks/SortTracksQPt --
  // needed so the extrapolated-track append order matches a CUDA/HIP build's sortOnDevice order
  // bit-for-bit, since nothing re-sorts Tracks() after ExtrapolationTracking appends to it.
  enum K { defaultKernel = 0,
           sort = 1 };

  struct GPUSharedMemory {
    GPUCA_SHARED_STORAGE(GPUTPCRow mRows[GPUTPCGeometry::NROWS]);
  };

  typedef GPUconstantref() GPUTPCTracker processorType;
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::TPCSectorTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker);

  GPUd() static int32_t ExtrapolationTrackingSectorOrder(int32_t iSector);
  GPUd() static void ExtrapolationTrackingSectorLeftRight(uint32_t iSector, uint32_t& left, uint32_t& right);

 private:
  GPUd() static int32_t PerformExtrapolationTrackingRun(GPUTPCTracker& tracker, GPUsharedref() GPUSharedMemory& smem, MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> sourceParam, int32_t sourceLocalTrackId, uint32_t sourceSector, int32_t rowIndex, float angle, int32_t direction);
  GPUd() static void ConsumeExtrapolationCandidates(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, uint32_t sourceSector, GPUsharedref() GPUSharedMemory& smem, GPUTPCTracker& sectorTarget, MemLayout::wrapper<GPUTPCExtrapolationCandidateSkeleton, MemLayout::pointer> cand, GPUglobalref() const int32_t* sortedIndex, uint32_t nCand, float angle);
};

class GPUTPCExtrapolationTrackingCopyNumbers : public GPUKernelTemplate
{
 public:
  typedef GPUconstantref() GPUTPCTracker processorType;
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::TPCSectorTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int32_t iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker, int32_t n);
};

} // namespace o2::gpu

#endif // GPUTPCTRACKLETCONSTRUCTOR_H
