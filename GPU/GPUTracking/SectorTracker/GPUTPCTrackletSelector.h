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

/// \file GPUTPCTrackletSelector.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKLETSELECTOR_H
#define GPUTPCTRACKLETSELECTOR_H

#include "GPUTPCDef.h"
#include "GPUTPCHitId.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
class GPUTPCTracker;

// Named functor, not an inline lambda, deliberately -- an inline lambda defined directly inside a
// runKernelBackendTimed<> specialization (GPUReconstructionCUDAKernelsSpecialize.inc) hits an NVCC
// extended-lambda compile error there; every comparator actually used from inside one of those
// specializations in this codebase is a named functor, never a raw lambda (see
// GPUTPCGMMergerSortTracks_comp, GPUTPCExtrapolationCandidateRowIndex_comp for the same pattern).
// Sorts tracklets primarily by length (LastRow - FirstRow), the quantity that directly bounds
// GPUTPCTrackletSelector::Thread<0>'s per-tracklet select-loop trip count, with LastRow as a
// secondary tie-break key.
struct GPUTPCTrackletSort_comp {
  const uint32_t* const mFirstRow;
  const uint32_t* const mLastRow;
  GPUhd() GPUTPCTrackletSort_comp(uint32_t* firstRow, uint32_t* lastRow) : mFirstRow(firstRow), mLastRow(lastRow) {}
  GPUd() bool operator()(int32_t a, int32_t b) const
  {
    const uint32_t lenA = mLastRow[a] - mFirstRow[a], lenB = mLastRow[b] - mFirstRow[b];
    return lenA != lenB ? lenA < lenB : mLastRow[a] < mLastRow[b];
  }
};

/**
 * @class GPUTPCTrackletSelector
 *
 */
class GPUTPCTrackletSelector : public GPUKernelTemplate
{
 public:
  // "sort" is a real dispatched kernel (needed so runKernel<> routes through runKernelBackendTimed,
  // which GPUReconstructionCUDAKernelsSpecialize.inc intercepts on CUDA/HIP to call sortOnDevice
  // instead -- see GPUChainTrackingSectorTracker.cxx for why that call can't live inline there).
  // Thread<sort>'s own body is the fallback for backends without that specialization active (e.g.
  // CPU-only builds): a real sort via sortDeviceDynamic, same #ifndef GPUCA_SPECIALIZE_THRUST_SORTS
  // pattern as GPUTPCGMMerger::SortTracks/SortTracksQPt.
  enum K { defaultKernel = 0,
           sort = 1 };

  struct GPUSharedMemory {
    int32_t mItr0;          // index of the first track in the block
    int32_t mNThreadsTotal; // total n threads
    int32_t mNTracklets;    // n of tracklets
    int32_t mReserved;      // for alignment reasons
    static_assert(GPUTPCGeometry::NROWS >= GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE);
    GPUTPCHitId mHits[GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE][GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCTrackletSelector)];
  };

  typedef GPUconstantref() GPUTPCTracker processorType;
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::TPCSectorTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker);
};
} // namespace o2::gpu

#endif // GPUTPCTRACKLETSELECTOR_H
