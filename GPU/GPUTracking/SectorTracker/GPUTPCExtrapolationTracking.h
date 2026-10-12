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

class GPUTPCExtrapolationTracking : public GPUKernelTemplate
{
 public:
  enum K { defaultKernel = 0,
           count = 1,
           offsets = 2,
           scatter = 3 };

  // Secondary sort-key bucket count for the candidate sort below (see count()/scatter() in the
  // .cxx) -- purely to spread the (RowIndex, QPt-bucket) histogram beyond RowIndex()'s own narrow
  // range and avoid the ~50% atomic-contention regression a too-narrow key caused for the sibling
  // GPUTPCTrackletSelector sort (fix-tracklet-sort-offsets branch history). Tunable; revisit with
  // real per-bucket occupancy data (see GPUTPCExtrapolationTracking.cxx's ExtrapCandSortKey).
  static constexpr int32_t QPtBuckets = 8;
  static constexpr uint32_t NSortKeys = GPUTPCGeometry::NROWS * QPtBuckets;

  // Fixed thread count the "offsets" sub-kernel runs with (single block). NSortKeys here is much
  // smaller than GPUTPCTrackletSelector's own NROWS*NROWS, so fewer threads keep each thread's
  // histogram chunk nontrivial.
  static constexpr int32_t OffsetsThreads = 128;

  struct GPUSharedMemory {
    GPUCA_SHARED_STORAGE(GPUTPCRow mRows[GPUTPCGeometry::NROWS]);
    uint32_t mOffsetsScan[OffsetsThreads]; // per-thread chunk totals / running block-wide scan, used only by "offsets"
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
  GPUd() static void ConsumeExtrapolationCandidates(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, uint32_t sourceSector, GPUsharedref() GPUSharedMemory& smem, GPUTPCTracker& sectorTarget, MemLayout::wrapper<GPUTPCExtrapolationCandidateSkeleton, MemLayout::pointer> cand, GPUglobalref() const uint32_t* sortedIndex, uint32_t nCand, float angle, int32_t direction);
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
