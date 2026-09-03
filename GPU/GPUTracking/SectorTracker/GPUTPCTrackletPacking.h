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

/// \file GPUTPCTrackletPacking.h

#ifndef GPUTPCTRACKLETPACKING_H
#define GPUTPCTRACKLETPACKING_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
class GPUTPCTracker;

/**
 * @class GPUTPCTrackletPacking
 *
 * Diagnostic prototype: assigns each tracklet a "lane" such that no two tracklets sharing a lane
 * overlap in row range, using close to the minimum number of lanes (greedy interval
 * partitioning: process tracklets in FirstRow order, reuse any lane freed by a tracklet whose
 * LastRow has already passed, otherwise open a new one -- provably optimal in lane count, since
 * interval graphs are perfect graphs). Lane assignments aren't consumed downstream yet; this
 * exists to measure the cost of the "sweep" sub-kernel's single-thread dependency chain before
 * committing to it as the basis for a row-major, coalesced tracklet storage layout.
 *
 * "sweep" additionally builds a per-lane singly linked list of the tracklets assigned to each
 * lane, in the same (LastRow -> FirstRow) order GPUTPCTrackletSelector's per-tracklet loop
 * already walks -- see the walk-direction discussion in GPUTPCTrackletPacking.cxx. "gather"
 * transposes TrackletRowHits() into a row-major [row][lane] layout so GPUTPCTrackletSelector's
 * selectPacked sub-kernel can read a lane's hit for the current row with coalesced access across
 * a warp of lanes, instead of a scattered per-tracklet index.
 */
class GPUTPCTrackletPacking : public GPUKernelTemplate
{
 public:
  enum K { defaultKernel = 0,
           count = 1,
           offsets = 2,
           scatter = 3,
           sweep = 4,
           gather = 5 };

  // Sentinel for "no tracklet"/"no lane" in TrackletLane(), LaneListNext()/Head()/Tail().
  static constexpr uint32_t NoLane = 0xFFFFFFFFu;

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

#endif // GPUTPCTRACKLETPACKING_H
