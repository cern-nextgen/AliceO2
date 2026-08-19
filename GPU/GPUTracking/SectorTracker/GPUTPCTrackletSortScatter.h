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

/// \file GPUTPCTrackletSortScatter.h
/// \author Oliver Rietmann

#ifndef GPUTPCTRACKLETSORTSCATTER_H
#define GPUTPCTRACKLETSORTSCATTER_H

#include "GPUTPCDef.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
class GPUTPCTracker;

/**
 * @class GPUTPCTrackletSortScatter
 *
 * Third pass of the tracklet sort: re-derives each tracklet's (LastRow, FirstRow) key,
 * atomically claims its slot from the offsets computed by GPUTPCTrackletSortOffsets
 * (which double as a per-key scatter cursor), and writes the tracklet's original index
 * into that slot of the sorted-index permutation consumed by GPUTPCTrackletSelector.
 */
class GPUTPCTrackletSortScatter : public GPUKernelTemplate
{
 public:
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

#endif // GPUTPCTRACKLETSORTSCATTER_H
