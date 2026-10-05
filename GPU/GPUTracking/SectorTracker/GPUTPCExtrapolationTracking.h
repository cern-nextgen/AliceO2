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
  struct GPUSharedMemory {
    GPUCA_SHARED_STORAGE(GPUTPCRow mRows[GPUTPCGeometry::NROWS]);
  };

  typedef GPUconstantref() GPUTPCTracker processorType;
  GPUhdi() constexpr static gpudatatypes::RecoStep GetRecoStep() { return gpudatatypes::RecoStep::TPCSectorTracking; }
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcTrackers;
  }
  template <int32_t iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& tracker);

  GPUd() static int32_t ExtrapolationTrackingSectorOrder(int32_t iSector);
  GPUd() static void ExtrapolationTrackingSectorLeftRight(uint32_t iSector, uint32_t& left, uint32_t& right);

 private:
  GPUd() static int32_t PerformExtrapolationTrackingRun(GPUTPCTracker& tracker, GPUsharedref() GPUSharedMemory& smem, MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> sourceParam, int32_t sourceLocalTrackId, uint32_t sourceSector, int32_t rowIndex, float angle, int32_t direction);
  GPUd() static void ConsumeExtrapolationCandidates(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, uint32_t sourceSector, GPUsharedref() GPUSharedMemory& smem, GPUTPCTracker& sectorTarget, MemLayout::wrapper<GPUTPCExtrapolationCandidateSkeleton, MemLayout::pointer> cand, uint32_t nCand, float angle, int32_t direction);
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
