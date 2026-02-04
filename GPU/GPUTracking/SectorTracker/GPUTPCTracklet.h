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

/// \file GPUTPCTracklet.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKLET_H
#define GPUTPCTRACKLET_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCDef.h"
#include "MemLayout.h"

namespace o2::gpu
{
/**
 * @class GPUTPCTracklet
 *
 * The class describes the reconstructed TPC track candidate.
 * The class is dedicated for internal use by the GPUTPCTracker algorithm.
 */
template <template <class> class F>
class GPUTPCTrackletSkeleton
{
 public:
  MEMLAYOUT_APPLY_UNARY(mFirstRow, mLastRow, mParam, mHitWeight, mFirstHit)
  MEMLAYOUT_APPLY_BINARY(GPUTPCTrackletSkeleton, MEMLAYOUT_EXPAND(mFirstRow), MEMLAYOUT_EXPAND(mLastRow), MEMLAYOUT_EXPAND(mParam), MEMLAYOUT_EXPAND(mHitWeight), MEMLAYOUT_EXPAND(mFirstHit))

//#if !defined(GPUCA_GPUCODE)
  //GPUTPCTrackletSkeleton() : mFirstRow(0), mLastRow(0), mParam(), mHitWeight(0), mFirstHit(0) {};
//#endif //! GPUCA_GPUCODE

  GPUhd() int32_t FirstRow() const { return mFirstRow; }
  GPUhd() int32_t LastRow() const { return mLastRow; }
  GPUhd() int32_t HitWeight() const { return mHitWeight; }
  GPUhd() uint32_t FirstHit() const { return mFirstHit; }
  GPUhd() MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> Param() const { return mParam; }

  GPUhd() void SetFirstRow(int32_t v) { mFirstRow = v; }
  GPUhd() void SetLastRow(int32_t v) { mLastRow = v; }
  GPUhd() void SetFirstHit(uint32_t v) { mFirstHit = v; }
  GPUhd() void SetParam(MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> v) { mParam = v; }
  GPUhd() void SetHitWeight(const int32_t w) { mHitWeight = w; }

// private:
  F<int32_t> mFirstRow;           // first TPC row // TODO: We can use smaller data format here!
  F<int32_t> mLastRow;            // last TPC row
  MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, F> mParam; // tracklet parameters
  F<int32_t> mHitWeight;          // Hit Weight of Tracklet
  F<uint32_t> mFirstHit;          // first hit in row hit array
};

} // namespace o2::gpu

#endif // GPUTPCTRACKLET_H
