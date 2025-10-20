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
// #include "wrapper.h"

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
#if !defined(GPUCA_GPUCODE)
  //GPUTPCTrackletSkeleton() : mFirstRow(0), mLastRow(0), mParam(), mHitWeight(0), mFirstHit(0) {};
#endif //! GPUCA_GPUCODE

  GPUhd() int32_t FirstRow() const { return mFirstRow; }
  GPUhd() int32_t LastRow() const { return mLastRow; }
  GPUhd() int32_t HitWeight() const { return mHitWeight; }
  GPUhd() uint32_t FirstHit() const { return mFirstHit; }
  GPUhd() GPUTPCBaseTrackParamSkeleton<wrapper::const_reference> Param() const { return mParam; }

  GPUhd() void SetFirstRow(int32_t v) { mFirstRow = v; }
  GPUhd() void SetLastRow(int32_t v) { mLastRow = v; }
  GPUhd() void SetFirstHit(uint32_t v) { mFirstHit = v; }
  GPUhd() void SetParam(GPUTPCBaseTrackParamSkeleton<wrapper::const_reference> v) {
    mParam.mX = v.mX;
    mParam.mC[0] = v.mC[0];
    mParam.mC[1] = v.mC[1];
    mParam.mC[2] = v.mC[2];
    mParam.mC[3] = v.mC[3];
    mParam.mC[4] = v.mC[4];
    mParam.mC[5] = v.mC[5];
    mParam.mC[6] = v.mC[6];
    mParam.mC[7] = v.mC[7];
    mParam.mC[8] = v.mC[8];
    mParam.mC[9] = v.mC[9];
    mParam.mC[10] = v.mC[10];
    mParam.mC[11] = v.mC[11];
    mParam.mC[12] = v.mC[12];
    mParam.mC[13] = v.mC[13];
    mParam.mC[14] = v.mC[14];
    mParam.mZOffset = v.mZOffset;
    mParam.mP[0] = v.mP[0];
    mParam.mP[1] = v.mP[1];
    mParam.mP[2] = v.mP[2];
    mParam.mP[3] = v.mP[3];
    mParam.mP[4] = v.mP[4];
  }
  GPUhd() void SetHitWeight(const int32_t w) { mHitWeight = w; }

// private:
  F<int32_t> mFirstRow;           // first TPC row // TODO: We can use smaller data format here!
  F<int32_t> mLastRow;            // last TPC row
  GPUTPCBaseTrackParamSkeleton<F> mParam; // tracklet parameters
  F<int32_t> mHitWeight;          // Hit Weight of Tracklet
  F<uint32_t> mFirstHit;          // first hit in row hit array
};

using GPUTPCTracklet = GPUTPCTrackletSkeleton<wrapper::value>;
using GPUTPCTracklet_reference = GPUTPCTrackletSkeleton<wrapper::reference>;

} // namespace o2::gpu

#endif // GPUTPCTRACKLET_H
