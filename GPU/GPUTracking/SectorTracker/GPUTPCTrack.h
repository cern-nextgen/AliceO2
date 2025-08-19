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

/// \file GPUTPCTrack.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCTRACK_H
#define GPUTPCTRACK_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCDef.h"
// #include "wrapper.h"

namespace o2::gpu
{
/**
 * @class GPUTPCTrack
 *
 * The class describes the [partially] reconstructed TPC track [candidate].
 * The class is dedicated for internal use by the GPUTPCTracker algorithm.
 * The track parameters at both ends are stored separately in the GPUTPCEndPoint class
 */
template <template <class> class F>
class GPUTPCTrackSkeleton
{
 public:
#if !defined(GPUCA_GPUCODE)
  GPUTPCTrackSkeleton();
  ~GPUTPCTrackSkeleton() = default;
#endif //! GPUCA_GPUCODE

  GPUhd() int32_t NHits() const { return mNHits; }
  GPUhd() int32_t LocalTrackId() const { return mLocalTrackId; }
  GPUhd() int32_t FirstHitID() const { return mFirstHitID; }
  GPUhd() const GPUTPCBaseTrackParam& Param() const { return mParam; }

  GPUhd() void SetNHits(int32_t v) { mNHits = v; }
  GPUhd() void SetLocalTrackId(int32_t v) { mLocalTrackId = v; }
  GPUhd() void SetFirstHitID(int32_t v) { mFirstHitID = v; }
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

 private:
  int32_t mFirstHitID;         // index of the first track cell in the track->cell pointer array
  int32_t mNHits;              // number of track cells
  int32_t mLocalTrackId;       // Id of local track this extrapolated track belongs to, index of this track itself if it is a local track
  GPUTPCBaseTrackParam mParam; // track parameters

 private:
};

#if !defined(GPUCA_GPUCODE)
template <>
GPUTPCTrackSkeleton<wrapper::value>::GPUTPCTrackSkeleton() : mFirstHitID(0), mNHits(0), mLocalTrackId(-1), mParam()
{
}
#endif //! GPUCA_GPUCODE

using GPUTPCTrack = GPUTPCTrackSkeleton<wrapper::value>;

} // namespace o2::gpu

#endif // GPUTPCTRACK_H
