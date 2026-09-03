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
#include "MemLayout.h"

namespace o2::gpu
{
constexpr MemLayout::Flag GPUTPCTrackLayout = MemLayout::Flag::aos;

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
    MEMLAYOUT_APPLY_UNARY(mFirstHitID, mNHits, mLocalTrackId, mParam)
    MEMLAYOUT_APPLY_BINARY(GPUTPCTrackSkeleton, MEMLAYOUT_EXPAND(mFirstHitID), MEMLAYOUT_EXPAND(mNHits), MEMLAYOUT_EXPAND(mLocalTrackId), MEMLAYOUT_EXPAND(mParam))

//#if !defined(GPUCA_GPUCODE)
//  GPUTPCTrack() : mFirstHitID(0), mNHits(0), mLocalTrackId(-1), mParam() {}
//  ~GPUTPCTrackSkeleton() = default;
//#endif //! GPUCA_GPUCODE

  GPUhd() int32_t NHits() const { return mNHits; }
  GPUhd() int32_t LocalTrackId() const { return mLocalTrackId; }
  GPUhd() int32_t FirstHitID() const { return mFirstHitID; }
  GPUhd() MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> Param() const { return mParam; }

  GPUhd() void SetNHits(int32_t v) { mNHits = v; }
  GPUhd() void SetLocalTrackId(int32_t v) { mLocalTrackId = v; }
  GPUhd() void SetFirstHitID(int32_t v) { mFirstHitID = v; }
  GPUhd() void SetParam(MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> v) { mParam = v; }

 //private:
  F<int32_t> mFirstHitID;         // index of the first track cell in the track->cell pointer array
  F<int32_t> mNHits;              // number of track cells
  F<int32_t> mLocalTrackId;       // Id of local track this extrapolated track belongs to, index of this track itself if it is a local track
  MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, F> mParam; // track parameters
};

// Needed for sorting
constexpr void swap(GPUTPCTrackSkeleton<MemLayout::reference> a, GPUTPCTrackSkeleton<MemLayout::reference> b) {
    std::swap(a.mFirstHitID, b.mFirstHitID);
    std::swap(a.mNHits, b.mNHits);
    std::swap(a.mLocalTrackId, b.mLocalTrackId);
    swap(a.mParam, b.mParam);
}

using GPUTPCTrack = MemLayout::wrapper<GPUTPCTrackSkeleton, MemLayout::value>;

} // namespace o2::gpu

namespace std {

template<class T> class iterator_traits;
class random_access_iterator_tag;

template<>
struct iterator_traits<MemLayout::wrapper<o2::gpu::GPUTPCTrackSkeleton, MemLayout::pointer>> {
    using iterator_category = random_access_iterator_tag;
    using difference_type = MemLayout::ptrdiff_t;
    using value_type = MemLayout::wrapper<o2::gpu::GPUTPCTrackSkeleton, MemLayout::value>;
    using pointer = void;
    using reference = MemLayout::wrapper<o2::gpu::GPUTPCTrackSkeleton, MemLayout::reference>;
};

}

#endif // GPUTPCTRACK_H
