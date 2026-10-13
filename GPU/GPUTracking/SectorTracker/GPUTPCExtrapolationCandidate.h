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

/// \file GPUTPCExtrapolationCandidate.h
/// \author Oliver Rietmann

#ifndef GPUTPCEXTRAPOLATIONCANDIDATE_H
#define GPUTPCEXTRAPOLATIONCANDIDATE_H

#include "GPUTPCBaseTrackParam.h"
#include "MemLayout.h"

namespace o2::gpu
{

// A candidate carries a value snapshot of the track's Param()/LocalTrackId(), captured at
// GPUTPCTrackletSelector write-time, instead of an index into Tracks() to re-dereference later --
// Tracks() gets physically reordered by GPUTPCSectorDebugSortKernels in deterministic mode, which
// would invalidate a stored slot index by the time GPUTPCExtrapolationTracking consumes it.
template <template <class> class F>
class GPUTPCExtrapolationCandidateSkeleton
{
 public:
  MEMLAYOUT_APPLY_UNARY(mParam, mLocalTrackId, mRowIndex, mDirection)
  MEMLAYOUT_APPLY_BINARY(GPUTPCExtrapolationCandidateSkeleton, MEMLAYOUT_EXPAND(mParam), MEMLAYOUT_EXPAND(mLocalTrackId), MEMLAYOUT_EXPAND(mRowIndex), MEMLAYOUT_EXPAND(mDirection))

  GPUhd() MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> Param() const { return mParam; }
  GPUhd() int32_t LocalTrackId() const { return mLocalTrackId; }
  GPUhd() int32_t RowIndex() const { return mRowIndex; }
  // +1 (candidate's last hit near the sector's upper row edge) or -1 (first hit near the lower
  // edge) -- which direction PerformExtrapolationTrackingRun's row-walk continues in once it
  // enters the neighbor sector. Previously implied by which of 4 separate arrays a candidate was
  // pushed into (LowerTo*/UpperTo*); now stored per-candidate since Lower/Upper candidates headed
  // to the same neighbor share one array (see ExtrapCandToLeft()/ExtrapCandToRight() on
  // GPUTPCTracker) -- merging them doesn't change which single consumer reads the array (still
  // exactly one neighbor sector either way), only collapses an axis that didn't need its own array.
  GPUhd() int8_t Direction() const { return mDirection; }

  GPUhd() void SetParam(MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> v) { mParam = v; }
  GPUhd() void SetLocalTrackId(int32_t v) { mLocalTrackId = v; }
  GPUhd() void SetRowIndex(int32_t v) { mRowIndex = v; }
  GPUhd() void SetDirection(int8_t v) { mDirection = v; }

  MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, F> mParam; // nested skeleton -- flattened recursively by ApplyRecursive
  F<int32_t> mLocalTrackId;
  F<int32_t> mRowIndex;
  F<int8_t> mDirection;
};

// Named, overridable layout flag -- same pattern as GPUTPCTrackLayout (GPUTPCTrack.h) and
// GPUTPCTrackletLayout (GPUTPCTracklet.h). Storage goes through this constant, not a hardcoded
// Flag::soa, so the AoS/SoA choice can be revisited in one place later without touching call sites.
constexpr MemLayout::Flag GPUTPCExtrapolationCandidateLayout = MemLayout::Flag::soa;

} // namespace o2::gpu

#endif // GPUTPCEXTRAPOLATIONCANDIDATE_H
