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

/// \file GPUTPCTrackletSelector.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCTrackletSelector.h"
#include "GPUTPCTrack.h"
#include "GPUTPCTracker.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCTracklet.h"
#include "GPUTPCTrackletPacking.h"
#include "GPUCommonMath.h"
#include "MemLayout.h"

using namespace o2::gpu;

// Tracklets are physically rearranged into (LastRow, FirstRow) order ahead of selection, via a
// counting sort rather than a comparison sort: both keys are bounded to [0, NROWS), so the
// combined key only spans NROWS*NROWS distinct values -- small enough that a histogram + prefix
// sum + scatter beats a general O(N log N) comparison sort. Unlike a sorted *index* permutation,
// physically moving the tracklet records (SoA field by SoA field, via MemLayout's wrapper
// assignment) means the "select" sub-kernel below can read TrackletsSorted()[i] directly, with i
// running sequentially across a warp -- restoring coalesced SoA access there, instead of forcing
// every field read through a scattered per-lane index into Tracklets().
// The three passes below share TrackletSortKeyCount() as histogram, then in-place prefix-sum
// offsets, then per-key scatter cursor.

template <>
GPUdii() void GPUTPCTrackletSelector::Thread<GPUTPCTrackletSelector::count>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  const uint32_t nTracklets = *tracker.NTracklets();
  for (uint32_t itr = iBlock * nThreads + iThread; itr < nTracklets; itr += nBlocks * nThreads) {
    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::const_reference_restrict> tracklet = tracker.Tracklets()[itr];
    const uint32_t key = tracklet.LastRow() * GPUTPCGeometry::NROWS + tracklet.FirstRow();
    CAMath::AtomicAdd(&tracker.TrackletSortKeyCount()[key], 1u);
  }
}

template <>
GPUdii() void GPUTPCTrackletSelector::Thread<GPUTPCTrackletSelector::offsets>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  if (iBlock != 0) {
    return;
  }
  // Parallel block-wide exclusive prefix sum over the NROWS*NROWS-key histogram, launched with
  // exactly OffsetsThreads threads in one block (see GPUChainTrackingSectorTracker.cxx). A single
  // serial thread doing this (the previous implementation) means ~23k *dependent* global-memory
  // round trips -- each atomic's write value depends on the previous one's result, so nothing can
  // overlap -- which measured at ~138ms on a partitioned H100, dwarfing every other pass in this
  // sort. Splitting the key range into OffsetsThreads chunks, prefix-summing each chunk locally,
  // then combining the per-chunk totals with a small (log2(OffsetsThreads)-step) shared-memory
  // scan removes that dependency chain almost entirely.
  GPUglobalref() GPUAtomic(uint32_t)* GPUrestrict() keyCount = tracker.TrackletSortKeyCount();
  constexpr uint32_t nKeys = GPUTPCGeometry::NROWS * GPUTPCGeometry::NROWS;
  const uint32_t chunk = (nKeys + nThreads - 1) / nThreads;
  const uint32_t begin = CAMath::Min(nKeys, (uint32_t)iThread * chunk);
  const uint32_t end = CAMath::Min(nKeys, begin + chunk);

  // Phase 1: each thread sums the raw counts in its own chunk (plain reads -- the "count" pass
  // that produced them ran to completion in an earlier kernel launch, so there is no concurrent
  // writer left to race with).
  uint32_t local = 0;
  for (uint32_t key = begin; key < end; key++) {
    local += keyCount[key];
  }
  s.mOffsetsScan[iThread] = local;
  GPUbarrier();

  // Phase 2: turn the per-thread chunk totals into an exclusive prefix sum, in shared memory
  // (Hillis-Steele scan: read-then-barrier-then-write-then-barrier avoids any read/write race
  // between a slot's own update and a neighbor reading its pre-update value this round).
  for (uint32_t offset = 1; offset < (uint32_t)nThreads; offset <<= 1) {
    const uint32_t addend = ((uint32_t)iThread >= offset) ? s.mOffsetsScan[iThread - offset] : 0;
    GPUbarrier();
    s.mOffsetsScan[iThread] += addend;
    GPUbarrier();
  }
  const uint32_t base = s.mOffsetsScan[iThread] - local; // inclusive -> exclusive

  // Phase 3: re-derive each key's count and write back its final offset (doubling as the
  // scatter kernel's per-key cursor), now fully in parallel across chunks.
  uint32_t running = base;
  for (uint32_t key = begin; key < end; key++) {
    const uint32_t c = keyCount[key];
    keyCount[key] = running;
    running += c;
  }
}

template <>
GPUdii() void GPUTPCTrackletSelector::Thread<GPUTPCTrackletSelector::scatter>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  const uint32_t nTracklets = *tracker.NTracklets();
  for (uint32_t itr = iBlock * nThreads + iThread; itr < nTracklets; itr += nBlocks * nThreads) {
    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::const_reference_restrict> tracklet = tracker.Tracklets()[itr];
    const uint32_t key = tracklet.LastRow() * GPUTPCGeometry::NROWS + tracklet.FirstRow();
    const uint32_t pos = CAMath::AtomicAdd(&tracker.TrackletSortKeyCount()[key], 1u);
    tracker.TrackletsSorted()[pos] = tracklet; // physical, per-SoA-field copy (MemLayout::wrapper assignment)
  }
}

template <>
GPUdii() void GPUTPCTrackletSelector::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  // select best tracklets and kill clones

  if (iThread == 0) {
    s.mNTracklets = *tracker.NTracklets();
    s.mNThreadsTotal = nThreads * nBlocks;
    s.mItr0 = nThreads * iBlock;
  }
  GPUbarrier();

  GPUTPCHitId trackHits[GPUTPCGeometry::NROWS - GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE];
  const float maxSharedFrac = tracker.Param().rec.tpc.trackletMaxSharedFraction;

  for (int32_t i = s.mItr0 + iThread; i < s.mNTracklets; i += s.mNThreadsTotal) {
    GPUbarrierWarp();

    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::reference_restrict> tracklet = tracker.TrackletsSorted()[i];

    int32_t firstRow = tracklet.FirstRow();
    int32_t lastRow = tracklet.LastRow();

    const int32_t w = tracklet.HitWeight();

    uint32_t gap = 0;
    uint32_t nShared = 0;
    uint32_t nHits = 0;
    const uint32_t minHits = tracker.Param().rec.tpc.minNClustersTrackSeed == -1 ? tracker.Param().tpcMinHitsB5(tracklet.Param().QPt() * tracker.Param().qptB5Scaler) : tracker.Param().rec.tpc.minNClustersTrackSeed;
    const uint32_t sharingMinNorm = minHits * tracker.Param().rec.tpc.trackletMinSharedNormFactor;
    const float maxSharedNorm = maxSharedFrac * sharingMinNorm;

    GPUCA_UNROLL(, U(1))
    for (int32_t irow = lastRow; irow >= firstRow && irow - firstRow + nHits >= minHits; irow--) {
      calink ih = tracker.TrackletRowHits()[tracklet.FirstHit() + (irow - firstRow)];
      if (ih != CALINK_DEAD_CHANNEL) {
        gap++;
      }
      if (ih != CALINK_INVAL && ih != CALINK_DEAD_CHANNEL) {
        GPUglobalref() const GPUTPCRow& row = tracker.Row(irow);
        const bool own = (tracker.HitWeight(row, ih) <= w);
        const bool sharedOK = nShared <= (nHits < sharingMinNorm ? maxSharedNorm : nHits * maxSharedFrac);
        if (own || sharedOK) { // SG!!!
          gap = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
          const bool inShared = nHits < (uint32_t)GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE;
#pragma GCC diagnostic pop
          if constexpr (GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE > 0) {
            if (inShared) {
              s.mHits[nHits][iThread].Set(irow, ih);
            }
          }
          if (!inShared) {
            trackHits[nHits - GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE].Set(irow, ih);
          }
          nHits++;
          if (!own) {
            nShared++;
          }
        }
      }

      if (gap > tracker.Param().rec.tpc.trackFollowingMaxRowGap || irow == firstRow) { // store
        if (nHits >= minHits) {
          uint32_t nFirstTrackHit = CAMath::AtomicAdd(tracker.NTrackHits(), (uint32_t)nHits);
          if (nFirstTrackHit + nHits > tracker.NMaxTrackHits()) {
            tracker.raiseError(GPUErrors::ERROR_TRACK_HIT_OVERFLOW, tracker.ISector(), nFirstTrackHit + nHits, tracker.NMaxTrackHits());
            CAMath::AtomicExch(tracker.NTrackHits(), tracker.NMaxTrackHits());
            return;
          }
          uint32_t itrout = CAMath::AtomicAdd(tracker.NTracks(), 1u);
          if (itrout >= tracker.NMaxTracks()) {
            tracker.raiseError(GPUErrors::ERROR_TRACK_OVERFLOW, tracker.ISector(), itrout, tracker.NMaxTracks());
            CAMath::AtomicExch(tracker.NTracks(), tracker.NMaxTracks());
            return;
          }
          tracker.Tracks()[itrout].SetLocalTrackId(itrout);
          tracker.Tracks()[itrout].SetParam(tracklet.Param());
          tracker.Tracks()[itrout].SetFirstHitID(nFirstTrackHit);
          tracker.Tracks()[itrout].SetNHits(nHits);
          for (uint32_t jh = 0; jh < nHits; jh++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
            const bool inShared = jh < (uint32_t)GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE;
#pragma GCC diagnostic pop
            if constexpr (GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE > 0) {
              if (inShared) {
                tracker.TrackHits()[nFirstTrackHit + nHits - 1 - jh] = s.mHits[jh][iThread];
              }
            }
            if (!inShared) {
              tracker.TrackHits()[nFirstTrackHit + nHits - 1 - jh] = trackHits[jh - GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE];
            }
          }
        }
        nHits = 0;
        gap = 0;
        nShared = 0;
      }
    }
  }
}

// GPUTPCTrackletPacking-based consumer: one thread per lane instead of per tracklet, walking
// every physical row (NROWS iterations, uniform across all lanes -- no length-driven trip-count
// divergence) rather than only its own tracklet's row range. Reads GPUTPCTrackletPacking's
// row-major LaneRowHit() instead of the scattered per-tracklet TrackletRowHits() index, and
// walks each lane's assigned tracklets via LaneListHead()/LaneListNext() in exactly the
// LastRow -> FirstRow order the inner logic below assumes (see the walk-direction note on
// GPUTPCTrackletPacking's "sweep"). That inner logic -- gap tracking, shared-hit bookkeeping,
// the minHits-reachability early exit, and the fact that one tracklet can emit several tracks if
// it has an internal gap -- is copied unchanged from Thread<0> above; the only new part is the
// outer per-row loop and the tracklet-boundary detection wrapped around it. Writes to
// TracksPacked()/TrackHitsPacked(), not Tracks()/TrackHits(), so this diagnostic can run
// alongside the production "select" pass above without corrupting its output.
template <>
GPUdii() void GPUTPCTrackletSelector::Thread<GPUTPCTrackletSelector::selectPacked>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  const uint32_t nLanes = *tracker.NLanesUsed();
  const uint32_t laneStride = tracker.NMaxTracklets();
  const float maxSharedFrac = tracker.Param().rec.tpc.trackletMaxSharedFraction;

  GPUTPCHitId trackHits[GPUTPCGeometry::NROWS - GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE];

  for (uint32_t lane = iBlock * nThreads + iThread; lane < nLanes; lane += nBlocks * nThreads) {
    uint32_t cur = tracker.LaneListHead()[lane];
    bool active = false;
    int32_t firstRow = 0;
    int32_t w = 0;
    uint32_t minHits = 0;
    uint32_t sharingMinNorm = 0;
    float maxSharedNorm = 0.f;
    uint32_t gap = 0;
    uint32_t nShared = 0;
    uint32_t nHits = 0;

    GPUCA_UNROLL(, U(1))
    for (int32_t row = GPUTPCGeometry::NROWS - 1; row >= 0; row--) {
      GPUbarrierWarp();

      if (!active && cur != GPUTPCTrackletPacking::NoLane) {
        GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::const_reference_restrict> newTracklet = tracker.Tracklets()[cur];
        if ((int32_t)newTracklet.LastRow() == row) {
          firstRow = newTracklet.FirstRow();
          w = newTracklet.HitWeight();
          minHits = tracker.Param().rec.tpc.minNClustersTrackSeed == -1 ? tracker.Param().tpcMinHitsB5(newTracklet.Param().QPt() * tracker.Param().qptB5Scaler) : tracker.Param().rec.tpc.minNClustersTrackSeed;
          sharingMinNorm = minHits * tracker.Param().rec.tpc.trackletMinSharedNormFactor;
          maxSharedNorm = maxSharedFrac * sharingMinNorm;
          gap = 0;
          nShared = 0;
          nHits = 0;
          active = true;
        }
      }

      if (!active) {
        continue;
      }

      if (row - firstRow + (int32_t)nHits < (int32_t)minHits) {
        // budget exhausted before reaching firstRow: give up on this tracklet's remaining rows
        active = false;
        cur = tracker.LaneListNext()[cur];
        continue;
      }

      calink ih = tracker.LaneRowHit()[row * laneStride + lane];
      if (ih != CALINK_DEAD_CHANNEL) {
        gap++;
      }
      if (ih != CALINK_INVAL && ih != CALINK_DEAD_CHANNEL) {
        GPUglobalref() const GPUTPCRow& hitRow = tracker.Row(row);
        const bool own = (tracker.HitWeight(hitRow, ih) <= w);
        const bool sharedOK = nShared <= (nHits < sharingMinNorm ? maxSharedNorm : nHits * maxSharedFrac);
        if (own || sharedOK) { // SG!!!
          gap = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
          const bool inShared = nHits < (uint32_t)GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE;
#pragma GCC diagnostic pop
          if constexpr (GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE > 0) {
            if (inShared) {
              s.mHits[nHits][iThread].Set(row, ih);
            }
          }
          if (!inShared) {
            trackHits[nHits - GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE].Set(row, ih);
          }
          nHits++;
          if (!own) {
            nShared++;
          }
        }
      }

      if (gap > tracker.Param().rec.tpc.trackFollowingMaxRowGap || row == firstRow) { // store
        if (nHits >= minHits) {
          uint32_t nFirstTrackHit = CAMath::AtomicAdd(tracker.NTrackHitsPacked(), (uint32_t)nHits);
          if (nFirstTrackHit + nHits > tracker.NMaxTrackHits()) {
            tracker.raiseError(GPUErrors::ERROR_TRACK_HIT_OVERFLOW, tracker.ISector(), nFirstTrackHit + nHits, tracker.NMaxTrackHits());
            CAMath::AtomicExch(tracker.NTrackHitsPacked(), tracker.NMaxTrackHits());
            return;
          }
          uint32_t itrout = CAMath::AtomicAdd(tracker.NTracksPacked(), 1u);
          if (itrout >= tracker.NMaxTracks()) {
            tracker.raiseError(GPUErrors::ERROR_TRACK_OVERFLOW, tracker.ISector(), itrout, tracker.NMaxTracks());
            CAMath::AtomicExch(tracker.NTracksPacked(), tracker.NMaxTracks());
            return;
          }
          GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::const_reference_restrict> curTracklet = tracker.Tracklets()[cur];
          tracker.TracksPacked()[itrout].SetLocalTrackId(itrout);
          tracker.TracksPacked()[itrout].SetParam(curTracklet.Param());
          tracker.TracksPacked()[itrout].SetFirstHitID(nFirstTrackHit);
          tracker.TracksPacked()[itrout].SetNHits(nHits);
          for (uint32_t jh = 0; jh < nHits; jh++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
            const bool inShared = jh < (uint32_t)GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE;
#pragma GCC diagnostic pop
            if constexpr (GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE > 0) {
              if (inShared) {
                tracker.TrackHitsPacked()[nFirstTrackHit + nHits - 1 - jh] = s.mHits[jh][iThread];
              }
            }
            if (!inShared) {
              tracker.TrackHitsPacked()[nFirstTrackHit + nHits - 1 - jh] = trackHits[jh - GPUCA_PAR_TRACKLET_SELECTOR_HITS_REG_SIZE];
            }
          }
        }
        nHits = 0;
        gap = 0;
        nShared = 0;
      }

      if (row == firstRow) {
        active = false;
        cur = tracker.LaneListNext()[cur];
      }
    }
  }
}
