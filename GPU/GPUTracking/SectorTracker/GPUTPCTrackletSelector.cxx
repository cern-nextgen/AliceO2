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
#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"
#include "MemLayout.h"

using namespace o2::gpu;

// Tracklets are sorted by (length, LastRow) ahead of selection (see GPUTPCTrackletSort_comp), via
// an index permutation sorted with GPUCommonAlgorithm::sortOnDevice (host-launched, CUB
// DeviceMergeSort) -- not the tracklet records themselves, since sortOnDevice needs a genuine
// contiguous T* array and the SoA-stored GPUTPCTrackletSkeleton records don't have one. The actual
// sortOnDevice call lives in GPUReconstructionCUDAKernelsSpecialize.inc (a
// runKernelBackendTimed<GPUTPCTrackletSelector, sort> specialization, active on real CUDA/HIP
// builds), not in Thread<sort> below -- same pattern GPUTPCGMMergerSortTracks/SortTracksQPt use,
// and for the same reason: sortOnDevice needs __CUDACC__/__HIPCC__ actually defined in the
// translation unit, which this plain, backend-agnostic .cxx file never has.
template <>
GPUdii() void GPUTPCTrackletSelector::Thread<GPUTPCTrackletSelector::sort>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
#ifndef GPUCA_SPECIALIZE_THRUST_SORTS
  // Real fallback for backends without the sortOnDevice specialization active (e.g. CPU-only
  // builds, where __CUDACC__/__HIPCC__ are never defined) -- mirrors
  // GPUTPCGMMerger::SortTracks/SortTracksQPt's identical #ifndef GPUCA_SPECIALIZE_THRUST_SORTS
  // pattern. Deliberately not a no-op: a no-op fallback would leave CPU builds' tracklet
  // visitation order diverging from a CUDA/HIP build's sortOnDevice order with no downstream
  // re-canonicalization covering it in general (only the deterministic-mode debug sort does, and
  // only for Tracks(), not Tracklets() itself).
  if (iThread == 0 && iBlock == 0) {
    GPUCommonAlgorithm::sortDeviceDynamic(tracker.TrackletSortedIndex(), tracker.TrackletSortedIndex() + *tracker.NTracklets(), GPUTPCTrackletSort_comp(tracker.Tracklets().mFirstRow, tracker.Tracklets().mLastRow));
  }
#endif
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

    const int32_t itr = tracker.TrackletSortedIndex()[i];
    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::reference_restrict> tracklet = tracker.Tracklets()[itr];

    int32_t firstRow = tracklet.FirstRow();
    int32_t lastRow = tracklet.LastRow();

    const int32_t w = tracklet.HitWeight();

    int32_t irow = firstRow;

    uint32_t gap = 0;
    uint32_t nShared = 0;
    uint32_t nHits = 0;
    const uint32_t minHits = tracker.Param().rec.tpc.minNClustersTrackSeed == -1 ? tracker.Param().tpcMinHitsB5(tracklet.Param().QPt() * tracker.Param().qptB5Scaler) : tracker.Param().rec.tpc.minNClustersTrackSeed;
    const uint32_t sharingMinNorm = minHits * tracker.Param().rec.tpc.trackletMinSharedNormFactor;
    const float maxSharedNorm = maxSharedFrac * sharingMinNorm;

    GPUCA_UNROLL(, U(1))
    for (irow = lastRow; irow >= firstRow && irow - firstRow + nHits >= minHits; irow--) {
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
