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

/// \file GPUTPCExtrapolationTracking.cxx
/// \author David Rohr

#include "GPUTPCDef.h"
#include "GPUTPCExtrapolationTracking.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCTrackletConstructor.h"
#include "GPUTPCTrackLinearisation.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"
#include "GPUParam.inc"
#include "MemLayout.h"

using namespace o2::gpu;

GPUd() int32_t GPUTPCExtrapolationTracking::PerformExtrapolationTrackingRun(GPUTPCTracker& tracker, GPUsharedref() GPUSharedMemory& smem, MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::const_reference> sourceParam, int32_t sourceLocalTrackId, uint32_t sourceSector, int32_t rowIndex, float angle, int32_t direction)
{
  /*for (int32_t j = 0;j < Tracks()[j].NHits();j++)
  {
    GPUInfo("Hit %3d: Row %3d: X %3.7lf Y %3.7lf", j, mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex(), Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()).X(),
    (float) Data().HitDataY(Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()), mTrackHits[Tracks()[iTrack].FirstHitID() + j].HitIndex()) * Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()).HstepY() + Row(mTrackHits[Tracks()[iTrack].FirstHitID() + j].RowIndex()).Grid().YMin());
  }*/

  MemLayout::wrapper<GPUTPCTrackParamSkeleton, MemLayout::value> tParam;
  tParam.InitParam();
  tParam.SetCov(0, 0.05f);
  tParam.SetCov(2, 0.05f);
  tParam.SetCov(5, 0.001f);
  tParam.SetCov(9, 0.001f);
  tParam.SetCov(14, 0.05f);
  tParam.SetParam(sourceParam);

  // GPUInfo("Parameters X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());
  if (!tParam.Rotate(angle, constants::MAX_SIN_PHI)) {
    return 0;
  }
  // GPUInfo("Rotated X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());

  int32_t maxRowGap = 10;
  GPUTPCTrackLinearisation t0(tParam);
  do {
    rowIndex += direction;
    if (!tParam.TransportToX(tracker.Row(rowIndex).X(), t0, tracker.Param().bzCLight, constants::MAX_SIN_PHI)) {
      return 0; // Reuse t0 linearization until we are in the next sector
    }
    // GPUInfo("Transported X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f (MaxY %f)", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi(), Row(rowIndex).MaxY());
    if (--maxRowGap == 0) {
      return 0;
    }
  } while (CAMath::Abs(tParam.Y()) > tracker.Row(rowIndex).MaxY());

  float err2Y, err2Z;
  tracker.GetErrors2Seeding(rowIndex, tParam.Z(), tParam.SinPhi(), tParam.DzDs(), -1.f, err2Y, err2Z); // TODO: Use correct time for multiplicity part of error estimation
  if (tParam.GetCov(0) < err2Y) {
    tParam.SetCov(0, err2Y);
  }
  if (tParam.GetCov(2) < err2Z) {
    tParam.SetCov(2, err2Z);
  }

  calink rowHits[GPUTPCGeometry::NROWS];
  int32_t nHits = GPUTPCTrackletConstructor::GPUTPCTrackletConstructorExtrapolationTracking(tracker, smem, tParam, rowIndex, direction, 0, rowHits);
  if (nHits >= tracker.Param().rec.tpc.extrapolationTrackingMinHits) {
    // GPUInfo("%d hits found", nHits);
    uint32_t hitId = CAMath::AtomicAdd(&tracker.CommonMemory()->nTrackHits, (uint32_t)nHits);
    if (hitId + nHits > tracker.NMaxTrackHits()) {
      tracker.raiseError(GPUErrors::ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW, tracker.ISector(), hitId + nHits, tracker.NMaxTrackHits());
      CAMath::AtomicExch(&tracker.CommonMemory()->nTrackHits, tracker.NMaxTrackHits());
      return 0;
    }
    uint32_t trackId = CAMath::AtomicAdd(&tracker.CommonMemory()->nTracks, 1u);
    if (trackId >= tracker.NMaxTracks()) { // >= since will increase by 1
      tracker.raiseError(GPUErrors::ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW, tracker.ISector(), trackId, tracker.NMaxTracks());
      CAMath::AtomicExch(&tracker.CommonMemory()->nTracks, tracker.NMaxTracks());
      return 0;
    }

    if (direction == 1) {
      int32_t i = 0;
      while (i < nHits) {
        const calink rowHit = rowHits[rowIndex];
        if (rowHit != CALINK_INVAL && rowHit != CALINK_DEAD_CHANNEL) {
          // GPUInfo("New track: entry %d, row %d, hitindex %d", i, rowIndex, mTrackletRowHits[rowIndex * tracker.CommonMemory()->nTracklets]);
          tracker.TrackHits()[hitId + i].Set(rowIndex, rowHit);
          // if (i == 0) tParam.TransportToX(Row(rowIndex).X(), Param().bzCLight(), constants::MAX_SIN_PHI); //Use transport with new linearisation, we have changed the track in between - NOT needed, fitting will always start at outer end of the extrapolated track!
          i++;
        }
        rowIndex++;
      }
    } else {
      int32_t i = nHits - 1;
      while (i >= 0) {
        const calink rowHit = rowHits[rowIndex];
        if (rowHit != CALINK_INVAL && rowHit != CALINK_DEAD_CHANNEL) {
          // GPUInfo("New track: entry %d, row %d, hitindex %d", i, rowIndex, mTrackletRowHits[rowIndex * tracker.CommonMemory()->nTracklets]);
          tracker.TrackHits()[hitId + i].Set(rowIndex, rowHit);
          i--;
        }
        rowIndex--;
      }
    }
    MemLayout::wrapper<GPUTPCTrackSkeleton, MemLayout::reference_restrict> track = tracker.Tracks()[trackId];
    track.SetParam(tParam.GetParam());
    track.SetNHits(nHits);
    track.SetFirstHitID(hitId);
    track.SetLocalTrackId((direction == 1 ? 0x40000000 : 0) | (sourceSector << 24) | sourceLocalTrackId);
  }

  return (nHits >= tracker.Param().rec.tpc.extrapolationTrackingMinHits);
}

GPUd() void GPUTPCExtrapolationTracking::ConsumeExtrapolationCandidates(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, uint32_t sourceSector, GPUsharedref() GPUSharedMemory& smem, GPUTPCTracker& GPUrestrict() sectorTarget, MemLayout::wrapper<GPUTPCExtrapolationCandidateSkeleton, MemLayout::pointer> cand, uint32_t nCand, float angle, int32_t direction)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < nCand; i += nThreads * nBlocks) {
    PerformExtrapolationTrackingRun(sectorTarget, smem, cand[i].Param(), cand[i].LocalTrackId(), sourceSector, cand[i].RowIndex(), angle, direction);
  }
}

template <>
GPUdii() void GPUTPCExtrapolationTracking::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() tracker)
{
  GPUCA_SHARED_CACHE(&smem.mRows[0], tracker.TrackingDataRows(), GPUTPCGeometry::NROWS * sizeof(GPUTPCRow));
  GPUbarrier();

  if (tracker.NHitsTotal() == 0) {
    return;
  }
  uint32_t sectorLeft, sectorRight;
  ExtrapolationTrackingSectorLeftRight(tracker.ISector(), sectorLeft, sectorRight);
  const GPUTPCTracker& GPUrestrict() left = tracker.GetConstantMem()->tpcTrackers[sectorLeft];
  const GPUTPCTracker& GPUrestrict() right = tracker.GetConstantMem()->tpcTrackers[sectorRight];

  // Candidates were classified and compacted by GPUTPCTrackletSelector when each track was stored
  // (see the piggyback block there) -- every entry in each of these 4 dense arrays is already known
  // to qualify, so these are plain, fully-dense grid-strides, not gated scans over all local tracks.
  ConsumeExtrapolationCandidates(nBlocks, nThreads, iBlock, iThread, sectorLeft, smem, tracker, left.ExtrapCandLowerToRight(), *left.NExtrapCandLowerToRight(), left.Param().dAlpha, -1);
  ConsumeExtrapolationCandidates(nBlocks, nThreads, iBlock, iThread, sectorLeft, smem, tracker, left.ExtrapCandUpperToRight(), *left.NExtrapCandUpperToRight(), left.Param().dAlpha, 1);
  ConsumeExtrapolationCandidates(nBlocks, nThreads, iBlock, iThread, sectorRight, smem, tracker, right.ExtrapCandLowerToLeft(), *right.NExtrapCandLowerToLeft(), -right.Param().dAlpha, -1);
  ConsumeExtrapolationCandidates(nBlocks, nThreads, iBlock, iThread, sectorRight, smem, tracker, right.ExtrapCandUpperToLeft(), *right.NExtrapCandUpperToLeft(), -right.Param().dAlpha, 1);
}

GPUd() int32_t GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorOrder(int32_t iSector)
{
  iSector++;
  if (iSector == GPUTPCGeometry::NSECTORS / 2) {
    iSector = 0;
  }
  if (iSector == GPUTPCGeometry::NSECTORS) {
    iSector = GPUTPCGeometry::NSECTORS / 2;
  }
  return iSector;
}

GPUd() void GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorLeftRight(uint32_t iSector, uint32_t& left, uint32_t& right)
{
  left = (iSector + (GPUTPCGeometry::NSECTORS / 2 - 1)) % (GPUTPCGeometry::NSECTORS / 2);
  right = (iSector + 1) % (GPUTPCGeometry::NSECTORS / 2);
  if (iSector >= (int32_t)GPUTPCGeometry::NSECTORS / 2) {
    left += GPUTPCGeometry::NSECTORS / 2;
    right += GPUTPCGeometry::NSECTORS / 2;
  }
}

template <>
GPUdii() void GPUTPCExtrapolationTrackingCopyNumbers::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() tracker, int32_t n)
{
  for (int32_t i = get_global_id(0); i < n; i += get_global_size(0)) {
    GPUconstantref() GPUTPCTracker& GPUrestrict() trk = (&tracker)[i];
    trk.CommonMemory()->nLocalTracks = trk.CommonMemory()->nTracks;
    trk.CommonMemory()->nLocalTrackHits = trk.CommonMemory()->nTrackHits;
  }
}
