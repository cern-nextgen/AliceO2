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

/// \file GPUTPCTrackletPacking.cxx

#include "GPUTPCTrackletPacking.h"
#include "GPUTPCTracker.h"
#include "GPUTPCTracklet.h"
#include "GPUCommonMath.h"
#include "MemLayout.h"

using namespace o2::gpu;

// Phase 1: histogram tracklets independently by FirstRow and by LastRow -- two NROWS-bucket
// counting sorts sharing one pass over the tracklets, feeding "offsets"/"scatter" below.
template <>
GPUdii() void GPUTPCTrackletPacking::Thread<GPUTPCTrackletPacking::count>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  const uint32_t nTracklets = *tracker.NTracklets();
  for (uint32_t itr = iBlock * nThreads + iThread; itr < nTracklets; itr += nBlocks * nThreads) {
    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::const_reference_restrict> tracklet = tracker.Tracklets()[itr];
    CAMath::AtomicAdd(&tracker.TrackletFirstRowCount()[tracklet.FirstRow()], 1u);
    CAMath::AtomicAdd(&tracker.TrackletLastRowCount()[tracklet.LastRow()], 1u);
  }
}

// Phase 2: turn both NROWS-length histograms into exclusive prefix sums. NROWS (152) is small
// enough that a single thread doing this serially is negligible -- unlike the NROWS*NROWS-key
// histogram in GPUTPCTrackletSelector.cxx's "offsets" pass, which needed a parallel block-wide
// scan to avoid dominating the sort (~138ms on a partitioned H100).
template <>
GPUdii() void GPUTPCTrackletPacking::Thread<GPUTPCTrackletPacking::offsets>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  if (iBlock != 0 || iThread != 0) {
    return;
  }
  GPUglobalref() GPUAtomic(uint32_t)* GPUrestrict() firstRowCount = tracker.TrackletFirstRowCount();
  GPUglobalref() GPUAtomic(uint32_t)* GPUrestrict() lastRowCount = tracker.TrackletLastRowCount();
  uint32_t runningFirst = 0;
  uint32_t runningLast = 0;
  for (uint32_t row = 0; row < GPUTPCGeometry::NROWS; row++) {
    const uint32_t cFirst = firstRowCount[row];
    firstRowCount[row] = runningFirst;
    runningFirst += cFirst;
    const uint32_t cLast = lastRowCount[row];
    lastRowCount[row] = runningLast;
    runningLast += cLast;
  }
}

// Phase 3: scatter each tracklet's index into its FirstRow- and LastRow-ordered position. After
// this pass, TrackletFirstRowCount()[r] / TrackletLastRowCount()[r] hold the *end* offset of row
// r's bucket in TrackletByFirstRow()/TrackletByLastRow() (== the start offset of row r+1's
// bucket, since keys are contiguous).
template <>
GPUdii() void GPUTPCTrackletPacking::Thread<GPUTPCTrackletPacking::scatter>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  const uint32_t nTracklets = *tracker.NTracklets();
  for (uint32_t itr = iBlock * nThreads + iThread; itr < nTracklets; itr += nBlocks * nThreads) {
    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::const_reference_restrict> tracklet = tracker.Tracklets()[itr];
    const uint32_t posFirst = CAMath::AtomicAdd(&tracker.TrackletFirstRowCount()[tracklet.FirstRow()], 1u);
    tracker.TrackletByFirstRow()[posFirst] = itr;
    const uint32_t posLast = CAMath::AtomicAdd(&tracker.TrackletLastRowCount()[tracklet.LastRow()], 1u);
    tracker.TrackletByLastRow()[posLast] = itr;
  }
}

// Phase 4: the actual experiment. Single-thread greedy interval partitioning, provably optimal
// in lane count (interval graphs are perfect graphs; greedy-by-start-time achieves the
// chromatic number == max simultaneous overlap) but a strictly sequential dependency chain: each
// row's free-lane state depends on every prior row's. That is the same *shape* of problem as the
// original (unfixed) offsets pass that measured ~138ms on a partitioned H100 -- this kernel
// exists to find out whether the same failure mode recurs here, at this problem's actual size
// (~n tracklets touched twice via monotonically-advancing bucket ranges, not ~NROWS*NROWS
// dependent atomics).
//
// Walked high row -> low row, with LastRow as the "open a lane" event and FirstRow as "release
// it" -- the mirror image of a naive FirstRow-ascending sweep. This matters for correctness, not
// just symmetry: GPUTPCTrackletSelector's per-tracklet loop walks lastRow -> firstRow (not the
// other way), and its gap-based track segmentation and minHits-reachability early exit are both
// order-dependent -- reusing them unchanged for the packed/lane-major consumer requires visiting
// each lane's tracklets in that same direction, which in turn requires the packing sweep itself
// to assign lanes while walking rows in that direction. Also builds, as a side effect, each
// lane's tracklet list as a singly linked list (LaneListHead()/LaneListNext()) in exactly that
// walk order -- free, since the sweep already visits tracklets in the right sequence to build it
// incrementally (append at the tail; a lane's tail is only ever read after that lane has already
// received at least one tracklet, so no separate initialization pass is needed).
template <>
GPUdii() void GPUTPCTrackletPacking::Thread<GPUTPCTrackletPacking::sweep>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  if (iBlock != 0 || iThread != 0) {
    return;
  }
  GPUglobalref() const GPUAtomic(uint32_t)* GPUrestrict() firstRowEnd = tracker.TrackletFirstRowCount();
  GPUglobalref() const GPUAtomic(uint32_t)* GPUrestrict() lastRowEnd = tracker.TrackletLastRowCount();
  GPUglobalref() const uint32_t* GPUrestrict() byFirstRow = tracker.TrackletByFirstRow();
  GPUglobalref() const uint32_t* GPUrestrict() byLastRow = tracker.TrackletByLastRow();
  GPUglobalref() uint32_t* GPUrestrict() lane = tracker.TrackletLane();
  GPUglobalref() uint32_t* GPUrestrict() freeStack = tracker.LaneFreeStack();
  GPUglobalref() uint32_t* GPUrestrict() listNext = tracker.LaneListNext();
  GPUglobalref() uint32_t* GPUrestrict() listHead = tracker.LaneListHead();
  GPUglobalref() uint32_t* GPUrestrict() listTail = tracker.LaneListTail();

  uint32_t freeTop = 0;
  uint32_t nextNewLane = 0;

  for (uint32_t row = GPUTPCGeometry::NROWS; row-- > 0;) {
    if (row + 1 < GPUTPCGeometry::NROWS) {
      // release lanes for tracklets whose FirstRow == row + 1 (the row just left behind)
      const uint32_t releaseBegin = firstRowEnd[row];     // end of FirstRow-bucket(row) == start of bucket(row + 1)
      const uint32_t releaseEnd = firstRowEnd[row + 1];   // end of FirstRow-bucket(row + 1)
      for (uint32_t idx = releaseBegin; idx < releaseEnd; idx++) {
        freeStack[freeTop++] = lane[byFirstRow[idx]];
      }
    }
    // open/extend lanes for tracklets whose LastRow == row
    const uint32_t openBegin = (row == 0) ? 0 : lastRowEnd[row - 1];
    const uint32_t openEnd = lastRowEnd[row];
    for (uint32_t idx = openBegin; idx < openEnd; idx++) {
      const uint32_t itr = byLastRow[idx];
      uint32_t l;
      if (freeTop > 0) {
        l = freeStack[--freeTop];
        listNext[listTail[l]] = itr; // append after this lane's existing tail
        listTail[l] = itr;
      } else {
        l = nextNewLane++; // brand new lane: this is its first tracklet
        listHead[l] = itr;
        listTail[l] = itr;
      }
      lane[itr] = l;
      listNext[itr] = NoLane;
    }
  }
  *tracker.NLanesUsed() = nextNewLane;
}

// Transpose TrackletRowHits() into a row-major [row][lane] layout: for every row in a tracklet's
// [FirstRow, LastRow] span, write its hit index into LaneRowHit() at that row's slot for the
// tracklet's assigned lane. Parallel over tracklets (LB), no atomics needed -- by the packing
// invariant, no two tracklets ever write the same (row, lane) cell.
template <>
GPUdii() void GPUTPCTrackletPacking::Thread<GPUTPCTrackletPacking::gather>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  const uint32_t nTracklets = *tracker.NTracklets();
  const uint32_t laneStride = tracker.NMaxTracklets();
  for (uint32_t itr = iBlock * nThreads + iThread; itr < nTracklets; itr += nBlocks * nThreads) {
    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::const_reference_restrict> tracklet = tracker.Tracklets()[itr];
    const uint32_t l = tracker.TrackletLane()[itr];
    const uint32_t firstRow = tracklet.FirstRow();
    const uint32_t lastRow = tracklet.LastRow();
    for (uint32_t row = firstRow; row <= lastRow; row++) {
      tracker.LaneRowHit()[row * laneStride + l] = tracker.TrackletRowHits()[tracklet.FirstHit() + (row - firstRow)];
    }
  }
}
