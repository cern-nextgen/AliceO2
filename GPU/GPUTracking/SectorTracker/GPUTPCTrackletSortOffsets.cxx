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

/// \file GPUTPCTrackletSortOffsets.cxx
/// \author Oliver Rietmann

#include "GPUTPCTrackletSortOffsets.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"

using namespace o2::gpu;

template <>
GPUdii() void GPUTPCTrackletSortOffsets::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  if (iBlock != 0) {
    return;
  }
  // Parallel block-wide exclusive prefix sum over the NROWS*NROWS-key histogram, launched with
  // exactly OffsetsThreads threads in one block (see GPUChainTrackingSectorTracker.cxx). Splitting
  // the key range into OffsetsThreads chunks, prefix-summing each chunk locally, then combining
  // the per-chunk totals with a small (log2(OffsetsThreads)-step) shared-memory scan replaces what
  // used to be ~23k *dependent* global-memory round trips on a single thread.
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
  s.mScan[iThread] = local;
  GPUbarrier();

  // Phase 2: turn the per-thread chunk totals into an exclusive prefix sum, in shared memory
  // (Hillis-Steele scan: read-then-barrier-then-write-then-barrier avoids any read/write race
  // between a slot's own update and a neighbor reading its pre-update value this round).
  for (uint32_t offset = 1; offset < (uint32_t)nThreads; offset <<= 1) {
    const uint32_t addend = ((uint32_t)iThread >= offset) ? s.mScan[iThread - offset] : 0;
    GPUbarrier();
    s.mScan[iThread] += addend;
    GPUbarrier();
  }
  const uint32_t base = s.mScan[iThread] - local; // inclusive -> exclusive

  // Phase 3: re-derive each key's count and write back its final offset (doubling as the
  // scatter kernel's per-key cursor), now fully in parallel across chunks.
  uint32_t running = base;
  for (uint32_t key = begin; key < end; key++) {
    const uint32_t c = keyCount[key];
    keyCount[key] = running;
    running += c;
  }
}
