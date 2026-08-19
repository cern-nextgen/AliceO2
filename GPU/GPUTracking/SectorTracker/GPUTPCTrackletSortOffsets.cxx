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
  if (iBlock != 0 || iThread != 0) {
    return;
  }
  GPUglobalref() GPUAtomic(uint32_t)* GPUrestrict() keyCount = tracker.TrackletSortKeyCount();
  constexpr uint32_t nKeys = GPUTPCGeometry::NROWS * GPUTPCGeometry::NROWS;
  uint32_t running = 0;
  for (uint32_t key = 0; key < nKeys; key++) {
    // AtomicExch both reads the old count and installs this key's start offset in one
    // portable op, avoiding a plain (non-atomic) store to a GPUAtomic(uint32_t) slot.
    const uint32_t c = CAMath::AtomicExch(&keyCount[key], running);
    running += c;
  }
}
