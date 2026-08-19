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

/// \file GPUTPCTrackletSortScatter.cxx
/// \author Oliver Rietmann

#include "GPUTPCTrackletSortScatter.h"
#include "GPUTPCTracker.h"
#include "GPUTPCTracklet.h"
#include "GPUCommonMath.h"
#include "MemLayout.h"

using namespace o2::gpu;

template <>
GPUdii() void GPUTPCTrackletSortScatter::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& s, processorType& GPUrestrict() tracker)
{
  const uint32_t nTracklets = *tracker.NTracklets();
  for (uint32_t itr = get_global_id(0); itr < nTracklets; itr += get_global_size(0)) {
    GPUglobalref() MemLayout::wrapper<GPUTPCTrackletSkeleton, MemLayout::reference_restrict> tracklet = tracker.Tracklets()[itr];
    const uint32_t key = tracklet.LastRow() * GPUTPCGeometry::NROWS + tracklet.FirstRow();
    const uint32_t pos = CAMath::AtomicAdd(&tracker.TrackletSortKeyCount()[key], 1u);
    tracker.TrackletSortedIndex()[pos] = itr;
  }
}
