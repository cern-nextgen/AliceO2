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

/// \file GPUReconstructionCUDAKernelsSpecialize.inc
/// \author David Rohr

#if defined(GPUCA_SPECIALIZE_THRUST_SORTS) && !defined(GPUCA_GPUCODE_COMPILEKERNELS)

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerMergeBorders, 3>(const krnlSetupTime& _xyz, GPUTPCGMBorderRange* const& range, int32_t const& N, int32_t const& cmpMax)
{
  if (cmpMax) {
    GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, range, N, MergeBorderTracks_compMax());
  } else {
    GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, range, N, MergeBorderTracks_compMin());
  }
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerSortTracks, 0>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackOrderProcess(), processors()->tpcMerger.NMergedTracks(), GPUTPCGMMergerSortTracks_comp(mProcessorsShadow->tpcMerger.MergedTracks()));
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerSortTracksQPt, 0>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackSort(), processors()->tpcMerger.NMergedTracks(), GPUTPCGMMergerSortTracksQPt_comp(mProcessorsShadow->tpcMerger.MergedTracks()));
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerMergeLoopers, 1>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.LooperCandidates(), processors()->tpcMerger.Memory()->nLooperMatchCandidates, GPUTPCGMMergerMergeLoopers_comp());
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMO2Output, GPUTPCGMO2Output::sort>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackSortO2(), processors()->tpcMerger.NOutputTracksTPCO2(), GPUTPCGMO2OutputSort_comp());
}

// Sorts GPUTPCExtrapolationTracking's 2 candidate arrays (ToLeft/ToRight) by RowIndex, via an
// index permutation (sortOnDevice needs a contiguous T* array; the SoA-stored candidate records
// don't have one). See GPUTPCExtrapolationTracking.cxx/GPUChainTrackingSectorTracker.cxx for why
// this lives here rather than inline in the (backend-agnostic, non-__CUDACC__) .cxx call site --
// same reason the sorts above do too. Uses the named GPUTPCExtrapolationCandidateRowIndex_comp
// functor (GPUTPCExtrapolationTracking.h), capturing only the plain mRowIndex leaf pointer via its
// constructor -- not the multi-pointer SoA wrapper ExtrapCandTo*() itself returns (capturing that
// caused an illegal memory access), and not an inline lambda (hit an NVCC extended-lambda compile
// error here -- every comparator actually used from inside one of these specializations elsewhere
// in this codebase is a named functor, never a raw lambda, for exactly this reason).
template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCExtrapolationTracking, GPUTPCExtrapolationTracking::sort>(const krnlSetupTime& _xyz)
{
  const uint32_t iSector = _xyz.y.index;
  GPUTPCTracker& trk = processors()->tpcTrackers[iSector];
  GPUTPCTracker& trkShadow = mProcessorsShadow->tpcTrackers[iSector];
  // Pass our own persistent scratch buffer (GPUTPCTracker::ExtrapCandSortTempMem(), registered the
  // normal way) instead of letting sortOnDevice fall back to AllocateVolatileDeviceMemory() -- see
  // GPUCommonAlgorithm.h's sortOnDevice comment for why that default is unsafe here (called once
  // per sector, concurrently across streams, unlike this codebase's other, rare/sequential callers).
  // Both calls below reuse the same buffer sequentially (stream-ordered: ToLeft's sort completes
  // before ToRight's begins), so one buffer per tracker is enough.
  const size_t tempSize = GPUTPCTracker::ExtrapCandSortTempMemSize(trk.NMaxTracks());
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, trkShadow.ExtrapCandSortedIndexToLeft(), (size_t)*trk.NExtrapCandToLeft(), GPUTPCExtrapolationCandidateRowIndex_comp(trkShadow.ExtrapCandToLeft().mRowIndex), trkShadow.ExtrapCandSortTempMem(), tempSize);
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, trkShadow.ExtrapCandSortedIndexToRight(), (size_t)*trk.NExtrapCandToRight(), GPUTPCExtrapolationCandidateRowIndex_comp(trkShadow.ExtrapCandToRight().mRowIndex), trkShadow.ExtrapCandSortTempMem(), tempSize);
}
#endif // GPUCA_SPECIALIZE_THRUST_SORTS

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  GPUChkErr(cudaMemsetAsync(ptr, 0, size, mInternals->Streams[_xyz.x.stream]));
}
