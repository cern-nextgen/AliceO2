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

/// \file GPUTPCBaseTrackParam.h
/// \author David Rohr, Sergey Gorbunov

#ifndef GPUTPCBASETRACKPARAM_H
#define GPUTPCBASETRACKPARAM_H

#include "GPUTPCDef.h"
#include "MemLayout.h"

namespace o2::gpu
{

namespace detail
{

struct GPUTPCCovariance
{
  float data[15];
  constexpr operator float*() { return data; }
  constexpr operator const float*() const { return data; }
  constexpr float& operator[](int32_t i) { return data[i]; }
  constexpr const float& operator[](int32_t i) const { return data[i]; }
};

template <template <class> class F>
struct GPUTPCParameterSkeleton
{
  MEMLAYOUT_APPLY_UNARY(mY, mZ, mSinPhi, mDzDs, mQPt)
  MEMLAYOUT_APPLY_BINARY(GPUTPCParameterSkeleton, MEMLAYOUT_EXPAND(mY), MEMLAYOUT_EXPAND(mZ), MEMLAYOUT_EXPAND(mSinPhi), MEMLAYOUT_EXPAND(mDzDs), MEMLAYOUT_EXPAND(mQPt))
  F<float> mY, mZ, mSinPhi, mDzDs, mQPt;
};

// Needed for sorting
constexpr void swap(GPUTPCParameterSkeleton<MemLayout::reference> a, GPUTPCParameterSkeleton<MemLayout::reference> b) {
    std::swap(a.mY, b.mY);
    std::swap(a.mZ, b.mZ);
    std::swap(a.mSinPhi, b.mSinPhi);
    std::swap(a.mDzDs, b.mDzDs);
    std::swap(a.mQPt, b.mQPt);
}

}

/**
 * @class GPUTPCBaseTrackParam
 *
 * GPUTPCBaseTrackParam class contains track parameters
 * used in output of the GPUTPCTracker sector tracker.
 * This class is used for transfer between tracker and merger and does not contain the covariance matrice
 */
template <template <class> class F>
struct GPUTPCBaseTrackParamSkeleton
{
  MEMLAYOUT_APPLY_UNARY(mX, mC, mZOffset, mP)
  MEMLAYOUT_APPLY_BINARY(GPUTPCBaseTrackParamSkeleton, MEMLAYOUT_EXPAND(mX), MEMLAYOUT_EXPAND(mC), MEMLAYOUT_EXPAND(mZOffset), MEMLAYOUT_EXPAND(mP))

  GPUd() float X() const { return mX; }
  GPUd() float Y() const { return mP.mY; }
  GPUd() float Z() const { return mP.mZ; }
  GPUd() float SinPhi() const { return mP.mSinPhi; }
  GPUd() float DzDs() const { return mP.mDzDs; }
  GPUd() float QPt() const { return mP.mQPt; }
  GPUd() float ZOffset() const { return mZOffset; }

  GPUd() float Err2Y() const { return mC[0]; }
  GPUd() float Err2Z() const { return mC[2]; }
  GPUd() float Err2SinPhi() const { return mC[5]; }
  GPUd() float Err2DzDs() const { return mC[9]; }
  GPUd() float Err2QPt() const { return mC[14]; }
  GPUhd() const float* Cov() const { return mC; }
  GPUd() float GetCov(int32_t i) const { return mC[i]; }
  GPUhd() void SetCov(int32_t i, float v) { mC[i] = v; }

  GPUhd() float GetX() const { return mX; }
  GPUhd() float GetY() const { return mP.mY; }
  GPUhd() float GetZ() const { return mP.mZ; }
  GPUhd() float GetSinPhi() const { return mP.mSinPhi; }
  GPUhd() float GetDzDs() const { return mP.mDzDs; }
  GPUhd() float GetQPt() const { return mP.mQPt; }
  GPUhd() float GetZOffset() const { return mZOffset; }

  GPUd() float GetKappa(float Bz) const { return -mP.mQPt * Bz; }

  GPUd() void SetX(float v) { mX = v; }
  GPUd() void SetY(float v) { mP.mY = v; }
  GPUd() void SetZ(float v) { mP.mZ = v; }
  GPUd() void SetSinPhi(float v) { mP.mSinPhi = v; }
  GPUd() void SetDzDs(float v) { mP.mDzDs = v; }
  GPUd() void SetQPt(float v) { mP.mQPt = v; }
  GPUd() void SetZOffset(float v) { mZOffset = v; }

  // WARNING, Track Param Data is copied in the GPU Tracklet Constructor element by element instead of using copy constructor!!!
  // This is neccessary for performance reasons!!!
  // Changes to Elements of this class therefore must also be applied to TrackletConstructor!!!
  F<float> mX;       // x position
  F<detail::GPUTPCCovariance> mC;  // the covariance matrix for Y,Z,SinPhi,..
  F<float> mZOffset; // z offset
  MemLayout::wrapper<detail::GPUTPCParameterSkeleton, F> mP;  // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
};

// Needed for sorting
constexpr void swap(GPUTPCBaseTrackParamSkeleton<MemLayout::reference> a, GPUTPCBaseTrackParamSkeleton<MemLayout::reference> b) {
    std::swap(a.mX, b.mX);
    std::swap(a.mC, b.mC);
    std::swap(a.mZOffset, b.mZOffset);
    swap(a.mP, b.mP);
}

using GPUTPCBaseTrackParam = MemLayout::wrapper<GPUTPCBaseTrackParamSkeleton, MemLayout::value>;

} // namespace o2::gpu

#endif
