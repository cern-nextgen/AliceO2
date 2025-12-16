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

struct Covariance {
    float m00, m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14;
    constexpr operator float*() { return &m00; }
    constexpr operator const float*() const { return &m00; }
    constexpr float& operator[](int32_t i) { return *((&m00) + i); }
    constexpr const float& operator[](int32_t i) const { return *((&m00) + i); }
};

struct Parameters {
    float mY, mZ, mSinPhi, mDzDs, mQPt;
    constexpr operator float*() { return &mY; }
    constexpr operator const float*() const { return &mY; }
    constexpr float& operator[](int32_t i) { return *((&mY) + i); }
    constexpr const float& operator[](int32_t i) const { return *((&mY) + i); }
};

/**
 * @class GPUTPCBaseTrackParam
 *
 * GPUTPCBaseTrackParam class contains track parameters
 * used in output of the GPUTPCTracker sector tracker.
 * This class is used for transfer between tracker and merger and does not contain the covariance matrice
 */
template <template <class> class F>
struct GPUTPCBaseTrackParamSkeleton : public MemLayout::CRTP<GPUTPCBaseTrackParamSkeleton, F>
{
  using Base = MemLayout::CRTP<GPUTPCBaseTrackParamSkeleton, F>;
  using Base::operator=;
  MEMLAYOUT_MEMBERFUNCTIONS(GPUTPCBaseTrackParamSkeleton, F, mX, mC, mZOffset, mP)

  GPUd() float X() const { return mX; }
  GPUd() float Y() const { return mP[0]; }
  GPUd() float Z() const { return mP[1]; }
  GPUd() float SinPhi() const { return mP[2]; }
  GPUd() float DzDs() const { return mP[3]; }
  GPUd() float QPt() const { return mP[4]; }
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
  GPUhd() float GetY() const { return mP[0]; }
  GPUhd() float GetZ() const { return mP[1]; }
  GPUhd() float GetSinPhi() const { return mP[2]; }
  GPUhd() float GetDzDs() const { return mP[3]; }
  GPUhd() float GetQPt() const { return mP[4]; }
  GPUhd() float GetZOffset() const { return mZOffset; }

  GPUd() float GetKappa(float Bz) const { return -mP[4] * Bz; }

  GPUhd() const float* Par() const { return mP; }
  GPUd() const float* GetPar() const { return mP; }
  GPUd() float GetPar(int32_t i) const { return (mP[i]); }

  GPUhd() void SetPar(int32_t i, float v) { mP[i] = v; }

  GPUd() void SetX(float v) { mX = v; }
  GPUd() void SetY(float v) { mP[0] = v; }
  GPUd() void SetZ(float v) { mP[1] = v; }
  GPUd() void SetSinPhi(float v) { mP[2] = v; }
  GPUd() void SetDzDs(float v) { mP[3] = v; }
  GPUd() void SetQPt(float v) { mP[4] = v; }
  GPUd() void SetZOffset(float v) { mZOffset = v; }

  // Needed for iterators and std::sort
  /*template <template <class> class F_in>
  constexpr GPUTPCBaseTrackParamSkeleton& operator=(GPUTPCBaseTrackParamSkeleton<F_in> other) {
    mX = other.mX;
    mC = other.mC;
    mZOffset = other.mZOffset;
    mP = other.mP;
    return *this;
  }*/
  //GPUTPCBaseTrackParamSkeleton() = default;
  //GPUTPCBaseTrackParamSkeleton(F<float> X, F<Covariance> C, F<float> ZOffset, F<Parameters> P) : mX(X), mC(C), mZOffset(ZOffset), mP(P) { }
  /*GPUTPCBaseTrackParamSkeleton(const GPUTPCBaseTrackParamSkeleton& other) = default;
  GPUTPCBaseTrackParamSkeleton& operator=(const GPUTPCBaseTrackParamSkeleton& other) {
    mX = other.mX;
    mC = other.mC;
    mZOffset = other.mZOffset;
    mP = other.mP;
    return *this;
  }
  friend void swap(GPUTPCBaseTrackParamSkeleton& a, GPUTPCBaseTrackParamSkeleton& b) {
    std::swap(a.mX, b.mX);
    std::swap(a.mC, b.mC);
    std::swap(a.mZOffset, b.mZOffset);
    std::swap(a.mP, b.mP);
  }*/

  // WARNING, Track Param Data is copied in the GPU Tracklet Constructor element by element instead of using copy constructor!!!
  // This is neccessary for performance reasons!!!
  // Changes to Elements of this class therefore must also be applied to TrackletConstructor!!!
  F<float> mX;       // x position
  F<Covariance> mC;   // the covariance matrix for Y,Z,SinPhi,..
  F<float> mZOffset; // z offset
  F<Parameters> mP;    // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
};

template <
    template <class> class F_left,
    template <class> class F_right,
    class FunctionObject
>
constexpr void memberwise(GPUTPCBaseTrackParamSkeleton<F_left>& left, GPUTPCBaseTrackParamSkeleton<F_right>& right, FunctionObject&& f) {
    f(left.mX, right.mX);
    f(left.mC, right.mC);
    f(left.mZOffset, right.mZOffset);
    f(left.mP, right.mP);
}

using GPUTPCBaseTrackParam = GPUTPCBaseTrackParamSkeleton<MemLayout::value>;

} // namespace o2::gpu

#endif
