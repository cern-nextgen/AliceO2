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

/// \file GPUTPCTrackParam.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKPARAM_H
#define GPUTPCTRACKPARAM_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCDef.h"
#include "GPUCommonMath.h"
// #include "wrapper.h"

namespace o2::gpu
{

template <template <template <class> class> class S, template <class> class F>
struct CRTP { };

template <template <template <class> class> class S>
struct CRTP<S, wrapper::value> {
  using Derived = S<wrapper::value>;
  constexpr operator S<wrapper::reference>() { return {{}, static_cast<Derived*>(this)->mParam, static_cast<Derived*>(this)->mSignCosPhi, static_cast<Derived*>(this)->mChi2, static_cast<Derived*>(this)->mNDF}; };
  constexpr operator S<wrapper::const_reference>() const { return {{}, static_cast<const Derived*>(this)->mParam, static_cast<const Derived*>(this)->mSignCosPhi, static_cast<const Derived*>(this)->mChi2, static_cast<const Derived*>(this)->mNDF}; };
};

template <template <template <class> class> class S>
struct CRTP<S, wrapper::reference> {
  using Derived = S<wrapper::reference>;
  constexpr operator S<wrapper::const_reference>() const { return {{}, static_cast<const Derived*>(this)->mParam, static_cast<const Derived*>(this)->mSignCosPhi, static_cast<const Derived*>(this)->mChi2, static_cast<const Derived*>(this)->mNDF}; };
};

class GPUTPCTrackLinearisation;

/**
 * @class GPUTPCTrackParam
 *
 * GPUTPCTrackParam class describes the track parametrisation
 * which is used by the GPUTPCTracker sector tracker.
 *
 */
template <template <class> class F>
class GPUTPCTrackParamSkeleton : public CRTP<GPUTPCTrackParamSkeleton, F>
{
 public:

  struct GPUTPCTrackFitParam {
    float bethe, e, theta2, EP2, sigmadE2, k22, k33, k43, k44; // parameters
  };

  GPUd() GPUTPCBaseTrackParamSkeleton<wrapper::const_reference> GetParam() const { return mParam; }
  GPUd() void SetParam(GPUTPCBaseTrackParamSkeleton<wrapper::const_reference> v) {
    mParam.mX = v.mX;
    mParam.mC[0] = v.mC[0];
    mParam.mC[1] = v.mC[1];
    mParam.mC[2] = v.mC[2];
    mParam.mC[3] = v.mC[3];
    mParam.mC[4] = v.mC[4];
    mParam.mC[5] = v.mC[5];
    mParam.mC[6] = v.mC[6];
    mParam.mC[7] = v.mC[7];
    mParam.mC[8] = v.mC[8];
    mParam.mC[9] = v.mC[9];
    mParam.mC[10] = v.mC[10];
    mParam.mC[11] = v.mC[11];
    mParam.mC[12] = v.mC[12];
    mParam.mC[13] = v.mC[13];
    mParam.mC[14] = v.mC[14];
    mParam.mZOffset = v.mZOffset;
    mParam.mP[0] = v.mP[0];
    mParam.mP[1] = v.mP[1];
    mParam.mP[2] = v.mP[2];
    mParam.mP[3] = v.mP[3];
    mParam.mP[4] = v.mP[4];
  }

  GPUd() void InitParam()
  {
    // Initialize Tracklet Parameters using default values
    SetSinPhi(0);
    SetDzDs(0);
    SetQPt(0);
    SetSignCosPhi(1);
    SetChi2(0);
    SetNDF(-3);
    SetCov(0, 1);
    SetCov(1, 0);
    SetCov(2, 1);
    SetCov(3, 0);
    SetCov(4, 0);
    SetCov(5, 1);
    SetCov(6, 0);
    SetCov(7, 0);
    SetCov(8, 0);
    SetCov(9, 1);
    SetCov(10, 0);
    SetCov(11, 0);
    SetCov(12, 0);
    SetCov(13, 0);
    SetCov(14, 1000.f);
    SetZOffset(0);
  }

  GPUd() float X() const { return mParam.X(); }
  GPUd() float Y() const { return mParam.Y(); }
  GPUd() float Z() const { return mParam.Z(); }
  GPUd() float SinPhi() const { return mParam.SinPhi(); }
  GPUd() float DzDs() const { return mParam.DzDs(); }
  GPUd() float QPt() const { return mParam.QPt(); }
  GPUd() float ZOffset() const { return mParam.ZOffset(); }
  GPUd() float SignCosPhi() const { return mSignCosPhi; }
  GPUd() float Chi2() const { return mChi2; }
  GPUd() int32_t NDF() const { return mNDF; }

  GPUd() float Err2Y() const { return mParam.Err2Y(); }
  GPUd() float Err2Z() const { return mParam.Err2Z(); }
  GPUd() float Err2SinPhi() const { return mParam.Err2SinPhi(); }
  GPUd() float Err2DzDs() const { return mParam.Err2DzDs(); }
  GPUd() float Err2QPt() const { return mParam.Err2QPt(); }

  GPUd() float GetX() const { return mParam.GetX(); }
  GPUd() float GetY() const { return mParam.GetY(); }
  GPUd() float GetZ() const { return mParam.GetZ(); }
  GPUd() float GetSinPhi() const { return mParam.GetSinPhi(); }
  GPUd() float GetDzDs() const { return mParam.GetDzDs(); }
  GPUd() float GetQPt() const { return mParam.GetQPt(); }
  GPUd() float GetSignCosPhi() const { return mSignCosPhi; }
  GPUd() float GetChi2() const { return mChi2; }
  GPUd() int32_t GetNDF() const { return mNDF; }

  GPUd() float GetKappa(float Bz) const { return mParam.GetKappa(Bz); }
  GPUd() float GetCosPhi() const { return mSignCosPhi * CAMath::Sqrt(1 - SinPhi() * SinPhi()); }

  GPUhd() const float* Par() const { return mParam.Par(); }
  GPUhd() const float* Cov() const { return mParam.Cov(); }

  GPUd() const float* GetPar() const { return mParam.GetPar(); }
  GPUd() float GetPar(int32_t i) const { return (mParam.GetPar(i)); }
  GPUd() float GetCov(int32_t i) const { return mParam.GetCov(i); }

  GPUhd() void SetPar(int32_t i, float v) { mParam.SetPar(i, v); }
  GPUhd() void SetCov(int32_t i, float v) { mParam.SetCov(i, v); }

  GPUd() void SetX(float v) { mParam.SetX(v); }
  GPUd() void SetY(float v) { mParam.SetY(v); }
  GPUd() void SetZ(float v) { mParam.SetZ(v); }
  GPUd() void SetSinPhi(float v) { mParam.SetSinPhi(v); }
  GPUd() void SetDzDs(float v) { mParam.SetDzDs(v); }
  GPUd() void SetQPt(float v) { mParam.SetQPt(v); }
  GPUd() void SetZOffset(float v) { mParam.SetZOffset(v); }
  GPUd() void SetSignCosPhi(float v) { mSignCosPhi = v >= 0 ? 1 : -1; }
  GPUd() void SetChi2(float v) { mChi2 = v; }
  GPUd() void SetNDF(int32_t v) { mNDF = v; }

  GPUd() float GetDist2(GPUTPCTrackParamSkeleton<wrapper::const_reference> t) const;
  GPUd() float GetDistXZ2(GPUTPCTrackParamSkeleton<wrapper::const_reference> t) const;

  GPUd() float GetS(float x, float y, float Bz) const;

  GPUd() void GetDCAPoint(float x, float y, float z, float& px, float& py, float& pz, float Bz) const;

  GPUd() bool TransportToX(float x, float Bz, float maxSinPhi = GPUCA_MAX_SIN_PHI);
  GPUd() bool TransportToXWithMaterial(float x, float Bz, float maxSinPhi = GPUCA_MAX_SIN_PHI);

  GPUd() bool TransportToX(float x, GPUTPCTrackLinearisation& t0, float Bz, float maxSinPhi = GPUCA_MAX_SIN_PHI, float* DL = nullptr);

  GPUd() bool TransportToX(float x, float sinPhi0, float cosPhi0, float Bz, float maxSinPhi = GPUCA_MAX_SIN_PHI);

  GPUd() bool TransportToXWithMaterial(float x, GPUTPCTrackLinearisation& t0, GPUTPCTrackFitParam& par, float Bz, float maxSinPhi = GPUCA_MAX_SIN_PHI);

  GPUd() bool TransportToXWithMaterial(float x, GPUTPCTrackFitParam& par, float Bz, float maxSinPhi = GPUCA_MAX_SIN_PHI);

  GPUd() static float ApproximateBetheBloch(float beta2);
  GPUd() static float BetheBlochGeant(float bg, float kp0 = 2.33f, float kp1 = 0.20f, float kp2 = 3.00f, float kp3 = 173e-9f, float kp4 = 0.49848f);
  GPUd() static float BetheBlochSolid(float bg);
  GPUd() static float BetheBlochGas(float bg);

  GPUd() void CalculateFitParameters(GPUTPCTrackFitParam& par, float mass = 0.13957f);
  GPUd() bool CorrectForMeanMaterial(float xOverX0, float xTimesRho, const GPUTPCTrackFitParam& par);

  GPUd() bool Rotate(float alpha, float maxSinPhi = GPUCA_MAX_SIN_PHI);
  GPUd() bool Rotate(float alpha, GPUTPCTrackLinearisation& t0, float maxSinPhi = GPUCA_MAX_SIN_PHI);
  GPUd() bool Filter(float y, float z, float err2Y, float err2Z, float maxSinPhi = GPUCA_MAX_SIN_PHI, bool paramOnly = false);

  GPUd() bool CheckNumericalQuality() const;

  GPUd() void ShiftZ(float z1, float z2, float x1, float x2, float bz, float defaultZOffsetOverR);
  GPUd() void ConstrainZ(float& z, int32_t sector, float& z0, float& lastZ);
  GPUd() int32_t GetPropagatedYZ(float bz, float x, float& projY, float& projZ) const;

  GPUdi() void ConstrainSinPhi(float limit = GPUCA_MAX_SIN_PHI)
  {
    if (GetSinPhi() > limit) {
      SetSinPhi(limit);
    } else if (GetSinPhi() < -limit) {
      SetSinPhi(-limit);
    }
  }

  GPUd() void Print() const;

#ifndef GPUCA_GPUCODE
 //private:
#endif                         //! GPUCA_GPUCODE
  GPUTPCBaseTrackParamSkeleton<F> mParam; // Track Parameters

 //private:
  // WARNING, Track Param Data is copied in the GPU Tracklet Constructor element by element instead of using copy constructor!!!
  // This is neccessary for performance reasons!!!
  // Changes to Elements of this class therefore must also be applied to TrackletConstructor!!!
  F<float> mSignCosPhi; // sign of cosPhi
  F<float> mChi2;       // the chi^2 value
  F<int32_t> mNDF;      // the Number of Degrees of Freedom
};

} // namespace o2::gpu

#endif // GPUTPCTRACKPARAM_H
