/*
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include "Utils.h"
#include <boost/algorithm/string.hpp>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;



//@angle_diff: unit is degree
//@dist_diff: unit is meter
vectorMatrix4f cluster_poses(float angle_diff, float dist_diff, const vectorMatrix4f &poses_in, const vectorMatrix4f &symmetry_tfs)
{
  printf("num original candidates = %zu\n",poses_in.size());
  vectorMatrix4f poses_out;
  poses_out.push_back(poses_in[0]);

  const float radian_thres = angle_diff/180.0*M_PI;

  for (int i=1;i<poses_in.size();i++)
  {
    bool isnew = true;
    Eigen::Matrix4f cur_pose = poses_in[i];
    for (const auto &cluster:poses_out)
    {
      Eigen::Vector3f t0 = cluster.block(0,3,3,1);
      Eigen::Vector3f t1 = cur_pose.block(0,3,3,1);

      if ((t0-t1).norm()>=dist_diff)
      {
        continue;
      }

      for (const auto &tf: symmetry_tfs)
      {
        Eigen::Matrix4f cur_pose_tmp = cur_pose*tf;
        float rot_diff = Utils::rotationGeodesicDistance(cur_pose_tmp.block(0,0,3,3), cluster.block(0,0,3,3));
        if (rot_diff < radian_thres)
        {
          isnew = false;
          break;
        }
      }

      if (!isnew) break;
    }

    if (isnew)
    {
      poses_out.push_back(poses_in[i]);
    }
  }

  printf("num of pose after clustering: %zu\n",poses_out.size());
  return poses_out;
}

static vectorMatrix4f ndarray_to_vecmat4f(const py::array & arr_any, const char * name)
{
  // float32 に強制キャスト + C連続
  py::array_t<float, py::array::c_style | py::array::forcecast> arr(arr_any);

  if (arr.ndim() != 3 || arr.shape(1) != 4 || arr.shape(2) != 4) {
    throw std::invalid_argument(std::string(name) + " must have shape (N,4,4)");
  }

  const ssize_t N = arr.shape(0);
  const float * p = arr.data();

  vectorMatrix4f out;
  out.reserve((size_t)N);

  // numpy は行優先(C order)が基本。Eigen::Matrix4f は列優先がデフォルト。
  // なので row-major Map で受けてから代入するのが安全。
  using RowMat4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

  for (ssize_t i = 0; i < N; ++i) {
    Eigen::Map<const RowMat4f> m(p + i * 16);
    out.push_back(m);  // 代入で col-major の Mat4f に詰め替え
  }
  return out;
}

static py::array_t<float> vecmat4f_to_ndarray(const vectorMatrix4f & mats)
{
  const ssize_t K = (ssize_t)mats.size();

  std::vector<ssize_t> shape = {K, 4, 4};
  py::array_t<float> out({shape});
  auto r = out.mutable_unchecked<3>();

  for (ssize_t i = 0; i < K; ++i) {
    // Python 側は (4,4) を row-major で期待することが多いので、要素で埋める
    for (ssize_t a = 0; a < 4; ++a)
      for (ssize_t b = 0; b < 4; ++b) r(i, a, b) = mats[(size_t)i](a, b);
  }
  return out;
}

PYBIND11_MODULE(_mycpp, m)
{
  m.def(
    "cluster_poses",
    [](float angle_diff, float dist_diff, py::array poses, py::array sym) {
      auto poses_v = ndarray_to_vecmat4f(poses, "poses");
      auto sym_v = ndarray_to_vecmat4f(sym, "symmetry_tfs");
      if (poses_v.empty()) throw py::value_error("poses is empty");
      if (sym_v.empty()) throw py::value_error("symmetry_tfs is empty");

      py::gil_scoped_release release;
      auto out_v = cluster_poses(angle_diff, dist_diff, poses_v, sym_v);
      py::gil_scoped_acquire acquire;

      return vecmat4f_to_ndarray(out_v);
    }, py::arg("angle_diff"), py::arg("dist_diff"), py::arg("poses_in"),
    py::arg("symmetry_tfs"));
}