#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2024 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv_bridge
import message_filters
import numpy as np
import rospkg
import rospy
import tf
import trimesh
from geometry_msgs.msg import PoseStamped
from pcl_msgs.msg import PointIndices
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from estimater import FoundationPose
from estimater import PoseRefinePredictor
from estimater import ScorePredictor
from estimater import dr


class PoseEstimation(object):
    """Pose estimation node."""

    def __init__(self) -> None:
        """Constructor."""
        pkg_path = rospkg.RosPack().get_path('foundation_pose')
        mesh_file = rospy.get_param('~reference_mesh', f'{pkg_path}/demo_data/mustard0/mesh/textured_simple.obj')
        self.__est_refine_iter = rospy.get_param('~est_refine_iter', 5)
        self.__track_refine_iter = rospy.get_param('~track_refine_iter', 2)

        mesh = trimesh.load(mesh_file)
        self.__to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.__est = FoundationPose(model_pts=mesh.vertices,
                                    model_normals=mesh.vertex_normals,
                                    mesh=mesh,
                                    scorer=scorer,
                                    refiner=refiner,
                                    debug_dir=pkg_path + '/debug',
                                    debug=1,
                                    glctx=glctx)
        self.__bridge = cv_bridge.CvBridge()
        self.__camera_info = rospy.wait_for_message('camera_info', CameraInfo)
        self.__pose = None

        self.__pub = rospy.Publisher('pose', PoseStamped, queue_size=1)

        rgb_sub = message_filters.Subscriber('color', Image)
        depth_sub = message_filters.Subscriber('depth', Image)
        indices_sub = message_filters.Subscriber('indices', PointIndices)
        self.__async = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, indices_sub], 10, 0.1)
        self.__async.registerCallback(self.__callback)

    def __callback(self, rgb_msg: Image, depth_msg: Image, indices: PointIndices) -> None:
        """Callback function for pose estimation.

        Args:
            rgb: RGB image.
            depth: Depth image.
            indices: Object indices.
        """
        if not indices.indices:
            return
        color = self.__bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        depth = self.__bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        mask = np.zeros((rgb_msg.height * rgb_msg.width), dtype=bool)
        mask[np.array(indices.indices)] = True
        mask = mask.reshape(rgb_msg.height, rgb_msg.width)
        k = np.array(self.__camera_info.K).reshape(3, 3)
        if self.__pose is None:
            self.__pose = self.__est.register(K=k,
                                              rgb=color,
                                              depth=depth,
                                              ob_mask=mask,
                                              iteration=self.__est_refine_iter)
        self.__pose = self.__est.track_one(rgb=color, depth=depth, K=k, iteration=self.__track_refine_iter)
        center_pose = self.__pose @ np.linalg.inv(self.__to_origin)

        pose_msg = PoseStamped()
        pose_msg.header = rgb_msg.header
        pose_msg.pose.position.x = center_pose[0, 3]
        pose_msg.pose.position.y = center_pose[1, 3]
        pose_msg.pose.position.z = center_pose[2, 3]
        q = tf.transformations.quaternion_from_matrix(center_pose)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        self.__pub.publish(pose_msg)


if __name__ == '__main__':
    rospy.init_node('pose_estimation')
    pose_estimation = PoseEstimation()
    rospy.spin()
