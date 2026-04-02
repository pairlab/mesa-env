# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
MimicGen environment interface classes for basic robosuite environments.
"""
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T

import mesa.mimicgen.utils.pose_utils as PoseUtils
from mesa.mimicgen.env_interfaces.base import MG_EnvInterface
from mesa.mimicgen.utils.misc_utils import get_controller


class RobosuiteInterface(MG_EnvInterface):
    """
    MimicGen environment interface base class for basic robosuite environments.
    """

    # Note: base simulator interface class must fill out interface type as a class property
    INTERFACE_TYPE = "robosuite"

    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """

        # OSC control frame is a MuJoCo site - just retrieve its current pose
        # TODO: this should work fine for single-arm robots, but not for multi-arm robots.
        obj_name = get_controller(self.env).ref_name
        return self.get_object_pose(
            obj_name=obj_name, 
            obj_type="site",
        )

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller and returns an action 
        (usually a normalized delta pose action) to try and achieve that target pose. 

        Args:
            target_pose (np.array): 4x4 target eef pose
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation)
        """

        # version check for robosuite - must be v1.2+, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] >= "2")

        # target position and rotation
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose()
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get maximum position and rotation action bounds
        max_dpos = get_controller(self.env).output_max[0]
        max_drot = get_controller(self.env).output_max[3]

        # relative = get_controller(self.env).input_type == "delta"
        if relative:
            # normalized delta position action
            delta_position = target_pos - curr_pos
            delta_position = np.clip(delta_position / max_dpos, -1., 1.)

            # normalized delta rotation action
            delta_rot_mat = target_rot.dot(curr_rot.T)
            delta_quat = T.mat2quat(delta_rot_mat)
            delta_rotation = T.quat2axisangle(delta_quat)
            delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
            return np.concatenate([delta_position, delta_rotation])

        # absolute position and rotation action
        target_quat = T.mat2quat(target_rot)
        abs_rotation = T.quat2axisangle(target_quat)
        return np.concatenate([target_pos, abs_rotation])

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """

        # version check for robosuite - must be v1.2+, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] >= "2")

        if (not relative):
            # convert absolute action to absolute pose
            target_pos = action[:3]
            target_quat = T.axisangle2quat(action[3:6])
            target_rot = T.quat2mat(target_quat)
        else:
            # get maximum position and rotation action bounds
            max_dpos = get_controller(self.env).output_max[0]
            max_drot = get_controller(self.env).output_max[3]

            # unscale actions
            delta_position = action[:3] * max_dpos
            delta_rotation = action[3:6] * max_drot

            # current position and rotation
            curr_pose = self.get_robot_eef_pose()
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            # get pose target
            target_pos = curr_pos + delta_position
            delta_quat = T.axisangle2quat(delta_rotation)
            delta_rot_mat = T.quat2mat(delta_quat)
            target_rot = delta_rot_mat.dot(curr_rot)

        target_pose = PoseUtils.make_pose(target_pos, target_rot)
        return target_pose

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """

        # last dimension is gripper action
        return action[-1:]

    # robosuite-specific helper method for getting object poses
    def get_object_pose(self, obj_name, obj_type):
        """
        Returns 4x4 object pose given the name of the object and the type.

        Args:
            obj_name (str): name of object
            obj_type (str): type of object - either "body", "geom", or "site"

        Returns:
            obj_pose (np.array): 4x4 object pose
        """
        assert obj_type in ["body", "geom", "site"]

        if obj_type == "body":
            obj_id = self.env.sim.model.body_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.body_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.body_xmat[obj_id].reshape(3, 3))
        elif obj_type == "geom":
            obj_id = self.env.sim.model.geom_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.geom_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.geom_xmat[obj_id].reshape(3, 3))
        elif obj_type == "site":
            obj_id = self.env.sim.model.site_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.site_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.site_xmat[obj_id].reshape(3, 3))

        return PoseUtils.make_pose(obj_pos, obj_rot)
