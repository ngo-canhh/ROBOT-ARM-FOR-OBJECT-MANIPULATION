import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import cv2
import os # Import os for path handling

class KukaVisionGraspingEnv(gym.Env):
    """
    Gymnasium environment for Kuka IIWA robot with gripper and cameras,
    using joint velocity control for movement. The agent is responsible
    for maintaining vertical gripper orientation via observation and reward.

    Goal: Position the gripper above an object and grasp it.
    Observations: Images from mounted cameras and orientation error vector.
    Actions: Joint velocities for arm joints
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode=None, max_steps=1000):
        """
        Initializes the Kuka grasping environment.

        Args:
            render_mode (str, optional): The rendering mode. Can be 'human', 'rgb_array', or None.
            max_steps (int): Maximum simulation steps per episode.
        """
        super().__init__()

        self.render_mode = render_mode
        self._max_steps = max_steps
        self._time_step = 1. / 240. # Simulation time step

        # --- Connect and setup PyBullet ---
        if self.render_mode == 'human':
            self.physics_client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.2])
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self._time_step)

        # --- Load Assets ---
        p.loadURDF("plane.urdf", [0, 0, 0])

        # Load Kuka WITH gripper
        try:
            script_dir = os.path.dirname(__file__)
            # Ensure the model directory path is correct relative to the script
            model_path = os.path.join(script_dir, "model", "kuka_with_gripper2.sdf")
            if not os.path.exists(model_path):
                 # Fallback or try alternative common locations if needed
                 alt_model_path = os.path.join(os.path.dirname(script_dir), "model", "kuka_with_gripper2.sdf") # Example: one level up
                 if os.path.exists(alt_model_path):
                     model_path = alt_model_path
                 else:
                      raise FileNotFoundError(f"SDF model not found at {model_path} or {alt_model_path}")
            self.robot_id = p.loadSDF(model_path)[0]
        except Exception as e:
             print(f"Error loading robot SDF: {e}")
             print("Please ensure 'model/kuka_with_gripper2.sdf' exists relative to the script or adjust the path.")
             raise e

        self._kuka_base_position = [0, 0, 0]
        p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

        # --- Identify robot components ---
        self.num_joints_total = p.getNumJoints(self.robot_id)
        self.controllable_joint_indices = []
        self.gripper_joint_indices = []
        self.gripper_tip_indices = []
        ee_link_name_options = ['gripper_base_link', 'iiwa_link_ee', 'tool_link_ee', 'base_link'] # Added 'gripper_base_link'

        self.end_effector_link_index = -1 # Initialize before loop

        for i in range(self.num_joints_total):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            joint_type = info[2]

            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                if 'J' in joint_name:
                    self.controllable_joint_indices.append(i)
                elif 'finger' in joint_name and 'joint' in joint_name:
                    self.gripper_joint_indices.append(i)
            elif joint_type == p.JOINT_FIXED:
                if 'tip' in joint_name and 'joint' in joint_name:
                    self.gripper_tip_indices.append(i)

            # Check link name for end effector AFTER checking joint type
            if link_name in ee_link_name_options and self.end_effector_link_index == -1: # Find first match
                 self.end_effector_link_index = i
                 print(f"Identified end-effector link: '{link_name}' (associated with joint index: {i})")
                 # It's actually the *link* index we want, not the joint index for getLinkState
                 # However, p.getJointInfo returns the index of the *joint* that is the parent of this link.
                 # Let's verify this or find a more direct way if needed. Often, the EE link is the last one.
                 # For now, assuming the joint index `i` corresponds to the link we need.


        # If no named link found, fall back to the last controllable joint's link
        if self.end_effector_link_index == -1 and self.controllable_joint_indices:
             last_joint_idx = self.controllable_joint_indices[-1]
             link_name = p.getJointInfo(self.robot_id, last_joint_idx)[12].decode('utf-8')
             print(f"Warning: Could not find a specific end-effector link among {ee_link_name_options}. Using link '{link_name}' associated with last controllable joint index {last_joint_idx} as reference.")
             self.end_effector_link_index = last_joint_idx # Use the joint index as link index reference for now
        elif self.end_effector_link_index == -1:
             raise ValueError("Could not identify any suitable end-effector link index.")

        self.num_controllable_joints = len(self.controllable_joint_indices)
        self.num_gripper_joints = len(self.gripper_joint_indices)

        print(f"Number of controllable arm joints: {self.num_controllable_joints}")
        print(f"Arm joint indices: {self.controllable_joint_indices}")
        print(f"Number of gripper joints: {self.num_gripper_joints}")
        print(f"Gripper joint indices: {self.gripper_joint_indices}")
        print(f"Gripper tip indices: {self.gripper_tip_indices}")
        print(f"Using Link Index for End-Effector: {self.end_effector_link_index}")

        if self.num_controllable_joints != 7:
             print(f"Warning: Expected 7 controllable arm joints, but found {self.num_controllable_joints}. Check SDF joint names.")
        if self.num_gripper_joints < 2:
             print(f"Warning: Expected at least 2 gripper joints, but found {self.num_gripper_joints}. Check SDF joint names.")


        self._rest_poses = [0, math.pi / 4, 0, -math.pi / 2, 0, math.pi / 4, 0]
        if len(self._rest_poses) != self.num_controllable_joints:
            print(f"Warning: Adjusting _rest_poses length.")
            self._rest_poses = [0] * self.num_controllable_joints

        # --- Setup cameras ---
        self.camera_width = 84
        self.camera_height = 84
        self.camera_configs = [
            { 'name': 'wrist_camera', 'link': self.end_effector_link_index, 'fov': 60, 'aspect': 1.0, 'nearVal': 0.01, 'farVal': 1.0, 'width': self.camera_width, 'height': self.camera_height, 'pos_offset': [0, 0, 0.05], 'ori_offset': p.getQuaternionFromEuler([0, -math.pi/2, 0]) },
            { 'name': 'front_camera', 'link': -1, 'fov': 60, 'aspect': 1.0, 'nearVal': 0.1, 'farVal': 5.0, 'width': self.camera_width, 'height': self.camera_height, 'pos': [0.8, 0, 1], 'target': [0.5, 0, 0.1] },
            { 'name': 'side_camera', 'link': -1, 'fov': 60, 'aspect': 1.0, 'nearVal': 0.1, 'farVal': 5.0, 'width': self.camera_width, 'height': self.camera_height, 'pos': [0.43, 0.65, 0.5], 'target': [0.43, 0, 0.23] }
        ]

        # --- Load object ---
        self._object_start_position = np.array([0.6, 0.0, 0.02])
        try:
            self._object_id = p.loadURDF("cube_small.urdf", basePosition=self._object_start_position)
            new_color_rgba = [0.0, 1.0, 0.0, 1.0]
            p.changeVisualShape(self._object_id, -1, rgbaColor=new_color_rgba)
        except p.error:
            print("Warning: 'cube_small.urdf' not found. Trying 'duck_vhacd.urdf'.")
            try:
                self._object_id = p.loadURDF("duck_vhacd.urdf", basePosition=self._object_start_position, globalScaling=0.8)
            except p.error:
                raise FileNotFoundError("Could not load 'cube_small.urdf' or 'duck_vhacd.urdf'.")
        p.changeDynamics(self._object_id, -1, mass=0.1, lateralFriction=0.8)

        # --- Target position ---
        self._grasp_height_offset = 0.2
        self._grasp_target_visual_id = None
        self._workspace_visual_ids = []  # Store workspace visual elements
        # --- Workspace boundaries ---
        self._workspace_low = [0.4, -0.2, self._object_start_position[2]]
        self._workspace_high = [0.7, 0.2, self._object_start_position[2]]
        
        if self.render_mode == 'human':
            visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.5])
            self._grasp_target_visual_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=[0,0,-1])
            
            # Create workspace visualization
            self._create_workspace_visualization()

        # --- Define Action Space ---
        self._max_joint_velocity = 0.5
        self.action_space = spaces.Box(
            low=-self._max_joint_velocity,
            high=self._max_joint_velocity,
            shape=(self.num_controllable_joints,),
            dtype=np.float32
        )

        # --- Define Observation Space (Images + Orientation Error) ---
        obs_spaces = {}
        # Camera image spaces
        for cam in self.camera_configs:
            obs_spaces[cam['name']] = spaces.Box(
                low=0, high=255,
                shape=(self.camera_height, self.camera_width, 1),
                dtype=np.uint8
            )
        # Orientation error space (dot product with vertical vector)
        obs_spaces['orientation_error'] = spaces.Box(
            low=-1, high=1, 
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict(obs_spaces)

        # --- Other parameters ---
        self._motor_force = 100
        self._gripper_max_force = 50
        self._target_height_threshold = 0.025
        self._target_xy_threshold = 0.025
        # self._orientation_tolerance = 0.2 # Radians tolerance for pitch/roll error before penalty
        self._orientation_penalty = -5.0 # Large negative reward for excessive orientation error

        

        self._gripper_state = 0
        self._object_grasped = False
        self._grasp_constraint = -1
        self._current_step = 0
        self._last_action = np.zeros(self.num_controllable_joints, dtype=np.float32)

        # Removed Jacobian-based orientation controller gain

        print("KukaVisionGraspingEnv initialized with orientation error observation.")

    def _create_workspace_visualization(self):
        """
        Creates visual elements to highlight the workspace boundaries.
        The workspace is defined by the random object initialization area.
        """
        # Use instance variables for workspace boundaries
        ws_low = self._workspace_low
        ws_high = self._workspace_high
        
        # Create a semi-transparent plane to show the workspace area
        workspace_center = [(ws_low[0] + ws_high[0])/2, (ws_low[1] + ws_high[1])/2, self._object_start_position[2] + 0.001]
        workspace_size = [ws_high[0] - ws_low[0], ws_high[1] - ws_low[1], 0.002]
        
        # Create a thin box as workspace plane
        workspace_plane_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[workspace_size[0]/2, workspace_size[1]/2, workspace_size[2]/2],
            rgbaColor=[0.2, 0.6, 1.0, 0.3]  # Light blue, semi-transparent
        )
        workspace_plane_id = p.createMultiBody(
            baseVisualShapeIndex=workspace_plane_shape, 
            basePosition=workspace_center
        )
        self._workspace_visual_ids.append(workspace_plane_id)
        
        # Create corner markers for better visibility
        corner_positions = [
            [ws_low[0], ws_low[1], self._object_start_position[2] + 0.005],  # Bottom-left
            [ws_high[0], ws_low[1], self._object_start_position[2] + 0.005],  # Bottom-right
            [ws_low[0], ws_high[1], self._object_start_position[2] + 0.005],  # Top-left
            [ws_high[0], ws_high[1], self._object_start_position[2] + 0.005]   # Top-right
        ]
        
        for i, corner_pos in enumerate(corner_positions):
            corner_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.015,
                length=0.01,
                rgbaColor=[1.0, 0.5, 0.0, 0.8]  # Orange color
            )
            corner_id = p.createMultiBody(
                baseVisualShapeIndex=corner_shape,
                basePosition=corner_pos
            )
            self._workspace_visual_ids.append(corner_id)
        
        # Create boundary lines (as thin cylinders)
        boundary_lines = [
            # Bottom edge
            [(ws_low[0] + ws_high[0])/2, ws_low[1], self._object_start_position[2] + 0.003],
            # Top edge  
            [(ws_low[0] + ws_high[0])/2, ws_high[1], self._object_start_position[2] + 0.003],
            # Left edge
            [ws_low[0], (ws_low[1] + ws_high[1])/2, self._object_start_position[2] + 0.003],
            # Right edge
            [ws_high[0], (ws_low[1] + ws_high[1])/2, self._object_start_position[2] + 0.003]
        ]
        
        line_lengths = [ws_high[0] - ws_low[0], ws_high[0] - ws_low[0], ws_high[1] - ws_low[1], ws_high[1] - ws_low[1]]
        orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),      # Bottom edge (along X)
            p.getQuaternionFromEuler([0, 0, 0]),      # Top edge (along X)
            p.getQuaternionFromEuler([0, 0, math.pi/2]),  # Left edge (along Y)
            p.getQuaternionFromEuler([0, 0, math.pi/2])   # Right edge (along Y)
        ]
        
        for i, (line_pos, line_length, orientation) in enumerate(zip(boundary_lines, line_lengths, orientations)):
            line_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.005,
                length=line_length,
                rgbaColor=[0.8, 0.2, 0.2, 0.9]  # Red color for boundaries
            )
            line_id = p.createMultiBody(
                baseVisualShapeIndex=line_shape,
                basePosition=line_pos,
                baseOrientation=orientation
            )
            self._workspace_visual_ids.append(line_id)
        
        print(f"Created workspace visualization with {len(self._workspace_visual_ids)} visual elements")

    def set_workspace_boundaries(self, low, high):
        """
        Updates the workspace boundaries.
        
        Args:
            low (list): Lower bounds [x_min, y_min, z_min]
            high (list): Upper bounds [x_max, y_max, z_max]
        """
        self._workspace_low = low.copy() if isinstance(low, list) else list(low)
        self._workspace_high = high.copy() if isinstance(high, list) else list(high)
        print(f"Updated workspace boundaries: {self._workspace_low} to {self._workspace_high}")

    def get_workspace_boundaries(self):
        """
        Returns the current workspace boundaries.
        
        Returns:
            tuple: (low_bounds, high_bounds)
        """
        return self._workspace_low.copy(), self._workspace_high.copy()

    def _calculate_orientation_error(self):
        """
        Calculates the orientation error between the end-effector's current orientation
        and the target vertical orientation (pitch = pi).

        Returns:
            np.ndarray: The dot product of the gripper's z-axis and the target vertical vector (0, 0, -1).
        """
        try:
            ee_link_state = p.getLinkState(self.robot_id, self.end_effector_link_index, computeForwardKinematics=True, physicsClientId=self.physics_client_id)
            current_orientation_quat = ee_link_state[1] # Quaternion [x, y, z, w]
            rot_matrix = p.getMatrixFromQuaternion(current_orientation_quat)
            # print(f"Current EE orientation (Euler): {current_euler}")
        except p.error as e:
             print(f"Warning: Could not get EE orientation: {e}. Returning zero error.")
             return np.zeros(9, dtype=np.float32)

        gripper_z_vector = np.array(rot_matrix[6:9]) # z vector in world frame
        nomalized_gripper_z_vector = gripper_z_vector / np.linalg.norm(gripper_z_vector) # Normalize
        targer_z_vector = np.array([0, 0, -1]) # Target vertical vector in world frame

        # Calculate dot product
        dot_product = np.dot(nomalized_gripper_z_vector, targer_z_vector)
        dot_product = np.clip(dot_product, -1.0, 1.0) # Clip to avoid NaN from acos

        return np.array([dot_product], dtype=np.float32)

    def _get_camera_images(self):
        """ Captures and returns Gray images from all configured cameras. (Code unchanged) """
        images = {}
        for camera in self.camera_configs:
            if camera['link'] == -1:
                view_matrix = p.computeViewMatrix(camera['pos'], camera['target'], [0, 0, 1])
            else:
                link_state = p.getLinkState(self.robot_id, camera['link'], computeForwardKinematics=True)
                link_pos_world, link_ori_world = link_state[0], link_state[1]
                cam_pos_world, cam_ori_world = p.multiplyTransforms(link_pos_world, link_ori_world, camera['pos_offset'], camera['ori_offset'])
                rot_matrix_world = p.getMatrixFromQuaternion(cam_ori_world)
                forward_vector_world = [rot_matrix_world[0], rot_matrix_world[3], rot_matrix_world[6]]
                up_vector_world = [rot_matrix_world[2], rot_matrix_world[5], rot_matrix_world[8]]
                camera_target_world = [cam_pos_world[i] + forward_vector_world[i] for i in range(3)]
                view_matrix = p.computeViewMatrix(cam_pos_world, camera_target_world, up_vector_world)

            proj_matrix = p.computeProjectionMatrixFOV(camera['fov'], camera['aspect'], camera['nearVal'], camera['farVal'])
            img_arr = p.getCameraImage(camera['width'], camera['height'], view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, flags=p.ER_NO_SEGMENTATION_MASK)
            rgb_opengl = np.reshape(img_arr[2], (camera['height'], camera['width'], 4))
            rgb_img = rgb_opengl[:, :, :3].astype(np.uint8)
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            gray_img = np.expand_dims(gray_img, axis=-1).astype(np.uint8) # Add channel dimension
            images[camera['name']] = gray_img
        return images

    def _get_observation(self):
        """
        Gets the complete observation dictionary, including camera images
        and the orientation error vector.

        Returns:
            dict: The observation dictionary matching self.observation_space.
        """
        # Get camera images
        images = self._get_camera_images()

        # Get orientation error
        orientation_error = self._calculate_orientation_error()

        # Combine into the observation dictionary
        observation = images # Start with the images dict
        observation['orientation_error'] = orientation_error

        return observation

    def _open_gripper(self):
        """ Opens the gripper. (Code unchanged) """
        if self._grasp_constraint != -1:
            try: p.removeConstraint(self._grasp_constraint)
            except p.error: pass
            self._grasp_constraint = -1
            self._object_grasped = False

        # Define open positions (adjust based on your gripper URDF/SDF)
        # Assuming symmetric gripper where negative/positive moves fingers apart
        open_pos_left = -0.4 # Example
        open_pos_right = 0.4 # Example

        if self.num_gripper_joints >= 2:
            left_finger_joint = self.gripper_joint_indices[0]
            right_finger_joint = self.gripper_joint_indices[1]
            p.setJointMotorControl2(self.robot_id, left_finger_joint, p.POSITION_CONTROL, targetPosition=open_pos_left, force=self._gripper_max_force)
            p.setJointMotorControl2(self.robot_id, right_finger_joint, p.POSITION_CONTROL, targetPosition=open_pos_right, force=self._gripper_max_force)
        elif self.num_gripper_joints == 1: # Handle single-joint gripper if necessary
             gripper_joint = self.gripper_joint_indices[0]
             open_pos = 0.4 # Example for single joint
             p.setJointMotorControl2(self.robot_id, gripper_joint, p.POSITION_CONTROL, targetPosition=open_pos, force=self._gripper_max_force)

        self._gripper_state = 0

    def _close_gripper(self):
        """ Closes the gripper. (Code unchanged) """
        # Define closed positions
        closed_pos_left = -0.05 # Example
        closed_pos_right = 0.05 # Example

        if self.num_gripper_joints >= 2:
            left_finger_joint = self.gripper_joint_indices[0]
            right_finger_joint = self.gripper_joint_indices[1]
            p.setJointMotorControl2(self.robot_id, left_finger_joint, p.POSITION_CONTROL, targetPosition=closed_pos_left, force=self._gripper_max_force)
            p.setJointMotorControl2(self.robot_id, right_finger_joint, p.POSITION_CONTROL, targetPosition=closed_pos_right, force=self._gripper_max_force)
        elif self.num_gripper_joints == 1:
             gripper_joint = self.gripper_joint_indices[0]
             closed_pos = 0.0 # Example
             p.setJointMotorControl2(self.robot_id, gripper_joint, p.POSITION_CONTROL, targetPosition=closed_pos, force=self._gripper_max_force)

        self._gripper_state = 1

    def _try_grasp_object(self):
        """ Attempts to grasp the object. (Code largely unchanged) """
        if self._object_grasped: return True

        self._close_gripper()
        for _ in range(60): # Simulate closing time
            p.stepSimulation()
            if self.render_mode == 'human': time.sleep(self._time_step)

        ee_link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        ee_position = np.array(ee_link_state[0])
        try:
            object_pos_orn = p.getBasePositionAndOrientation(self._object_id)
            object_position = np.array(object_pos_orn[0])
        except p.error: return False

        distance_ee_to_object = np.linalg.norm(object_position - ee_position)
        grasp_proximity_threshold = self._grasp_height_offset

        # Optional: Add contact check if proximity isn't reliable enough
        contact_points_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self._object_id, linkIndexA=self.gripper_tip_indices[0])
        contact_points_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self._object_id, linkIndexA=self.gripper_tip_indices[1])
        is_touching_object = len(contact_points_left) > 0 and len(contact_points_right) > 0

        if distance_ee_to_object < grasp_proximity_threshold or is_touching_object:
            ee_world_to_local = p.invertTransform(ee_link_state[0], ee_link_state[1])
            object_pos_in_ee, object_ori_in_ee = p.multiplyTransforms(ee_world_to_local[0], ee_world_to_local[1], object_pos_orn[0], object_pos_orn[1])
            try:
                self._grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=self.robot_id, parentLinkIndex=self.gripper_tip_indices[0],
                    childBodyUniqueId=self._object_id, childLinkIndex=-1,
                    jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                    parentFramePosition=object_pos_in_ee, childFramePosition=[0, 0, 0],
                    parentFrameOrientation=object_ori_in_ee # Match orientation too
                )
                self._object_grasped = True
                print("Object successfully grasped!")
                return True
            except p.error as e:
                print(f"Error creating grasp constraint: {e}")
                self._object_grasped = False
                # Optionally, you can try to re-open the gripper if grasping fails
                self._open_gripper()
                for _ in range(60): p.stepSimulation()
                return False
        else:
            print(f"Grasp attempt failed - Distance: {distance_ee_to_object:.4f}")
            # Optionally, you can try to re-open the gripper if grasping fails
            self._open_gripper()
            for _ in range(60): p.stepSimulation()
            return False

    def _apply_joint_velocities(self, joint_velocities):
        """
        Applies target velocities to the controllable arm joints.
        Orientation correction is now handled by the agent via observation/reward.

        Args:
            joint_velocities (np.ndarray): Array of desired velocities for each
                                           controllable arm joint (rad/s).
        """
        # Ensure it's a numpy array and clipped (redundant if clipped in step)
        target_velocities = np.array(joint_velocities, dtype=np.float32)

        # --- Apply final velocities (NO Jacobian correction) ---
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.controllable_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_velocities.tolist(),
            forces=[self._motor_force] * self.num_controllable_joints,
            physicsClientId=self.physics_client_id
        )

    def reset(self, seed=None, options=None):
        """ Resets the environment. (Calls _get_observation) """
        super().reset(seed=seed)
        self._current_step = 0
        self._object_grasped = False
        self._last_action = np.zeros(self.num_controllable_joints, dtype=np.float32)

        if self._grasp_constraint != -1:
            try: p.removeConstraint(self._grasp_constraint)
            except p.error: pass
            self._grasp_constraint = -1

        for i, joint_index in enumerate(self.controllable_joint_indices):
            p.resetJointState(self.robot_id, joint_index, self._rest_poses[i], 0.0)
            p.setJointMotorControl2(self.robot_id, joint_index, p.VELOCITY_CONTROL, force=0)

        self._open_gripper()
        for _ in range(30): p.stepSimulation()

        # Use instance variables for workspace boundaries
        ws_low = self._workspace_low.copy()  # Use copy to avoid modifying original
        ws_high = self._workspace_high.copy()
        random_object_position = self.np_random.uniform(low=ws_low, high=ws_high)
        try: # Get object Z from physics if possible, else use default
             current_obj_z = p.getBasePositionAndOrientation(self._object_id)[0][2]
             random_object_position[2] = max(current_obj_z, 0.02) # Ensure slightly above plane
        except p.error:
             random_object_position[2] = 0.02 # Fallback Z

        random_yaw = self.np_random.uniform(low=-math.pi, high=math.pi)
        object_orientation = p.getQuaternionFromEuler([0, 0, random_yaw])
        p.resetBasePositionAndOrientation(self._object_id, random_object_position.tolist(), object_orientation)
        self._object_start_position = random_object_position

        if self.render_mode == 'human' and self._grasp_target_visual_id is not None:
            target_pos = self._get_target_grasp_position()
            p.resetBasePositionAndOrientation(self._grasp_target_visual_id, target_pos, [0,0,0,1])

        for _ in range(20): p.stepSimulation()

        # Get initial observation using the new method
        observation = self._get_observation()
        info = self._get_info()

        if hasattr(self, '_camera_windows_created'): delattr(self, '_camera_windows_created')
        if self.render_mode == 'human':
            cv2.destroyAllWindows()
            self.render()

        return observation, info

    def _get_target_grasp_position(self):
        """ Calculates the target position above the object. (Code unchanged) """
        try:
            object_pos, _ = p.getBasePositionAndOrientation(self._object_id)
            return np.array([object_pos[0], object_pos[1], object_pos[2] + self._grasp_height_offset])
        except p.error:
            print(f"Warning: Error getting object position for target calc.")
            return np.zeros(3) # Default target if object error

    def _get_info(self):
        """ Collects auxiliary environment information. (Adds orientation error) """
        info = {}
        try:
             ee_link_state = p.getLinkState(self.robot_id, self.end_effector_link_index, computeForwardKinematics=True)
             info['ee_position'] = list(ee_link_state[0])
             ee_ori_quat = ee_link_state[1]
             info['ee_orientation_euler'] = list(p.getEulerFromQuaternion(ee_ori_quat))
        except p.error:
             info['ee_position'] = [0,0,0]
             info['ee_orientation_euler'] = [0,0,0]

        try:
            obj_pos_orn = p.getBasePositionAndOrientation(self._object_id)
            info['object_position'] = list(obj_pos_orn[0])
        except p.error:
            info['object_position'] = list(self._object_start_position) # Use start pos as fallback

        target_grasp_pos = self._get_target_grasp_position()
        info['target_grasp_position'] = target_grasp_pos.tolist()

        ee_pos_arr = np.array(info['ee_position'])
        target_pos_arr = np.array(target_grasp_pos)
        info['distance_ee_to_target'] = float(np.linalg.norm(target_pos_arr - ee_pos_arr))
        info['distance_ee_xy'] = float(np.linalg.norm(target_pos_arr[:2] - ee_pos_arr[:2]))
        info['distance_ee_z'] = float(abs(target_pos_arr[2] - ee_pos_arr[2]))

        info['orientation_error'] = self._calculate_orientation_error() # Add error to info

        try:
            joint_states = p.getJointStates(self.robot_id, self.controllable_joint_indices)
            info['joint_positions'] = [s[0] for s in joint_states]
            info['joint_velocities'] = [s[1] for s in joint_states]
        except p.error:
             info['joint_positions'] = [0.0] * self.num_controllable_joints
             info['joint_velocities'] = [0.0] * self.num_controllable_joints


        info['gripper_state'] = self._gripper_state
        info['object_grasped'] = self._object_grasped
        info['last_action'] = self._last_action.tolist()
        return info

    def _is_gripper_at_target(self):
        """ Checks if the gripper is at the pre-grasp target pose. (Code unchanged) """
        try:
             ee_link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
             ee_position = np.array(ee_link_state[0])
        except p.error: return False

        target_grasp_position = self._get_target_grasp_position()
        if np.all(target_grasp_position == 0): return False # Target calc failed

        distance_ee_xy = np.linalg.norm(target_grasp_position[:2] - ee_position[:2])
        distance_ee_z = abs(target_grasp_position[2] - ee_position[2])

        xy_aligned = (distance_ee_xy < self._target_xy_threshold)
        z_aligned = (distance_ee_z < self._target_height_threshold)
        return xy_aligned and z_aligned

    def _calc_reward(self):
        """
        Calculates the reward signal.

        Includes:
        - Reward for moving closer to the target pre-grasp position.
        - Penalty for excessive orientation error (deviation from vertical).
        - Bonus for reaching the target pre-grasp position.
        - Large bonus for successful grasp.
        """
        info = self._get_info() # Get current state details, includes 'orientation_error'

        # --- Position Reward ---
        dist_xy = info['distance_ee_xy']
        dist_z = info['distance_ee_z']

        if dist_xy >= self._target_xy_threshold:
            reward_xy = -5 * np.tanh(dist_xy / (self._target_xy_threshold * 2 * 10))
            # range (-5, -0.02498)
        else:
            reward_xy = - (2 / self._target_xy_threshold) * dist_xy + 3 # if dist_xy < threshold, reward = 1 to 3
        
        if dist_z >= self._target_height_threshold:
            reward_z = -5 * np.tanh(dist_z / (self._target_height_threshold * 2 * 10))
            # range (-5, -0.02498)
        else:
            reward_z = - (2 / self._target_height_threshold) * dist_z + 3 # if dist_z < threshold, reward = 1 to 3

        position_reward = reward_xy + reward_z


        # --- Orientation Penalty ---
        orientation_error = np.array(info['orientation_error'])

        # Calculate the orientation error (dot product)  
        orientation_penalty_applied = -3 * (1 - orientation_error[0]) ** (0.75) 
        # orientation in (-1, 1) => penalty in (-5.65, 0)

        # --- Bonus Rewards ---
        # reach_target_bonus = 0.0
        grasp_success_bonus = 0.0

        # if self._is_gripper_at_target():
        #     reach_target_bonus = 2.0

        if self._object_grasped:
            grasp_success_bonus = 100.0 # Significant bonus

        # --- Combine Rewards ---
        w_pos = 1.0
        # w_reach = 1.0
        w_grasp = 1.0
        w_orientation = 1.0
        # Note: Orientation is handled by penalty, not weighted positive reward

        total_reward = (w_pos * position_reward +
                        # w_reach * reach_target_bonus +
                        w_grasp * grasp_success_bonus +
                        w_orientation* orientation_penalty_applied) # Add the penalty here
        
        # print(f'Reward: {total_reward:.4f}: \n(Pos: {w_pos * position_reward:.4f}, Reach: {w_reach * reach_target_bonus:.4f}, Grasp: {w_grasp * grasp_success_bonus:.4f}, Orientation Penalty: {orientation_penalty_applied:.4f})')
        reward_info = {
            'Reward': total_reward,
            'position_reward': position_reward * w_pos,
            # 'reach_target_bonus': reach_target_bonus * w_reach,
            'grasp_success_bonus': grasp_success_bonus * w_grasp,
            'orientation_penalty': orientation_penalty_applied * w_orientation,
        }
        # for key, val in reward_info.items():
        #     print(f"{key}: {val:.4f}")
        return float(total_reward), reward_info


    def step(self, action):
        """ Executes one time step in the environment. (Calls _get_observation, no orientation correction) """
        self._last_action = np.array(action, dtype=np.float32).clip(
             self.action_space.low, self.action_space.high
        )

        # Apply joint velocities WITHOUT internal orientation correction
        self._apply_joint_velocities(self._last_action)

        # Step simulation
        num_sim_steps_per_env_step = 10
        for _ in range(num_sim_steps_per_env_step):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        self._current_step += 1

        # Check for grasp attempt condition
        terminated = False

        try_grasp = False
        if not self._object_grasped and self._is_gripper_at_target():
            # print(f"Step {self._current_step}: Gripper at target. Attempting grasp...")
            current_joint_states = p.getJointStates(self.robot_id, self.controllable_joint_indices)
            current_joint_vels = [s[1] for s in current_joint_states]
            self._apply_joint_velocities(np.zeros(self.num_controllable_joints)) # Stop before grasp
            for _ in range(30): p.stepSimulation(physicsClientId=self.physics_client_id)
            grasp_success = self._try_grasp_object()
            if grasp_success:
                terminated = True
                # print(f"Step {self._current_step}: Successfully grasped object!")
                # Large reward is handled in _calc_reward now
            else:
                try_grasp = True
                # print(f"Step {self._current_step}: Grasp attempt failed.")
                # Apply last action again after grasp attempt
                self._apply_joint_velocities(current_joint_vels)
                for _ in range(30): p.stepSimulation(physicsClientId=self.physics_client_id)
                 

        # Get new observation (includes orientation error)
        observation = self._get_observation()
        reward, reward_info = self._calc_reward()
        # if try_grasp:
        #     reward -= 1.0
        #     reward_info['grasp_fail_penalty'] = -1.0
        # else:
        #     reward_info['grasp_fail_penalty'] = 0.0
        info = self._get_info() # Info also updated
        info.update(reward_info)


        # Check truncation
        truncated = False
        if self._current_step >= self._max_steps:
            truncated = True
            # print(f"Step {self._current_step}: Max steps reached, truncating.")

        # Update visual target
        if self.render_mode == 'human' and self._grasp_target_visual_id is not None:
             target_pos = self._get_target_grasp_position()
             # Avoid error if target calc failed
             if not np.all(target_pos == 0):
                 p.resetBasePositionAndOrientation(self._grasp_target_visual_id, target_pos, [0,0,0,1])

        if self.render_mode == 'human':
             self.render()

        return observation, reward, terminated, truncated, info

    def _display_camera_images(self):
        """ Displays camera images in OpenCV windows. (Code unchanged) """
        if not hasattr(self, '_camera_windows_created'):
            for name in self.camera_configs: cv2.namedWindow(f"Camera: {name['name']}", cv2.WINDOW_AUTOSIZE)
            self._camera_windows_created = True

        images = self._get_camera_images() # Fetch fresh images
        for name, img in images.items():
            # img_display_bgr = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Camera: {name}", img)

    def render(self):
        """ Renders the environment. (Code unchanged regarding logic, calls _display_camera_images) """
        if self.render_mode == 'rgb_array':
            images = self._get_camera_images()
            # Use get() with a default to prevent KeyError if camera name is mistyped/missing
            return images.get('side_camera', np.zeros((self.camera_height, self.camera_width, 1), dtype=np.uint8))
        elif self.render_mode == 'human':
            self._display_camera_images()
            cv2.waitKey(1) # Essential for OpenCV window refresh
            return None
        else:
             return None

    def close(self):
        """ Cleans up resources. (Code unchanged) """
        print("Closing KukaVisionGraspingEnv...")
        if hasattr(self, 'physics_client_id') and self.physics_client_id is not None:
            try:
                if p.isConnected(self.physics_client_id):
                    # Clean up workspace visual elements
                    if hasattr(self, '_workspace_visual_ids'):
                        for visual_id in self._workspace_visual_ids:
                            try:
                                p.removeBody(visual_id, physicsClientId=self.physics_client_id)
                            except p.error:
                                pass  # Visual element may already be removed
                        self._workspace_visual_ids.clear()
                    
                    print("Disconnecting from PyBullet.")
                    p.disconnect(physicsClientId=self.physics_client_id)
            except p.error as e: print(f"Error during PyBullet disconnection: {e}")
            finally: self.physics_client_id = None
        print("Closing OpenCV windows.")
        if hasattr(self, '_camera_windows_created'): delattr(self, '_camera_windows_created')

        if self.render_mode == 'human':
            cv2.destroyAllWindows()
        print("KukaVisionGraspingEnv closed.")

if __name__ == "__main__":
    env = KukaVisionGraspingEnv(render_mode='human')
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()