# --- Gymnasium-ready header ---
import rospy
import numpy as np
import math
import time
import os
import random
import rospkg

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils import seeding

from openai_ros.robot_envs import tiago_env
from openai_ros.task_envs.tiago import obstacles_management

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import Vector3, PoseStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler

from std_srvs.srv import Empty
from typing import List, Tuple, Optional

from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt  # solo se ti serve davvero in runtime

# The path is __init__.py of openai_ros, where we import the TiagoMazeEnv directly
max_episode_steps_per_episode = 250

register(
    id="TiagoNavigation-v0",
    entry_point="openai_ros.task_envs.tiago.tiago_navigation:TiagoNav",
    max_episode_steps=max_episode_steps_per_episode,
)

class TiagoNav(tiago_env.TiagoEnv):
    def __init__(self):
        """
        Task Env per la navigazione di Tiago.
        Versione Gymnasium: spazi definiti con gymnasium.spaces e dtype np.float32.
        """
        # Publisher vari
        self.reward_debug_pub = rospy.Publisher('/tiago_navigation/reward', Float32MultiArray, queue_size=1)
        self.velocity_pub     = rospy.Publisher('/mean_velocity',           Float32MultiArray, queue_size=1)
        self.laser_filtered_pub = rospy.Publisher('/tiago/laser/scan_filtered', LaserScan, queue_size=1)

        # Gymnasium seeding helper
        self.np_random, _ = seeding.np_random(None)

        # Range reward (opzionale)
        self.reward_range = (-np.inf, np.inf)

        # Goal position
        self.goal_pos = np.array([
            rospy.get_param('/Test_Goal/x'),
            rospy.get_param('/Test_Goal/y'),
            rospy.get_param('/Test_Goal/z')
        ], dtype=np.float32)

        # Goal eps
        self.goal_eps = np.array([
            rospy.get_param('/Test_Goal/eps_x'),
            rospy.get_param('/Test_Goal/eps_y'),
            rospy.get_param('/Test_Goal/eps_z')
        ], dtype=np.float32)

        # Parametri azioni/laser
        self.new_ranges           = rospy.get_param('/Tiago/new_ranges')
        self.max_laser_value      = rospy.get_param('/Tiago/max_laser_value')
        self.min_laser_value      = rospy.get_param('/Tiago/min_laser_value')
        self.max_linear_velocity  = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity  = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')
        self.min_range            = rospy.get_param('/Tiago/min_range')

        # Reward weights
        self.collision_weight    = rospy.get_param("/Reward_param/collision_weight")
        self.guide_weight        = rospy.get_param("/Reward_param/guide_weight")
        self.proximity_weight    = rospy.get_param("/Reward_param/proximity_weight")
        self.collision_reward    = rospy.get_param("/Reward_param/collision_reward")
        self.obstacle_proximity  = rospy.get_param("/Reward_param/obstacle_proximity")
        self.distance_weight     = rospy.get_param("/Reward_param/distance_weight")

        # Training params
        self.single_goal   = rospy.get_param("/Training/single_goal")
        self.dyn_path      = rospy.get_param("/Training/dyn_path")
        self.waypoint_dist = rospy.get_param("/Training/dist_waypoint")
        self.n_waypoint    = rospy.get_param('/Training/n_waypoint')
        self.ahead_dist    = rospy.get_param("/Training/ahead_dist")

        self.algo = rospy.get_param("/Training/attention_method")

        # --- Observation space (Gymnasium) ---
        """laser_scan = self._check_laser_scan_ready()
        num_laser_readings = int(len(laser_scan.ranges))
        rospy.logdebug("num_laser_readings : %d", num_laser_readings)

        # low/high come array e dtype esplicito
        high = np.full((num_laser_readings,), laser_scan.range_max, dtype=np.float32)
        low  = np.full((num_laser_readings,), laser_scan.range_min, dtype=np.float32)"""

        self.observation_space = spaces.Dict({
            "cartesian_scan": spaces.Box(-np.inf, np.inf, shape=(2*120,), dtype=np.float32),
            "waypoints":      spaces.Box(-np.inf, np.inf, shape=(2*self.n_waypoint,),   dtype=np.float32),
            "tagd":           spaces.Box(-np.inf, np.inf, shape=(4*20,), dtype=np.float32),
        })

        # --- Action space (Gymnasium) ---
        min_velocity = np.array([self.min_linear_velocity,  self.min_angular_velocity], dtype=np.float32)
        max_velocity = np.array([self.max_linear_velocity,  self.max_angular_velocity], dtype=np.float32)
        self.action_space = spaces.Box(low=min_velocity, high=max_velocity, dtype=np.float32)

        rospy.loginfo("ACTION SPACE => %s", str(self.action_space))
        rospy.logdebug("OBS SPACE    => %s", str(self.observation_space))

        # Stato interno
        self.curr_robot_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.max_stationary_step = 0

        # Chiamata al base env (assicurati che tiago_env.TiagoEnv sia già Gymnasium-ready)
        super(TiagoNav, self).__init__()

        self.rospack = rospkg.RosPack()

       
        self.prev_goal_dis = 0.0

        self.mean_linear_velocity  = 0.0
        self.mean_angular_velocity = 0.0
        self.n_step = 0

        self.environment_management = obstacles_management.Environment_Management()

        self.path = []

        # Parametri laser
        self.initial_angle   = rospy.get_param("/Tiago/initial_angle")
        self.angle_increment = rospy.get_param("/Tiago/angle_increment")
        self.n_discard_scan  = rospy.get_param("/Tiago/remove_scan")
        self.bounded_ray     = rospy.get_param("/Training/max_ray_value")

        self.phase = '/' + rospy.get_param("/Curriculum_param/curriculum_phase")
        self.orientation = rospy.get_param(self.phase + "/orientation")

    def _set_init_pose(self):
        #reset simulation
        self.environment_management.remove_obstacles()
        while True :
            x , y, goal_x, goal_y = self.environment_management.generate_coords()
            self.goal_pos[0] = goal_x
            self.goal_pos[1] = goal_y
            self.path = self.goal_setting(self.goal_pos[0], self.goal_pos[1], 0.0, 0.0, x , y)
            if self.path is not None:
                rospy.logdebug("Path generated successfully")
                break
        
        self.gazebo_reset(x , y , self.orientation)
        
        # Short delay for stability
        rospy.sleep(0.5)
        self.base_coord = self.amcl_position()
        self.prev_goal_dis = np.linalg.norm(self.base_coord[:2] - self.goal_pos[:2])
        
        self.move_base( 0.0,
                        0.0)
        
        self.environment_management.obstacles_generation(self.path)
    
        rospy.sleep(0.5)
        
        self.K, self.translation, _   = self.get_camera_parameters()

        # trasformazione laser->camera
        translation_laser_camera = np.array([0.0140, 0.0469, 1.1215], dtype=np.float32)  # [x,y,z] m
        rotation_laser_camera    = np.array([0.5007, -0.5008, 0.4992, -0.4991], dtype=np.float32)  # [qx,qy,qz,qw]
        qx, qy, qz, qw = rotation_laser_camera
        R = np.array([
            [1-2*(qy*qy+qz*qz),  2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),  1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
        ], dtype=np.float32)
        T = np.eye(4, dtype=np.float32); T[:3, :3] = R; T[:3, 3] = translation_laser_camera
        self.T_base_from_cam = T


        if self.algo == "Depth_Temporal" or self.algo == "Depth":
            depth_image = self.get_processed_depth_image()
            bounded_scan , _ = self.fused_laser_scan(depth_image , self.get_laser_scan())
            _ , cartesian_scan = self.discretize_scan_observation(self.get_laser_scan()) 
        else:
            bounded_scan , cartesian_scan = self.discretize_scan_observation(self.get_laser_scan()) 
        #bounded_scan , cartesian_scan = self.discretize_scan_observation(self.get_laser_scan())
        self.laser_data = bounded_scan        
        self.prev_scan = np.asarray(cartesian_scan, dtype=float).reshape(-1, 2)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self.terminated = False
        self.truncated = False
        if self.n_step >0:
            self.publish_velocity(self.mean_linear_velocity/self.n_step, self.mean_angular_velocity/self.n_step)
        self.cumulated_steps = 0
        self.mean_angular_velocity = 0.0
        self.mean_linear_velocity = 0.0
        self.n_step = 0
        self.info = {}
        self.info["is_success"] = False
        
        

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the tiago
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        """with open("/home/violo/tesi_project/src/tiago_navigation/data/linear_velocity.txt", 'a') as file:
            # Append the new data to the end of the file
            file.write(str(action[0]) + "\n")

        with open("/home/violo/tesi_project/src/tiago_navigation/data/angular_velocity.txt", 'a') as file:
            # Append the new data to the end of the file
            file.write(str(action[1]) + "\n")"""
        #linear = ((action[0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity)
        #angular = ((action[1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity)
        self.mean_linear_velocity += action[0]
        self.mean_angular_velocity += action[1]
        self.n_step += 1
        # We tell Tiago the linear and angular speed to set to execute
        self.move_base(action[0] , action[1])
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TiagoEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        #update obstacles position
        self.environment_management.dynamic_obstacles_movement()
        # We get the laser scan data
        
        #rospy.loginfo("Laser scan data : " + str(laser_scan))
        #rospy.loginfo("Laser scan data : " + str(laser_scan))
        self.base_coord = self.amcl_position()
        #self.path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , base_coord[0] , base_coord[1])
        info = {}
        #rospy.loginfo("Base coord : " + str(base_coord))

        """if self.dyn_path:
            new_path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , base_coord[0] , base_coord[1])
            if new_path is not None:
                self.path = new_path"""
        if self.algo == "Depth_Temporal" or self.algo == "Depth":
            depth_image = self.get_processed_depth_image()
            bounded_scan , fused_cartesian_scan = self.fused_laser_scan(depth_image , self.get_laser_scan())
            original_bounded , cartesian_scan = self.discretize_scan_observation(self.get_laser_scan()) 
            
        else:
            bounded_scan , cartesian_scan = self.discretize_scan_observation(self.get_laser_scan()) 
        self.laser_data = bounded_scan
        #rospy.loginfo("Discretize scan data : " + str(discretized_observations))
        waypoints , final_pos = self.find_upcoming_waypoint(self.path , self.base_coord[:2] , self.n_waypoint , self.base_coord[3])
        #bounded_scan = self.gen_bounded_scan(discretized_observations.copy())
        #scan,_,_ = self.generate_rays(bounded_scan.copy() , self.n_discard_scan , self.initial_angle  , self.angle_increment)
        curr_scan = np.asarray(cartesian_scan.copy(), dtype=float).reshape(-1, 2)
        #curr_scan = np.asarray(new_laser_scan.copy(), dtype=float).reshape(-1, 2)
        tagd = self.generate_tagd(self.prev_scan, curr_scan)
        #self.tagd_plot(self.prev_scan, curr_scan, tagd)
        tagd = np.concatenate([np.concatenate(pair) for pair in tagd]).tolist()
        waypoints = [element for tuple in waypoints for element in tuple]
        #depth_image_processed = self.image_processing(depth_image)
        
        self.prev_scan = curr_scan
        
        rospy.logdebug("END Get Observation ==>")
        #control of stationary of robot
        
        """if np.linalg.norm(self.curr_robot_pos - base_coord[:3]) < 0.001:
            self.max_stationary_step += 1
        else:
            self.curr_robot_pos = base_coord[:3]
            self.max_stationary_step = 0 """  
        if self.algo == "Depth_Temporal" or self.algo == "Depth" :
            cartesian_scan = fused_cartesian_scan
        #return discretized_observations #, info
        obs = {
            "cartesian_scan": np.array(cartesian_scan).astype(np.float32, copy=False),  # (2*N_RAYS,) float32
            "waypoints":      np.array(waypoints).astype(np.float32, copy=False),    # (2*n_waypoint,) float32
            "tagd":           np.array(tagd).astype(np.float32, copy=False),   # (TAGD_DIM,) float32
        }

        return obs , info
    
    def find_upcoming_waypoint(self, path_coords, robot_pos, n_waypoint, yaw):
        waypoints = []
        # 1. closest vertex
        closest_idx, _ = self.find_closest_waypoint(path_coords, robot_pos)

        # 2. march forward at fixed spacing
        lookahead_d = [self.waypoint_dist * k for k in range(1, n_waypoint + 1)]
        accum = 0.0
        i = closest_idx
        for d in lookahead_d:
            # advance along the poly-line until cumulative length ≥ desired
            while i < len(path_coords) - 1:
                seg_len = np.hypot(*(path_coords[i + 1] - path_coords[i]))
                if accum + seg_len >= d:
                    break
                accum += seg_len
                i += 1
            if i == len(path_coords) - 1:
                # ran out of path → use final vertex
                waypoint = path_coords[-1]
            else:
                # linear interpolation inside the segment
                ratio = (d - accum) / seg_len
                waypoint = path_coords[i] + ratio * (path_coords[i + 1] - path_coords[i])
            waypoints.append(self.convert_global_to_robot_coord(*waypoint, robot_pos, yaw))

        # 3. terminal point (goal) in robot frame
        final_pos = [self.convert_global_to_robot_coord(*path_coords[-1], robot_pos, yaw)]
        return waypoints, final_pos
    
    def convert_global_to_robot_coord(self , x_i , y_i , robot_pos , yaw):

        # Translate
        x_prime = x_i - robot_pos[0]
        y_prime = y_i - robot_pos[1]
            
        # Rotate
        x_double_prime = x_prime * math.cos(yaw) + y_prime * math.sin(yaw)
        y_double_prime = -x_prime * math.sin(yaw) + y_prime * math.cos(yaw)

        #if self.norm_input:
            # Convert to simple float values
        x_double_prime = float(x_double_prime)/3.5
        y_double_prime = float(y_double_prime)/3.5

        """if x_double_prime > 1.0:
            x_double_prime = 1.0
        elif x_double_prime < -1.0:
            x_double_prime = -1.0       
        if y_double_prime > 1.0:    
            y_double_prime = 1.0
        elif y_double_prime < -1.0:
            y_double_prime = -1.0"""

        return x_double_prime , y_double_prime
        

    def _is_done(self, observations ):
        #info= {}
        if min(self.laser_data) <= self.min_range :
            rospy.logerr("Tiago is Too Close to wall==>")
            self.truncated = True  
            self.info["is_success"] = False

        if abs(self.goal_pos[0] - self.base_coord[0]) < self.goal_eps[0] and abs(self.goal_pos[1] - self.base_coord[1]) < self.goal_eps[1] :
            self.terminated = True
            self.info["is_success"] = True
            rospy.loginfo("Goal reach at position : " + str(self.base_coord))
        
        #control if robot are block
        """if self.max_stationary_step == 30:
            self.max_stationary_step = 0
            self._episode_done = True
            rospy.loginfo("Robot are block!")"""
              

        return self.terminated , self.truncated , self.info

    def _compute_reward(self, observations):

        #base_coord = self.amcl_position()
        reward = 0

        #ORIGINAL REWARD FUNCTION
        #collision reward
        """collision_reward = 0.0
        if min(self.laser_data) < self.min_range :
            collision_reward = self.collision_weight * self.collision_reward
            reward += collision_reward


        #proximity reward   
        proximity_reward = 0.0
        if min(self.laser_data) < 0.8: 
            proximity_reward = -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , min(observations['laser_scan'])))
            reward += proximity_reward
        
        #guide reward
        guide_reward  = self.guide_reward(self.path , self.base_coord[:2])
        guide_reward = -0.2*guide_reward
        reward += guide_reward

        #robot_vel = self.robot_curr_velocity()
        #reward += -0.05 * (robot_vel[1]**2)

        #publish reward value
        self.publish_reward(collision_reward , proximity_reward , guide_reward , 0 , reward)"""
        
        
        
        #SECOND REWARD FUNCTION
        robot_vel = self.robot_curr_velocity()
        d_min = min(self.laser_data)
        collision_reward = 0.0
        if d_min < self.min_range:
            collision_reward = self.collision_weight * self.collision_reward
            reward += -50

        if d_min < 0.45:
            #collision = -self.proximity_weightabs( self.obstacle_proximity - min(self.obstacle_proximity , collision_distance))
            #obstalce_reward = -self.proximity_weightabs( self.obstacle_proximity - min(self.obstacle_proximity , min(observations)))
            proximity_reward = -1.5*abs( 0.45 - min(0.45 , d_min)) #for correct model 1.5
            reward += proximity_reward
        
        if d_min < 1.2:
            reward -= 0.5 * (1.2 / (d_min + 1e-3) - 1.2)  # 0 a 1m, cresce sotto 1m

        curr_dist = np.linalg.norm(self.base_coord[:2] - self.goal_pos[:2])
        distance_error = 5.0*(self.prev_goal_dis - curr_dist)
        #distance_error = -np.linalg.norm(self.goal_pos - base_coord[:3])
        reward += distance_error
        self.prev_goal_dis = curr_dist
        
        #if robot_vel[1] > 0.3:
        #reward += -0.1 * (robot_vel[1] ** 2)
        reward += -0.1
        #angular reward
        angular_reward = 0.0
        # compute angle between robot’s heading and vector to goal
        goal_vec = self.goal_pos - self.base_coord[:3]
        heading = self.base_coord[3] # radians
        yaw_to_goal = np.arctan2(goal_vec[1], goal_vec[0])
        yaw_error = (((yaw_to_goal - heading) + math.pi) % (2*math.pi)) - math.pi
        if curr_dist > 1.0 and d_min > 0.5: 
            if robot_vel[0] > 0.05:
                angular_reward = 0.5 * np.cos(yaw_error)
        reward += angular_reward
        reward += -0.1 * (robot_vel[1] ** 2)
        #guide_reward  = self.guide_reward(self.path , self.base_coord[:2])
        #guide_reward = -0.5*guide_reward
        #reward += guide_reward
        #reward += 0.5*(robot_vel[0] - abs(robot_vel[1]))
        if (abs(self.goal_pos[0] - self.base_coord[0]) < self.goal_eps[0] and
            abs(self.goal_pos[1] - self.base_coord[1]) < self.goal_eps[1]):
            reward += 50

        self.publish_reward(collision_reward , distance_error , 0 , 0 , reward)
        
        return reward

        
    
    def compute_exploration_reward(self, current_position, weight=0.2):
        """
        Incoraggia l'esplorazione quando il robot è bloccato
        """
        if not hasattr(self, 'position_history'):
            self.position_history = []
            self.stuck_counter = 0
        
        self.position_history.append(current_position[:2].copy())
        
        # Mantieni solo le ultime 10 posizioni
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Controlla se il robot è bloccato (movimento molto limitato)
        if len(self.position_history) >= 5:
            recent_positions = np.array(self.position_history[-5:])
            movement_variance = np.var(recent_positions, axis=0).sum()
            
            if movement_variance < 0.01:  # Soglia per "bloccato"
                self.stuck_counter += 1
                # Incoraggia movimento laterale o rotazione
                return weight * self.stuck_counter
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
        
        return 0.0
    
    def publish_reward(self , collision_reward , proximity_reward , guide_reward , angular_reward , reward):
        # Create and publish multi-array message
        reward_msg = Float32MultiArray()
        reward_msg.data = [collision_reward , proximity_reward , guide_reward , angular_reward , reward]
        self.reward_debug_pub.publish(reward_msg)

    def guide_reward(self,
                     path_coords: List[Tuple[float, float]],
                     robot_pos: Tuple[float, float]) -> float:
        """
        Compute guidance reward: negative Euclidean distance from robot to
        the interpolated guide point exactly `guide_distance` ahead on the path.
        """
        # Find closest waypoint
        closest_idx, _ = self.find_closest_waypoint(path_coords, robot_pos)
        # Interpolate the exact point `guide_distance` ahead
        goal_position = self.calculate_goal_position(path_coords,
                                                     closest_idx,
                                                     self.ahead_dist)
        # Compute and return reward
        return np.linalg.norm(np.array(goal_position) - np.array(robot_pos))


    def find_closest_waypoint(self,
                              path_coords: List[Tuple[float, float]],
                              robot_pos: Tuple[float, float]) -> Tuple[int, float]:
        """
        Return index and distance of the closest waypoint to the robot.
        """
        path = np.array(path_coords, dtype=np.float32)
        robot = np.array(robot_pos, dtype=np.float32)
        dists = np.linalg.norm(path - robot, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def calculate_goal_position(self,
                                path_coords: List[Tuple[float, float]],
                                closest_idx: int,
                                cum_distance: float) -> Tuple[float, float]:
        """
        Walk along the path, interpolating within the segment where
        cumulative distance reaches `cum_distance`.
        """
        path = np.array(path_coords, dtype=np.float32)
        # If at or beyond final waypoint, return last
        if closest_idx >= len(path) - 1:
            return tuple(path[-1])

        travelled = 0.0
        curr_idx = closest_idx
        curr_point = path[curr_idx]

        # Traverse segments
        while curr_idx < len(path) - 1 and travelled < cum_distance:
            next_point = path[curr_idx + 1]
            seg_len = np.linalg.norm(next_point - curr_point)
            remaining = cum_distance - travelled

            if seg_len >= remaining:
                # Interpolate within this segment
                t = remaining / seg_len
                interp_point = curr_point + t * (next_point - curr_point)
                return (float(interp_point[0]), float(interp_point[1]))

            # Move to next waypoint
            travelled += seg_len
            curr_idx += 1
            curr_point = next_point

        # If end is reached without full distance, return last waypoint
        return tuple(path[-1])
    
    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        new_data = data.ranges[20:-20]
        self._episode_done = False
        
        discretized_ranges = []
        #mod = len(data.ranges)/new_ranges
        
        for i, item in enumerate(new_data):
            #if (i%new_ranges==0):
            if item == float ('inf') or np.isinf(item) or (float((item)) > self.max_laser_value):
                discretized_ranges.append(self.max_laser_value)
            elif np.isnan(item) or (float(item) < self.min_laser_value):
                discretized_ranges.append(self.min_laser_value)
            else:
                discretized_ranges.append(float(item))
                """    
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                """    
        rospy.logdebug("New observation dimension : " + str(len(discretized_ranges)))
        
        bounded_scan = self.gen_bounded_scan(discretized_ranges)
        cartesian_scan = self.generate_rays(bounded_scan[13:-13].copy() , self.n_discard_scan , self.initial_angle  , self.angle_increment)
        
        return bounded_scan, cartesian_scan
    
    def publish_velocity(self , linear_velocity, angular_velocity ):
        # Create a Float32MultiArray message
        vel_msg = Float32MultiArray()
        vel_msg.data = [linear_velocity, angular_velocity]

        # Publish the message
        self.velocity_pub.publish(vel_msg)
    

    ##### LASER SCAN PREPROCESSING METHODS #####
    def gen_bounded_scan(self , laser_scan , max_range = 3.5):        
        for i in range(0 , len(laser_scan)):
            if laser_scan[i] >= max_range:
                laser_scan[i] = max_range
            #if rospy.get_param('/Training/input_normalization'): 
            #laser_scan[i] = laser_scan[i]/3.5    
        return laser_scan    
    
    def generate_rays(self , laser_scan, n_discard_scan, initial_angle, angle_increment):
        # 1) throw away the first & last n_discard_scan readings
        valid_ranges = laser_scan
        # angles start at initial_angle + n_discard_scan*angle_increment
        start_angle = initial_angle + n_discard_scan * angle_increment

        # how many rays (groups) we want
        n = int(rospy.get_param("/Spatial_Attention/n_rays"))
        # size of each group of original scans
        group_size = len(valid_ranges) // n

        cartesian_scan   = []      # flat [x0, y0, x1, y1, …]

        for g in range(n):
            # slice out this group
            start_idx = g * group_size
            end_idx   = start_idx + group_size
            group_ranges = valid_ranges[start_idx:end_idx]

            # angles for this group
            group_angles = start_angle + (start_idx + np.arange(group_size)) * angle_increment

            # index of the closest hit in the group
            k        = int(np.argmin(group_ranges))
            r_min    = float(group_ranges[k])
            theta_k  = group_angles[k]

            # Cartesian coordinates *normalised by MAX_LIDAR_RANGE*
            x = (r_min * math.cos(theta_k)) / 3.5
            y = (r_min * math.sin(theta_k)) / 3.5

            # push into the flat scan vector
            cartesian_scan.append(x) 
            cartesian_scan.append(y) 

        # sanity‐check
        if len(cartesian_scan) != 2 * n:
            rospy.logerr(f"generate_rays: expected {2*n} coords, got {len(cartesian_scan)}")

        # build your point‐cloud
        #pointcloud = np.column_stack((np.array(x_arr), np.array(y_arr)))

        return cartesian_scan #, pointcloud, cartesian_scan_raw
    
    def icp_align(self,
                B_prev: np.ndarray,
                B_curr: np.ndarray,
                max_iter: int = 50,
                tolerance: float = 1e-4) -> np.ndarray:
        """
        Align B_prev to B_curr with a full 2-D rigid transform.

        Parameters
        ----------
        B_prev, B_curr : (N,2) arrays
        """
        P = B_prev.copy()

        for _ in range(max_iter):
            # 1) nearest-neighbour correspondences
            _, idx = cKDTree(B_curr).query(P)
            Q = B_curr[idx]

            # 2) best-fit rotation R and translation t  (Horn 1987)
            cP = P.mean(axis=0)
            cQ = Q.mean(axis=0)
            X  = P - cP
            Y  = Q - cQ
            U, _, Vt = np.linalg.svd(X.T @ Y)
            R = Vt.T @ U.T
            # guard against reflection
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            t = cQ - R @ cP

            P_new = (R @ P.T).T + t
            if np.linalg.norm(P_new - P) < tolerance:
                break
            P = P_new

        return P


    ###############################################################################
    # 3. TAGD generation with the paper’s defaults                               #
    ###############################################################################
    def _wrap_angle(self , a):
        # wrap to (-pi, pi]
        return (a + np.pi) % (2*np.pi) - np.pi

    def _closest_by_angle(self , P_cart, theta_points, theta_ref):
        """Return the point in P_cart whose angle is closest to theta_ref."""
        if P_cart.size == 0:
            return None
        diff = np.abs(self._wrap_angle(theta_points - theta_ref))
        j = np.argmin(diff)
        return P_cart[j]

    def generate_tagd(self,
                    B_prev: np.ndarray,
                    B_curr: np.ndarray,
                    d_thresh: float = 0.25,
                    d_max: float   = 3.5,
                    Nc: int        = 20,
                    fov_center_deg: float = 0.0,
                    fov_deg: float = 200.0):
        """
        Always return exactly Nc TAGDs over a limited lidar FOV (no NaNs).

        Each TAGD is a tuple (c_t, c_t-1).
        - If a sector has points within d_thresh of its center g_i, use centroids.
        - Else fall back to the nearest-by-angle point in each scan.
        - If a scan has no points at all (edge case), use g_i for that side.
        """
        TAGDs = []

        # 1) Align previous to current (rigid 2D)
        B_prev_aligned = self.icp_align(B_prev, B_curr)

        # Precompute polar for both scans
        theta_curr = np.arctan2(B_curr[:, 1], B_curr[:, 0])
        r_curr     = np.linalg.norm(B_curr, axis=1)

        theta_prev = np.arctan2(B_prev_aligned[:, 1], B_prev_aligned[:, 0])
        r_prev     = np.linalg.norm(B_prev_aligned, axis=1)

        # 2) Sectors over the 200° FOV
        fov_center = np.deg2rad(fov_center_deg)
        fov = np.deg2rad(fov_deg)
        theta_min = self._wrap_angle(fov_center - fov/2)
        thetas_ref = theta_min + np.linspace(0.0, fov, Nc, endpoint=False)
        thetas_ref = self._wrap_angle(thetas_ref)
        sector_half_width = (fov / Nc) / 2.0

        for theta_ref in thetas_ref:
            # Points in current scan inside this sector
            diff_curr = self._wrap_angle(theta_curr - theta_ref)
            mask_curr = np.abs(diff_curr) <= sector_half_width

            # Sector center g_i (range = closest observed in-sector or d_max if empty)
            if mask_curr.any():
                r_min = min(r_curr[mask_curr].min(), d_max)
            else:
                r_min = d_max
            g_i = np.array([r_min * np.cos(theta_ref),
                            r_min * np.sin(theta_ref)])

            # Grouping around g_i (current & aligned previous)
            d_to_g_curr = np.linalg.norm(B_curr         - g_i, axis=1)
            d_to_g_prev = np.linalg.norm(B_prev_aligned - g_i, axis=1)

            G_t    = B_curr[d_to_g_curr <= d_thresh]
            G_tm1  = B_prev_aligned[d_to_g_prev <= d_thresh]

            # Choose representative points for this sector for both scans
            if G_t.size > 0:
                c_t = G_t.mean(axis=0)
            else:
                # Fallback: nearest-by-angle in current scan, else g_i
                p = self._closest_by_angle(B_curr, theta_curr, theta_ref)
                c_t = p if p is not None else g_i

            if G_tm1.size > 0:
                c_tm1 = G_tm1.mean(axis=0)
            else:
                # Fallback: nearest-by-angle in previous-aligned scan, else g_i
                p = self._closest_by_angle(B_prev_aligned, theta_prev, theta_ref)
                c_tm1 = p if p is not None else g_i

            TAGDs.append((c_t, c_tm1))
            #TAGDs_norm = [(c_t/3.5, c_tm1/3.5) for (c_t, c_tm1) in TAGDs]

        return TAGDs
    
    ###############################################################################
    # DEPTH IMAGE PROCESSING                                                      #
    ###############################################################################
    
    def image_processing(self , image):
            gamma = 0.10
            stride_n = 8
            sensor_max = 3.5  # max range del sensore (metri)
            h_r = self.translation[2]  # NOTE: se il tuo world frame è Y-up, usa translation[1]

            H, W = image.shape
            u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

            # --- Bound e sanitizzazione profondità --------------------------------
            Zc = image.astype(np.float32)
            # invalidiamo valori non finiti o <=0
            Zc[~np.isfinite(Zc)] = np.nan
            Zc[Zc <= 0.0] = np.nan
            # applichiamo bound superiore (chi vuole saturare può usare clip,
            # qui invece invalidiamo > sensor_max così restano solo punti nel range)
            Zc[Zc > sensor_max] = np.nan

            # --- Geometria: Y in camera frame -------------------------------------
            fy, cy = self.K[4], self.K[5]
            Yc = (v_grid - cy) * Zc / fy

            # Maschera "floor band": |Yc - h_r| < gamma  -> invalidiamo
            floor_band = np.abs(Yc - h_r) < gamma

            # Profondità filtrata: solo validi restano numerici, il resto NaN
            F = Zc.copy()
            F[floor_band] = np.nan

            # --- Layout a strisce --------------------------------------------------
            W_trim = (W // stride_n) * stride_n
            F_trim = F[:, :W_trim]
            n_stripes = W_trim // stride_n

            # Reshape a (H, n_stripes, stride_n)
            stripes = F_trim.reshape(H, n_stripes, stride_n)

            # --- Minimo per striscia ignorando NaN --------------------------------
            # ranges: minimo valido per striscia; se una striscia è tutta NaN -> NaN
            ranges = np.nanmin(stripes, axis=(0, 2))  # shape: (n_stripes,)
            valid_stripes = np.isfinite(ranges)       # True se esiste almeno un punto valido

            # --- Indici (row/col) del punto di minimo per le sole strisce valide ---
            # Per ottenere gli argmin ignorando NaN, sostituiamo i NaN con +inf SOLO per argmin
            stripes_pos = np.where(np.isfinite(stripes), stripes, np.inf)
            stripes_flat = stripes_pos.reshape(H * stride_n, n_stripes)  # (H*stride_n, n_stripes)
            min_indices_all = np.argmin(stripes_flat, axis=0)            # anche per le non valide, ma le filtriamo dopo

            # Mappiamo l'indice piatto a (riga, colonna)
            rows_all = (min_indices_all // stride_n).astype(np.int32)       # 0..H-1
            cols_within_all = (min_indices_all % stride_n).astype(np.int32) # 0..stride_n-1
            cols_all = (np.arange(n_stripes) * stride_n + cols_within_all).astype(np.int32)

            # --- Filtriamo per tenere SOLO le strisce valide -----------------------
            stripe_indices = np.nonzero(valid_stripes)[0]    # indici delle strisce valide
            ranges_valid = ranges[valid_stripes].astype(np.float32)
            rows_valid = rows_all[valid_stripes]
            cols_valid = cols_all[valid_stripes]

            # --- Normalizzazione per la NN (solo sulle strisce valide) -------------
            M_norm = ranges_valid / sensor_max  # shape: (n_valid_stripes,)

            # --- Back-projection 3D dei minimi validi ------------------------------
            fx, cx = self.K[0], self.K[2]
            fy, cy = self.K[4], self.K[5]

            u = cols_valid.astype(np.float32)
            v = rows_valid.astype(np.float32)
            z = ranges_valid.astype(np.float32)

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_3d = np.column_stack([x, y, z])  # (N, 3) solo per strisce valide

            
            return M_norm, F, points_3d, rows_valid, cols_valid, stripe_indices

    
    ###############################################################################
    # New laser scan generation                                                   #
    ###############################################################################

    def camera_points_to_virtual_scan(self ,points_cam, T_base_from_cam,
                                  angle_min=-np.pi, angle_max=np.pi, angle_inc=np.deg2rad(1.0),
                                  range_min=0.05, range_max=25.0,
                                  keep_front_only=True):
        """
        Converte punti 3D in camera frame in uno scan 2D virtuale nel frame base_footprint
        usando r = min range per bin angolare (ignora l'altezza).
        """
        if points_cam.size == 0:
            K = int(np.floor((angle_max - angle_min)/angle_inc)) + 1
            angles = angle_min + np.arange(K)*angle_inc
            return np.full(K, np.inf, np.float32), angles.astype(np.float32)

        # Omogeneizza e trasforma in base_footprint
        N = points_cam.shape[0]
        pts_cam_h = np.hstack([points_cam.astype(np.float32), np.ones((N,1), np.float32)])  # (N,4)
        pts_base  = (T_base_from_cam @ pts_cam_h.T).T[:, :3]                                # (N,3)

        # Se vuoi solo avanti (tipico per lidar frontale)
        mask = np.isfinite(pts_base).all(axis=1)
        if keep_front_only:
            mask &= pts_base[:,0] > 0.0
        pts = pts_base[mask]

        if pts.size == 0:
            K = int(np.floor((angle_max - angle_min)/angle_inc)) + 1
            angles = angle_min + np.arange(K)*angle_inc
            return np.full(K, np.inf, np.float32), angles.astype(np.float32)

        # Converti in polar (XY del base)
        x = pts[:,0].astype(np.float32)
        y = pts[:,1].astype(np.float32)
        r  = np.hypot(x, y)
        th = np.arctan2(y, x)

        # Filtri FOV e range
        valid = (th >= angle_min) & (th <= angle_max) & (r >= range_min) & (r <= range_max)
        r, th = r[valid], th[valid]

        # Bin angolare e min-range per bin
        K = int(np.floor((angle_max - angle_min)/angle_inc)) + 1
        angles = angle_min + np.arange(K)*angle_inc
        ranges = np.full(K, np.inf, np.float32)
        if r.size > 0:
            bin_idx = np.clip(((th - angle_min)/angle_inc).astype(int), 0, K-1)
            np.minimum.at(ranges, bin_idx, r)

        return ranges, angles.astype(np.float32)

    # -------------------------------
    # 2) Fusione con LiDAR reale
    # -------------------------------
    def fuse_scans_min(self, lidar, angles_lidar, cam, angles_cam):
        """
        Fusione per-angolo con 'controllo minimo':
        prendi il valore della depth SOLO se è più vicino del LiDAR di almeno min_diff.
        Niente analisi percentuali o soglie adattive: una sola soglia fissa.

        Attributo opzionale:
        self.min_diff : differenza minima [m] perché la depth sovrascriva il LiDAR (default 0.12 m)
        """
        import numpy as np

        if len(lidar) != len(cam):
            raise ValueError("Gli scan devono avere stessa dimensione/risoluzione angolare.")
        if not np.allclose(angles_lidar, angles_cam, atol=1e-6):
            raise ValueError("Gli array degli angoli devono combaciare (min/max/increment).")

        #lidar = np.asarray(ranges_lidar, dtype=np.float32)
        #cam   = np.asarray(ranges_cam,   dtype=np.float32)

        # Soglia fissa (grande differenza)
        min_diff = float(getattr(self, "min_diff", 0.12))  

        # Usa la depth solo se è valida e significativamente più corta del LiDAR
        cam_valid = np.isfinite(cam) & (cam > 0.0)
        use_cam = cam_valid & ((lidar - cam) > min_diff)
        fused = lidar.copy()
        fused[use_cam] = cam[use_cam]
        return fused


    
    def generate_angles_from_ranges(self,ranges_lidar, angle_min, angle_increment):
        n = len(ranges_lidar)
        angles_lidar = angle_min + np.arange(n) * angle_increment
        return angles_lidar

    def fused_laser_scan(self , depth_image , laser_scan ):

        M_norm, F, points_3d, rows, cols, valid = self.image_processing(depth_image)

        new_initial_angle = self.initial_angle + (20 * self.angle_increment)

        # 2) Crea lo scan virtuale dalla camera
        ranges_cam, angles_cam = self.camera_points_to_virtual_scan(
            points_cam=points_3d,
            T_base_from_cam=self.T_base_from_cam,
            angle_min=new_initial_angle, angle_max= -1 * new_initial_angle, angle_inc=self.angle_increment,
            range_min=0.05, range_max=3.5,
            keep_front_only=True
        )
        #laser_scan = laser_scan.ranges[20:-20]
        ranges_lidar = np.array(laser_scan.ranges[20:-20], dtype=np.float32)
        angles_lidar = self.generate_angles_from_ranges(ranges_lidar, new_initial_angle, self.angle_increment)

        # 3)fusione con lo scan reale del LiDAR
        ranges_fused = self.fuse_scans_min(ranges_lidar, angles_lidar, ranges_cam, angles_cam)
        #self.plot_scans(ranges_lidar, angles_lidar, ranges_cam, angles_cam, ranges_fused)

        bounded_scan = self.gen_bounded_scan(ranges_fused)
        #rospy.loginfo("Fused scan: " + str(bounded_scan))
        cartesian_scan = self.generate_rays(bounded_scan[13:-13].copy() , self.n_discard_scan , self.initial_angle  , self.angle_increment)
        #rospy.loginfo("Cartesian fused scan: " + str(cartesian_scan))
        return bounded_scan , cartesian_scan
    
    def tagd_plot(self , B_prev_cartesian, B_curr_cartesian, TAGDs):
        # Plot before alignment
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(B_prev_cartesian[:, 0], B_prev_cartesian[:, 1], c='blue', label='B_prev_cartesian', alpha=0.6)
        plt.scatter(B_curr_cartesian[:, 0], B_curr_cartesian[:, 1], c='red', label='B_curr_cartesian', alpha=0.6)
        plt.title('Before Alignment')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.axis('equal')

        # Align B_prev_cartesian to B_curr_cartesian using ICP
        B_prev_aligned = self.icp_align(B_prev_cartesian, B_curr_cartesian)

        plt.subplot(1, 2, 2)
        plt.scatter(B_prev_aligned[:, 0], B_prev_aligned[:, 1], c='blue', label='B_prev_aligned', alpha=0.6)
        plt.scatter(B_curr_cartesian[:, 0], B_curr_cartesian[:, 1], c='red', label='B_curr_cartesian', alpha=0.6)

        # Plot TAGD centroids
        for c_t, c_t_prev in TAGDs:
            plt.plot([c_t_prev[0], c_t[0]], [c_t_prev[1], c_t[1]], 'g-o', linewidth=2, markersize=5)

        plt.title('After Alignment with TAGD')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.axis('equal')

        plt.tight_layout()
        plt.show()

    def plot_scans(self,ranges_lidar, angles_lidar, ranges_cam = None, angles_cam = None, ranges_fused=None):
        """
        Plot in XY dei tre insiemi: LiDAR, Camera(virtual scan), e Fuso (opzionale).
        """
        def polar_to_xy(r, th):
            mask = np.isfinite(r) & (r < np.inf)
            return (r[mask]*np.cos(th[mask]), r[mask]*np.sin(th[mask]))

        xL, yL = polar_to_xy(ranges_lidar, angles_lidar)
        if ranges_cam is not None or angles_cam is not None:
            xC, yC = polar_to_xy(ranges_cam, angles_cam)

        plt.figure(figsize=(7,7))
        """if xL.size:
            plt.scatter(xL, yL, s=6, label="LiDAR (reale)", alpha=0.8)
        if ranges_cam is not None or angles_cam is not None :
            if xC.size:
                plt.scatter(xC, yC, s=6, label="Camera -> virtual scan", alpha=0.8)"""

        if ranges_fused is not None:
            xF, yF = polar_to_xy(ranges_fused, angles_lidar)
            if xF.size:
                plt.scatter(xF, yF, s=8, label="Fusione (min per angolo)", alpha=0.9)

        plt.scatter(0, 0, c="red", marker="x", label="Robot (base_footprint)")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xlabel("X [m] (forward)")
        plt.ylabel("Y [m] (left)")
        plt.title("Confronto: LiDAR vs Camera (virtual scan) in base_footprint")
        #plt.legend()
        plt.show()
    
    