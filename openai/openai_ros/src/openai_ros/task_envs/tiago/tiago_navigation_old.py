import rospy
import numpy
import time
import math
from gym import spaces
from openai_ros.robot_envs import tiago_env
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
#####
import rospy
import numpy as np
from openai_ros.robot_envs import tiago_env
from openai_ros.task_envs.tiago import obstacles_management
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math
import time
from tf.transformations import quaternion_from_euler
import os
import rospkg
import random
import matplotlib.pyplot as plt

from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray
from typing import List, Tuple, Optional

from sklearn.cluster import KMeans
from scipy.spatial import cKDTree


# The path is __init__.py of openai_ros, where we import the TiagoMazeEnv directly
max_episode_steps_per_episode = 150 #75 # Can be any Value

register(
        id='TiagoNavigation-v0',
        entry_point='openai_ros.task_envs.tiago.tiago_navigation:TiagoNav',
        max_episode_steps=max_episode_steps_per_episode,
    )

class TiagoNav(tiago_env.TiagoEnv):
    def __init__(self):
        """
        This Task Env is designed for having the Tiago in some kind of maze.
        It will learn how to move around the maze without crashing.
        """
        self.reward_debug_pub = rospy.Publisher('/tiago_navigation/reward', Float32MultiArray, queue_size=1)
        self.velocity_pub = rospy.Publisher('/mean_velocity', Float32MultiArray, queue_size=1)
        # Only variable needed to be set here
        #number_actions = rospy.get_param('/tiago/n_actions')
        #self.action_space = spaces.Discrete(number_actions)
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

          # Goal position
        self.goal_pos = np.array([
                                rospy.get_param('/Test_Goal/x'),
                                rospy.get_param('/Test_Goal/y'),
                                rospy.get_param('/Test_Goal/z')
                                ])
        
        # Goal position  epsilon , maximum error for goal position 
        self.goal_eps = np.array([
                                    rospy.get_param('/Test_Goal/eps_x'),
                                    rospy.get_param('/Test_Goal/eps_y'),
                                    rospy.get_param('/Test_Goal/eps_z')
                                ])
        
        
        #number_observations = rospy.get_param('/tiago/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """
        
        # Actions and Observations
        self.new_ranges = rospy.get_param('/Tiago/new_ranges')
        self.max_laser_value = rospy.get_param('/Tiago/max_laser_value')
        self.min_laser_value = rospy.get_param('/Tiago/min_laser_value')
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')
        self.min_range = rospy.get_param('/Tiago/min_range') # Minimum meters below wich we consider we have crashed

        #reward weights
        self.collision_weight = rospy.get_param("/Reward_param/collision_weight")
        self.guide_weight = rospy.get_param("/Reward_param/guide_weight")
        self.proximity_weight = rospy.get_param("/Reward_param/proximity_weight")
        self.collision_reward = rospy.get_param("/Reward_param/collision_reward")
        self.obstacle_proximity = rospy.get_param("/Reward_param/obstacle_proximity")
        self.distance_weight = rospy.get_param("/Reward_param/distance_weight")

        #training parameter 
        self.single_goal = rospy.get_param("/Training/single_goal")
        self.dyn_path = rospy.get_param("/Training/dyn_path")
        self.waypoint_dist = rospy.get_param("/Training/dist_waypoint")
        self.n_waypoint = rospy.get_param('/Training/n_waypoint')
        self.ahead_dist = rospy.get_param("/Training/ahead_dist")

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        
        laser_scan = self._check_laser_scan_ready()
        num_laser_readings = int(len(laser_scan.ranges))
        rospy.logdebug("num_laser_readings : " + str(num_laser_readings))
        high = np.full((num_laser_readings), laser_scan.range_max)
        low = np.full((num_laser_readings), laser_scan.range_min)
        
        # Generate observation space
        self.observation_space = spaces.Box(low, high)
    
        # Set possible value of linear velocity and angular velocity 
        min_velocity = [self.min_linear_velocity , self.min_angular_velocity]
        max_velocity = [self.max_linear_velocity , self.max_angular_velocity]

        #Generate action space
        self.action_space = spaces.Box(np.array(min_velocity), np.array(max_velocity))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.curr_robot_pos = np.array([0,0,0])
        #used for control if the robot are block 
        self.max_stationary_step = 0

       
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TiagoNav, self).__init__()
        self.rospack = rospkg.RosPack()
        #self.path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , 0.0 , 0.0)

        self.laser_filtered_pub = rospy.Publisher('/tiago/laser/scan_filtered', LaserScan, queue_size=1)

        self.initial_position = self.gazebo_robot_state()
        self.initx = 1.0
        self.inity = -1.0
        self.initw = self.initial_position[3]
        self.yaw_goal = 0.0
        self.prev_goal_dis = 0.0

        self.mean_linear_velocity = 0.0
        self.mean_angular_velocity = 0.0
        self.n_step = 0
        self.start_x = 0.0
        self.start_y = 0.0

        self.model_name = []
        self.model_coord = []

        self.environment_management = obstacles_management.Environment_Management() 

        self.path = []

        #value for generate the correct version of laser scan 
        self.initial_angle = rospy.get_param("/Tiago/initial_angle")
        self.angle_increment = rospy.get_param("/Tiago/angle_increment")
        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")
        self.bounded_ray = rospy.get_param("/Training/max_ray_value")

        laser_scan = self.get_laser_scan()
        #rospy.loginfo("Laser scan data : " + str(laser_scan))
        _ , cartesian_scan = self.discretize_scan_observation(laser_scan)
        #rospy.loginfo("Discretize scan data : " + str(discretized_observations)) 
        #scan = self.gen_bounded_scan(discretized_observations.copy())
        #rospy.loginfo("Bounded scan data : " + str(scan))
        #scan,_,_ = self.generate_rays(scan.copy() , self.n_discard_scan , self.initial_angle  , self.angle_increment)
        #rospy.loginfo("Rays scan data : " + str(scan))
        self.prev_scan = np.asarray(cartesian_scan, dtype=float).reshape(-1, 2)
        #self.spawn_model_from_sdf('box1' , '/home/violo/tiago_public_ws/src/pal_gazebo_worlds/models/box/box.sdf' , x = 1.0)
        self.reward_max = [-100 , -100 , -100]
        self.reward_min = [100 , 100 , 100]

        self.K , self.translation , _ = self.get_camera_parameters()


    def _set_init_pose(self):
        #reset simulation
        self.environment_management.remove_obstacles()
        while True :
            x , y, goal_x, goal_y = self.environment_management.generate_coords()
            self.goal_pos[0] = goal_x
            self.goal_pos[1] = goal_y
            self.path = self.goal_setting(self.goal_pos[0], self.goal_pos[1], 0.0, 0.0, x , y)
            if self.path is not None:
                rospy.loginfo("Path generated successfully")
                break
        #self.random_init_goal_pos()

        #rospy.loginfo(str(self.initx)+ " " + str(self.inity))
        #self.gazebo_reset(-3.8 , 7.4 , 0.0)
        #reset gazebo position 
        self.gazebo_reset(x , y , 0.0)

        #rospy.logerr("New position setting complete")
        
        # Short delay for stability
        rospy.sleep(0.5)
        base_coord = self.amcl_position()
        self.prev_goal_dis = np.linalg.norm(base_coord[:3] - self.goal_pos[:3])
        #random_index = random.randint(0, (len(self.paths) - 1))
        #self.path = self.paths[random_index]
        #rospy.loginfo(" Initial position : " + str(self.gazebo_robot_state()))
        #rospy.loginfo("Current Path : " + str(self.path))
        rospy.loginfo(str(self.amcl_position()))
        # self.reset_position()
        self.move_base( 0.0,
                        0.0)
        self.environment_management.obstacles_generation(self.path)
        rospy.loginfo("Reward max : " + str(self.reward_max) + " Reward min : " + str(self.reward_min))
        #rospy.loginfo("Path : " + str(self.path))
        rospy.sleep(0.5)
        laser_scan = self.get_laser_scan()
        #rospy.loginfo("Laser scan data : " + str(laser_scan))
        _ , cartesian_scan = self.discretize_scan_observation(laser_scan)
        #rospy.loginfo("Discretize scan data : " + str(discretized_observations)) 
        #scan = self.gen_bounded_scan(discretized_observations.copy())
        #rospy.loginfo("Bounded scan data : " + str(scan))
        #scan,_,_ = self.generate_rays(scan.copy() , self.n_discard_scan , self.initial_angle  , self.angle_increment)
        #rospy.loginfo("Rays scan data : " + str(scan))
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
        self._episode_done = False
        self.truncated = False
        if self.n_step >0:
            self.publish_velocity(self.mean_linear_velocity/self.n_step, self.mean_angular_velocity/self.n_step)
        self.cumulated_steps = 0
        self.mean_angular_velocity = 0.0
        self.mean_linear_velocity = 0.0
        self.n_step = 0
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        
        # TODO: Add reset of published filtered laser readings
        #laser_scan = self.get_laser_scan()
        #discretized_ranges = laser_scan.ranges
        #self.publish_filtered_laser_scan(   laser_original_data=laser_scan,
        #                                 new_filtered_laser_range=discretized_ranges)


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the tiago
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
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
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        depth_image = self.get_processed_depth_image()
        #rospy.loginfo("Laser scan data : " + str(laser_scan))
        #rospy.loginfo("Laser scan data : " + str(laser_scan))
        base_coord = self.amcl_position()
        #self.path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , base_coord[0] , base_coord[1])
        info = {}
        #rospy.loginfo("Base coord : " + str(base_coord))

        """if self.dyn_path:
            new_path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , base_coord[0] , base_coord[1])
            if new_path is not None:
                self.path = new_path"""
        bounded_scan , cartesian_scan = self.discretize_scan_observation(laser_scan) 
        #rospy.loginfo("Discretize scan data : " + str(discretized_observations))
        waypoints , final_pos = self.find_upcoming_waypoint(self.path , base_coord[:2] , self.n_waypoint , base_coord[3])
        #bounded_scan = self.gen_bounded_scan(discretized_observations.copy())
        #scan,_,_ = self.generate_rays(bounded_scan.copy() , self.n_discard_scan , self.initial_angle  , self.angle_increment)
        curr_scan = np.asarray(cartesian_scan.copy(), dtype=float).reshape(-1, 2)
        tagd = self.generate_tagd(self.prev_scan, curr_scan)
        #self.tagd_plot(self.prev_scan, curr_scan, tagd)
        tagd = np.concatenate([np.concatenate(pair) for pair in tagd]).tolist()
        depth_image_processed = self.image_processing(depth_image)
        
        self.prev_scan = curr_scan
        
        rospy.logdebug("END Get Observation ==>")
        #control of stationary of robot
        
        if np.linalg.norm(self.curr_robot_pos - base_coord[:3]) < 0.001:
            self.max_stationary_step += 1
        else:
            self.curr_robot_pos = base_coord[:3]
            self.max_stationary_step = 0 

        observations = {}
        if abs(self.goal_pos[0] - base_coord[0]) < self.goal_eps[0] and abs(self.goal_pos[1] - base_coord[1]) < self.goal_eps[1] :
            self.truncated = True
            rospy.loginfo("Goal reach at position : " + str(base_coord))
            observations["goal_reach"] = True   
        observations['laser_scan'] = bounded_scan
        observations['cartesian_scan'] = cartesian_scan 
        observations['waypoints'] = waypoints
        observations['tagd'] = tagd
        observations['depth_image'] = depth_image_processed
        observations['final_pos'] = []
        observations['final_pos'].append(self.convert_global_to_robot_coord( float(rospy.get_param('/Test_Goal/x')) , float(rospy.get_param('/Test_Goal/y')) , base_coord[:2] , base_coord[3]))
        observations['curr_pos'] = []
        observations['curr_pos'].append(base_coord[:2])
        observations['truncated'] = self.truncated
        self.environment_management.dynamic_obstacles_movement()
        #return discretized_observations #, info
        return observations #, info
    
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
        

    def _is_done(self, observations):
        
        if min(observations['laser_scan']) <= self.min_range :
            rospy.logerr("Tiago is Too Close to wall==>")
            self._episode_done = True    
        
        #control if robot are block
        """if self.max_stationary_step == 30:
            self.max_stationary_step = 0
            self._episode_done = True
            rospy.loginfo("Robot are block!")"""
              

        return self._episode_done

    def _compute_reward(self, observations, done):

        base_coord = self.amcl_position()
        reward = 0

        #FIRST REWARD FUNCTION
        """#collision reward
        collision_reward = 0.0
        if min(observations['laser_scan']) < self.min_range :
            collision_reward = self.collision_weight * self.collision_reward
            reward += collision_reward


        #proximity reward   
        proximity_reward = 0.0
        if min(observations['laser_scan']) < 0.8: 
            proximity_reward = -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , min(observations['laser_scan'])))
            reward += proximity_reward
        
        #guide reward
        guide_reward  = self.guide_reward(self.path , base_coord[:2])
        guide_reward = -0.5*guide_reward
        reward += guide_reward

        #robot_vel = self.robot_curr_velocity()
        #reward += -0.05 * (robot_vel[1]**2)

        #publish reward value
        self.publish_reward(collision_reward , proximity_reward , guide_reward , 0 , reward)"""
        
        
        
        #SECOND REWARD FUNCTION
        collision_reward = 0.0
        if min(observations['laser_scan']) < self.min_range :
            collision_reward = self.collision_weight * self.collision_reward
            reward += -100

        #proximity reward   
        proximity_reward = 0.0
        if min(observations['laser_scan']) < 0.8: 
            #collision = -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , collision_distance))
            #obstalce_reward = -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , min(observations)))
            proximity_reward = -1.5*abs( 0.8 - min(0.8 , min(observations['laser_scan']))) #for correct model 1.5   
            reward += proximity_reward
        curr_dist = np.linalg.norm(base_coord[:3] - self.goal_pos[:3])
        # 1) Progress toward goal
        distance_error = 2.0*(self.prev_goal_dis - curr_dist)
        #distance_error = -np.linalg.norm(self.goal_pos - base_coord[:3])
        reward += distance_error
        self.prev_goal_dis = curr_dist

        #angular reward
        angular_reward = 0.0

        goal_direction = math.atan2((self.goal_pos[1] - base_coord[1]), (self.goal_pos[0] - base_coord[0]))
        # Raw yaw error
        yaw_error = goal_direction - base_coord[3]

        # 2) Heading alignment: reward forward‐facing motion
        # compute angle between robot’s heading and vector to goal
        goal_vec = self.goal_pos - base_coord[:3]
        heading = base_coord[3]  # radians
        yaw_to_goal = np.arctan2(goal_vec[1], goal_vec[0])
        yaw_error = (((yaw_to_goal - heading) + math.pi) % (2*math.pi)) - math.pi
        if curr_dist < 2.0:
            angular_reward = 3.0 * np.cos(yaw_error)
        else:
            angular_reward = 0.5 * np.cos(yaw_error)

        reward += angular_reward

        
        robot_vel = self.robot_curr_velocity()
        reward += -0.05 * (robot_vel[1]**2)
        if min(observations['laser_scan']) > 0.8:  
            reward += 0.5*(robot_vel[0])
        #reward += -0.01
        #reward += -0.1

        if self.truncated:
            reward += 100
    
        # THIRD REWARD FUNCTION
        """# ---- Distance term ----
        curr_dist = np.linalg.norm(base_coord[:3] - self.goal_pos[:3])
        distance_error = self.prev_goal_dis - curr_dist
        distance_reward = distance_error * 2.0  # same as before

        # ---- Heading term (penalize misalignment) ----
        # Goal angle in radians: 0 = perfect alignment, pi = facing away
        angle_error = abs(math.atan2((self.goal_pos[1] - base_coord[1]), (self.goal_pos[0] - base_coord[0])))

        # Normalize to [0,1] for penalty scaling
        max_angle = np.pi
        normalized_angle_error = angle_error / max_angle  # 0 aligned, 1 worst

        # Reward is negative penalty for being misaligned
        heading_reward = 1.0 - normalized_angle_error * 2.0  # from +1 aligned to -1 very misaligned
        heading_reward = np.clip(heading_reward, -1.0, 1.0)

        # Extra bonus for being very well aligned (< 0.1 rad)
        if angle_error < 0.1:
            heading_reward += 0.5

        # ---- Proximity term ----
        # Encourage keeping a safe clearance (>0.8m), penalize getting too close
        if min(observations['laser_scan']) > 0.8:
            proximity_reward = 0.3
        elif min(observations['laser_scan']) > 0.5:
            proximity_reward = 0.0
        else:
            proximity_reward = -0.3  # penalty for being too close

        # ---- Combine rewards ----
        reward = distance_reward + heading_reward + proximity_reward

        # ---- Collision penalty ----
        if min(observations['laser_scan']) < self.min_range :
            reward += -50

        # Optional: small time penalty to encourage faster goal reach
        reward -= 0.01

        # Update previous distance for next step
        self.prev_goal_dis = curr_dist"""

        return reward
    
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
        new_data = data.ranges[self.n_discard_scan:-self.n_discard_scan]
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
        cartesian_scan = self.generate_rays(bounded_scan.copy() , self.n_discard_scan , self.initial_angle  , self.angle_increment)
        
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
                max_iter: int = 30,
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

        return TAGDs
    
    def image_processing(self , image):
        gamma = 0.10        # floor band half-thickness in metres
        # Camera height above the floor, taken from the translation vector
        h_r = self.translation[2]   # NOTE: if your world frame is Y‑up, use translation[1]

        H, W = image.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        Zc = image                              # Z in camera frame == depth
        Yc = (v - self.K[5]) * Zc / self.K[4]

        # Mask of pixels INSIDE the floor band  |Yc - h_r| < γ
        floor_band = np.abs(Yc - h_r) < gamma

        # Set those depths to 0.0 (not NaN!) so they are ignored later
        F = image.copy()
        F[floor_band] = 0.0

        # --- parameters you can tune --------------------------------
        stride_n   = 8          # pixels per stripe   (8 or 16 in the paper)
        sensor_max = 5.0        # camera max‑range (metres); for normalisation

        # --- 1. prepare the depth map --------------------------------
        H, W = F.shape
        # If width isn't divisible by n, drop the rightmost few columns
        W_trim = (W // stride_n) * stride_n
        F_trim = F[:, :W_trim]

        # Replace floor‑band zeros by +inf *before* taking the min
        F_positive = F_trim.copy()
        F_positive[F_positive == 0] = np.inf     # make them invisible to the min

        # --- 2. reshape so each stripe is contiguous in memory -------
        #     New shape:  (H,  n_stripes,  n)
        n_stripes = W_trim // stride_n
        stripes = F_positive.reshape(H, n_stripes, stride_n)

        # --- 3. take the minimum *non‑zero* depth in each stripe -----
        M = np.nanmin(stripes, axis=(0, 2))      # (n_stripes,) array

        # If a stripe was all floor / invalid, set its entry to 0
        M[np.isinf(M)] = 0.0

        # --- 4. normalise to [0,1] for the neural network ------------
        M_norm = M / sensor_max

        return M_norm
    
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
    
    