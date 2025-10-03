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


# The path is __init__.py of openai_ros, where we import the TiagoMazeEnv directly
max_episode_steps_per_episode = 150 # Can be any Value

register(
        id='TiagoNavigation-v1',
        entry_point='openai_ros.task_envs.tiago.tiago_simply_nav:TiagoNav2',
        max_episode_steps=max_episode_steps_per_episode,
    )

class TiagoNav2(tiago_env.TiagoEnv):
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
        self.reward_range = (-100,100)

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

        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")

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
        min_velocity = [0 , -1.0]
        max_velocity = [1.0 , 1.0]

        #Generate action space
        self.action_space = spaces.Box(np.array(min_velocity), np.array(max_velocity))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.curr_robot_pos = np.array([0,0,0])
        #used for control if the robot are block 
        self.max_stationary_step = 0

       
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TiagoNav2, self).__init__()
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
        self.curr_linear_velocity = 0.0
        self.curr_angular_velocity = 0.0
        self.n_step = 0

        self.model_name = ['box1' , 'box2']
        self.model_coord = []


        #self.spawn_model_from_sdf('box1' , '/home/violo/tiago_public_ws/src/pal_gazebo_worlds/models/box/box.sdf' , x = 1.0)



    def _set_init_pose(self):
        #reset simulation
        print("Sono in init pose!!!!")
        #self.delete_model_from_gazebo(self.model_name)
        #self.model_coord = self.generate_obstacle_coord(self.model_name)
        self.spawn_model_from_sdf(
                'box1',
                '/home/violo/tiago_public_ws/src/pal_gazebo_worlds/models/box/box.sdf',
                x=0.5,
                y=0.0
            )
        #self.generate_init_pos()
        rospy.loginfo(str(self.initx)+ " " + str(self.inity))
        self.gazebo_reset(random.uniform(-3.0, 0.0) , random.uniform(-3.0, 3.0) , 0)
        self.goal_pos = np.array([random.uniform(0.0, 3.0) , random.uniform(-3.0, 3.0) , 0.0])
        rospy.logerr("New position setting complete")
        

        print("Gazebo reset")
        # Short delay for stability
        rospy.sleep(2.0)
        self.path = self.goal_setting(self.goal_pos[0], self.goal_pos[1], 0.0, 0.0, -3.5, 0.0)
        self.prev_goal_dis = self.amcl_position()
        #random_index = random.randint(0, (len(self.paths) - 1))
        #self.path = self.paths[random_index]
        #rospy.loginfo(" Initial position : " + str(self.gazebo_robot_state()))
        #rospy.loginfo("Current Path : " + str(self.path))
        rospy.loginfo(str(self.amcl_position()))
        # self.reset_position()
        self.move_base( 0.0,
                        0.0)
        return True
    
    def is_position_unique(self, x, y, min_distance=0.5):
        return all(((coord[0] - x)**2 + (coord[1] - y)**2)**0.5 >= min_distance for coord in self.model_coord)

    def generate_obstacle_coord(self, model_names):
        coords = []
        for name in model_names:
            while True:
                x = random.uniform(-2.0, 2.0)
                y = random.uniform(-3.0, 3.0)
                if self.is_position_unique(x, y):
                    break
            self.spawn_model_from_sdf(
                name,
                '/home/violo/tiago_public_ws/src/pal_gazebo_worlds/models/box/box.sdf',
                x=x,
                y=y
            )
            coords.append([x, y])
        return coords
    
    def generate_init_pos(self, min_distance_from_obstacles=0.8):
        """
        Generates a valid initial position for the robot, ensuring it's not too close to the goal
        or any obstacles.
        
        :param min_distance_from_obstacles: Minimum distance to keep from any obstacle (in meters)
        """
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            self.initx = random.uniform(-3.0, 0.0)
            self.inity = random.uniform(-3.0, 3.0)

            # Avoid placing too close to the goal or fixed "don't start here" zones
            if ((abs(self.initx - 1.0) <= 0.5 and abs(self.inity - 0.0) <= 0.5) or
                (abs(self.initx - self.goal_pos[0]) <= 0.5 and abs(self.inity - self.goal_pos[1]) <= 0.5)):
                continue

            # Avoid placing too close to any obstacle
            is_far_from_obstacles = all(
                ((obs_x - self.initx)**2 + (obs_y - self.inity)**2)**0.5 >= min_distance_from_obstacles
                for obs_x, obs_y in self.model_coord
            )

            if not is_far_from_obstacles:
                continue

            # Generate path and validate
            self.path = self.goal_setting(self.goal_pos[0], self.goal_pos[1], 0.0, 0.0, self.initx, self.inity)
            if self.path is not None:
                return  # Valid initial position found

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
        self.curr_linear_velocity = 0.0
        self.curr_angular_velocity = 0.0
        self.n_step = 0
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)
        
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
        self.curr_linear_velocity = action[0]
        self.curr_angular_velocity = action[1]
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
        #rospy.loginfo("Laser scan data : " + str(laser_scan))
        base_coord = self.amcl_position()
        info = {}
        #rospy.loginfo("Base coord : " + str(base_coord))

        """if self.dyn_path:
            new_path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , base_coord[0] , base_coord[1])
            if new_path is not None:
                self.path = new_path"""
        
        #generate input of the network
        
        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                    ) 
        
        discretized_observations = self.subsample_laser_scan(discretized_observations)
        
        for i in range(len(discretized_observations)):
            if discretized_observations[i] > 2.0:
                discretized_observations[i] = 2.0
        goal_dist = np.linalg.norm(self.goal_pos - base_coord[:3])

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_pos[0] - base_coord[0]
        skew_y = self.goal_pos[1] - base_coord[1]
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - base_coord[3]  # base_coord[3] is the yaw of the robot
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        
        robot_state = [goal_dist , theta , self.curr_linear_velocity , self.curr_angular_velocity]
        network_input = np.append(discretized_observations, robot_state)
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

        observations['network_input'] = network_input
        observations['laser_scan'] = discretized_observations
        observations['final_pos'] = []
        observations['final_pos'].append(self.convert_global_to_robot_coord( float(rospy.get_param('/Test_Goal/x')) , float(rospy.get_param('/Test_Goal/y')) , base_coord[:2] , base_coord[3]))
        observations['curr_pos'] = []
        observations['curr_pos'].append(base_coord[:2])
        observations['truncated'] = self.truncated

        #return discretized_observations #, info
        return observations #, info
    
    def subsample_laser_scan(self, laser_scan , n_readings = 20):

        
        # 1) throw away the first & last n_discard_scan readings
        valid_ranges = laser_scan

        # how many rays (groups) we want
        n = n_readings
        # size of each group of original scans
        group_size = len(valid_ranges) // n

        cartesian_scan_raw = []    # just the raw min-range values [r0, r1, …]

        for g in range(n):
            # slice out this group
            start_idx = g * group_size
            end_idx   = start_idx + group_size
            group_ranges = valid_ranges[start_idx:end_idx]

            # find the minimum range and its index
            min_idx   = int(np.argmin(group_ranges))
            r_min     = group_ranges[min_idx]
            # store the raw min‐range
            cartesian_scan_raw.append(r_min)

        # sanity‐check
        if len(cartesian_scan_raw) != n:
            rospy.logerr(f"generate_rays: expected {2*n} coords, got {len(cartesian_scan_raw)}")


        return cartesian_scan_raw
        
    def find_upcoming_waypoint(self , path_coords , robot_pos , n_waypoint , yaw):
        waypoints = []
        count_waypoint = 1
        
        closest_idx, _ = self.find_closest_waypoint(path_coords, robot_pos)

        #rospy.loginfo("Closest waypoint index : " + str(closest_idx))
        #rospy.loginfo("Closest waypoint : " + str(path_coords[closest_idx]))
        #rospy.loginfo("Next Closest waypoint : " + str(path_coords[closest_idx+1]))
        if closest_idx != path_coords.shape[0] - 1 :
            curr_waypoint_x , curr_waypoint_y = path_coords[closest_idx+1] 
            # Store result
            waypoints.append(self.convert_global_to_robot_coord(curr_waypoint_x.copy() , curr_waypoint_y.copy() , robot_pos , yaw))

            for x_i, y_i in [tuple(coord) for coord in path_coords[closest_idx+1:]]: 
                
                if math.sqrt((x_i - curr_waypoint_x)**2 + (y_i - curr_waypoint_y)**2) >= self.waypoint_dist:
                    #update waypoint
                    curr_waypoint_x = x_i
                    curr_waypoint_y = y_i
                    #rospy.loginfo("New waypoint : " + str(curr_waypoint_x) + " , " + str(curr_waypoint_y))
                    # Store result
                    waypoints.append(self.convert_global_to_robot_coord(x_i , y_i , robot_pos , yaw))

                    count_waypoint += 1

                    if count_waypoint == self.n_waypoint :
                        break 

        
        # If not enough waypoints were added, fill with the last element of path_coords
        if len(waypoints) < n_waypoint:
            curr_waypoint_x , curr_waypoint_y = path_coords[-1] 
        
            waypoints.extend([self.convert_global_to_robot_coord(curr_waypoint_x.copy() , curr_waypoint_y.copy() , robot_pos , yaw)] * (n_waypoint - len(waypoints)))
        #generate coord of terminal position 
        final_pos = []

        curr_waypoint_x , curr_waypoint_y = path_coords[-1] 
        
        final_pos.append(self.convert_global_to_robot_coord(curr_waypoint_x.copy() , curr_waypoint_y.copy() , robot_pos , yaw))

        return waypoints , final_pos
    
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
        if self.max_stationary_step == 30:
            self.max_stationary_step = 0
            self._episode_done = True
            rospy.loginfo("Robot are block!")
              

        return self._episode_done

    def _compute_reward(self, observations, done):

        base_coord = self.amcl_position()
        reward = 0
        min_laser_range = min(observations['laser_scan'])
        collision_reward = 0.0
        if min_laser_range < self.min_range :
            return -100

        elif self.truncated:
            return 100
        
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return self.curr_linear_velocity / 2 - abs(self.curr_angular_velocity) / 2 - r3(min_laser_range) / 2
    
    def publish_reward(self , collision_reward , proximity_reward , guide_reward , angular_reward , reward):
        # Create and publish multi-array message
        
        reward_msg = Float32MultiArray()
        reward_msg.data = [collision_reward , proximity_reward , guide_reward , angular_reward , reward]
        self.reward_debug_pub.publish(reward_msg)

    def guide_reward(self , path_coords, robot_pos , yaw):
        """
        Calculate the reward value that permit to calculate the distance of the robot respect a waypoint into the global path 
        with a distance ahead respect the robot defined inside yaml file.

        Args :
            path_coords : 
        
        """
        if len(path_coords) == 0:
            rospy.logerr("Empty path!")
            return 0.0
        # Find closest waypoint
        closest_idx, min_distance = self.find_closest_waypoint(path_coords, robot_pos)
        # Calculate the goal position to reach 0.6m ahead
        goal_position = self.calculate_goal_position(path_coords , closest_idx , self.ahead_dist)
        #goal_position = self.convert_global_to_robot_coord(goal_position[0] , goal_position[1] , robot_pos , yaw)
        # Calculate reward (negative distance to guidance point)
        #distance_to_guidance = np.linalg.norm(goal_position - robot_pos)
        goal_direction  = math.atan2((goal_position[0] - robot_pos[0]), (goal_position[1] - robot_pos[1]))

        return -np.linalg.norm(goal_position - robot_pos) , goal_direction


    def find_closest_waypoint(self , path_coords, robot_pos):
        """
        Find the closest waypoint on the path to the robot's position.
        
        """
        path_coords = np.array(path_coords, dtype=np.float32)
        robot_pos = np.array(robot_pos, dtype=np.float32)
        distances = np.linalg.norm(path_coords - robot_pos, axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx, distances[closest_idx]
    
    def calculate_goal_position(self , path_coords , closest_idx , cum_distance):
        """
        Calculate cumulative distances along the path.
        
        """

        ahead_dist = 0.0
        curr_index = closest_idx
        curr_pos = path_coords[closest_idx]

        #control if there is only the final goal position of global path
        if curr_index == path_coords.shape[0] - 1 :
              return curr_pos

        goal_coord = curr_pos

        while ahead_dist < cum_distance :
            curr_index += 1
            goal_coord = path_coords[curr_index]    
            ahead_dist += np.linalg.norm(goal_coord - curr_pos)
            curr_pos = goal_coord
            #control if reach the final position of global path 
            if curr_index == path_coords.shape[0] - 1 :
                break
        return goal_coord
        

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        new_data = data.ranges[self.n_discard_scan:-self.n_discard_scan]
        self._episode_done = False
        
        discretized_ranges = []
        #mod = len(data.ranges)/new_ranges
        
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("n_elements=" + str(len(new_data)/new_ranges))
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
        
        #for make the laser scan value divisible by 90
        discretized_ranges[:2] = [25.0, 25.0]
        discretized_ranges[-2:] = [25.0, 25.0]
        
        return discretized_ranges
    
    def publish_velocity(self , linear_velocity, angular_velocity ):
        # Create a Float32MultiArray message
        vel_msg = Float32MultiArray()
        vel_msg.data = [linear_velocity, angular_velocity]

        # Publish the message
        self.velocity_pub.publish(vel_msg)
    
    