import rospy
import numpy
import time
import math
from gym import spaces
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
#####
import rospy
import numpy as np
from openai_ros.robot_envs import tiago_arm_env
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
max_episode_steps_per_episode = 1000 # Can be any Value

register(
        id='TiagoManipulation-v0',
        entry_point='openai_ros.task_envs.tiago.tiago_manipulation:TiagoNav',
        max_episode_steps=max_episode_steps_per_episode,
    )

class TiagoManipulation(tiago_arm_env.TiagoEnv):
    def __init__(self):
        """
        This Task Env is designed for having the Tiago in some kind of maze.
        It will learn how to move around the maze without crashing.
        """
        
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TiagoManipulation, self).__init__()
        
       
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=-1)

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
        
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)
        
        # TODO: Add reset of published filtered laser readings
        laser_scan = self.get_laser_scan()
        discretized_ranges = laser_scan.ranges
        self.publish_filtered_laser_scan(   laser_original_data=laser_scan,
                                         new_filtered_laser_range=discretized_ranges)


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the tiago
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"
        
        # We tell Tiago the linear and angular speed to set to execute
        self.move_base( linear_speed,
                        angular_speed,
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=self.min_range)
        
        rospy.logdebug("END Set Action ==>"+str(action)+", NAME="+str(self.last_action))

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
        
        rospy.logdebug("BEFORE DISCRET _episode_done==>"+str(self._episode_done))
        
        discretized_observations = self.discretize_observation( laser_scan,
                                                                self.new_ranges
                                                                )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("AFTER DISCRET_episode_done==>"+str(self._episode_done))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logdebug("Tiago is Too Close to wall==>"+str(self._episode_done))
        else:
            rospy.logerr("Tiago is Ok ==>")

        return self._episode_done

    def _compute_reward(self, observations, done):

        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


   