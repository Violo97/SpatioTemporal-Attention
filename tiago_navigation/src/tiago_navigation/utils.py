import torch
import torch.nn as nn
import copy
import rospy
from std_msgs.msg import String
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def clone_model(original_model):
    # Create a deep copy of the model
    clone_model = copy.deepcopy(original_model)
    
    # Ensure the clone has the same device as the original
    device = next(original_model.parameters()).device
    clone_model.to(device)

    # Verify that the clone has the same structure and weights
    for (name1, param1), (name2, param2) in zip(original_model.named_parameters(), clone_model.named_parameters()):
      if not torch.all(param1.eq(param2)):
        rospy.logerr(f"Parameters are different !")
    
    
    return clone_model


def target_weight_update( target_network , network , update_coeff):
   
   for target_weight , weight in zip(target_network.parameters(), network.parameters()):
            # Update the weights of network B
            #target_weight.data = update_coeff * weight.data + (1 - update_coeff) * target_weight.data
            target_weight.data.copy_(update_coeff * weight.data + (1 - update_coeff) * target_weight.data)


def remove_file(folder_path , filename):
     # Create the full path to the file
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file exists
    if os.path.isfile(file_path):
        try:
            # Delete the file
            os.remove(file_path)
            rospy.loginfo(f"File '{filename}' has been deleted successfully.")
        except Exception as e:
            rospy.loginfo(f"An error occurred while deleting the file: {e}")
    else:
        rospy.loginfo(f"File '{filename}' not found in the folder '{folder_path}'.")


# need to update
def polar_laser_scan(laser_scan , n_discard_scan , initial_angle , angle_increment):
    start_angle = initial_angle + (n_discard_scan * angle_increment)
    polar_scan = []
    for i in range(len(laser_scan)):
        polar_scan.append(laser_scan[i])
        polar_scan.append(start_angle + (i * (2*angle_increment)))
    return polar_scan

def cartesian_laser_scan(laser_scan , n_discard_scan , initial_angle , angle_increment):
    start_angle = initial_angle + (n_discard_scan * angle_increment)
    cartesian_scan = []
    x_arr = []
    y_arr = []
    for i in range(len(laser_scan)):
        angle = start_angle + (i * (angle_increment))
        x = laser_scan[i] * math.cos(angle)
        x_arr.append(x)
        y = laser_scan[i] * math.sin(angle)
        y_arr.append(y)
        cartesian_scan.append(laser_scan[i] * math.cos(angle))
        cartesian_scan.append(laser_scan[i] * math.sin(angle))
    point_cloud = np.column_stack((np.array(x_arr) , np.array(y_arr)))
    return cartesian_scan , point_cloud

def gen_bounded_scan(laser_scan , max_range = 3.5):        
    for i in range(0 , len(laser_scan)):
        if laser_scan[i] >= max_range:
            laser_scan[i] = max_range
        #if rospy.get_param('/Training/input_normalization'): 
        #laser_scan[i] = laser_scan[i]/3.5    
    return laser_scan    


"""def generate_rays(laser_scan , n_discard_scan , initial_angle  , angle_increment):
    start_angle = initial_angle + (n_discard_scan * angle_increment)
    cartesian_scan = []
    cartesian_scan_raw = []
    x_arr = []
    y_arr = []
    n = rospy.get_param("/Spatial_Attention/n_rays")
    ray_group_dim = round(len(laser_scan)/n)
    min_rays = 26.0
    min_rays_angle = 0
    for i in range(len(laser_scan)):
        angle = start_angle + (i * angle_increment)
        if laser_scan[i] < min_rays:
            min_rays = laser_scan[i]
            min_rays_angle = angle
        if (i+1) % ray_group_dim == 0:
            if 2*n == len(cartesian_scan):
                break
            cartesian_scan_raw.append(min_rays)
            #adjust this part for return the cortesian coordinates correct for the thesis
            x = min_rays * math.cos(min_rays_angle)
            x_arr.append(x)
            y = min_rays * math.sin(min_rays_angle)
            y_arr.append(y) 
            cartesian_scan.append(laser_scan[i] * math.cos(angle))
            cartesian_scan.append(laser_scan[i] * math.sin(angle))
            min_rays = 26.0
    if 2*n > len(cartesian_scan) or 2*n < len(cartesian_scan):
        print("Rays array is incorrect ! , n elements : " + str(len(cartesian_scan)))
    if len(cartesian_scan) != 2*n:
        rospy.logerr("error in dimension of input ! " + str(len(input))+ " laser scan dim : " + str(len(laser_scan)))
    pointcloud = np.column_stack((np.array(x_arr) , np.array(y_arr)))
    return cartesian_scan , pointcloud , cartesian_scan_raw """

def generate_rays(laser_scan, n_discard_scan, initial_angle, angle_increment):
    # 1) throw away the first & last n_discard_scan readings
    valid_ranges = laser_scan
    # angles start at initial_angle + n_discard_scan*angle_increment
    start_angle = initial_angle + n_discard_scan * angle_increment

    # how many rays (groups) we want
    n = int(rospy.get_param("/Spatial_Attention/n_rays"))
    # size of each group of original scans
    group_size = len(valid_ranges) // n

    cartesian_scan   = []      # flat [x0, y0, x1, y1, …]
    cartesian_scan_raw = []    # just the raw min-range values [r0, r1, …]
    x_arr            = []      # for building the Nx2 point cloud
    y_arr            = []

    for g in range(n):
        # slice out this group
        start_idx = g * group_size
        end_idx   = start_idx + group_size
        group_ranges = valid_ranges[start_idx:end_idx]

        # compute the angles for each element in the group
        angles = [
            start_angle + (n_discard_scan + start_idx + i) * angle_increment
            for i in range(len(group_ranges))
        ]

        # find the minimum range and its index
        min_idx   = int(np.argmin(group_ranges))
        r_min     = group_ranges[min_idx]
        theta_min = angles[min_idx]

        # store the raw min‐range
        cartesian_scan_raw.append(r_min)

        # convert it to Cartesian
        x = r_min * math.cos(theta_min)
        y = r_min * math.sin(theta_min)
        x_arr.append(x)
        y_arr.append(y)

        # push into the flat scan vector
        cartesian_scan.append(x) # attention !!!!!
        cartesian_scan.append(y) # attention !!!!!

    # sanity‐check
    if len(cartesian_scan) != 2 * n:
        rospy.logerr(f"generate_rays: expected {2*n} coords, got {len(cartesian_scan)}")

    # build your point‐cloud
    pointcloud = np.column_stack((np.array(x_arr), np.array(y_arr)))

    return cartesian_scan, pointcloud, cartesian_scan_raw



def laser_plot(source, target=None, tagd_list=None , rays = False):
    rospy.loginfo(str(source))
    # Plotting the source and target scans
    plt.figure(figsize=(8, 6))

    # Plot source (prev_scan) points in blue
    if rays:
        for i in range( 0,len(source) , 2):
            plt.scatter(source[i], source[i+1], label='Source (Prev Scan)', color='blue', s=5)
    else:    
        plt.scatter(source[:, 0], source[:, 1], label='Source (Prev Scan)', color='blue', s=5)
    if target is not None:
        # Plot target (curr_scan) points in red
        plt.scatter(target[:, 0], target[:, 1], label='Target (Curr Scan)', color='red', s=5)

    if tagd_list is not None:
        # Plot centroids of filtered_prev_scan in green
        for i, data in enumerate(tagd_list):
            if i % 2 == 0:
                plt.scatter(data[0], data[1], c='purple', label='Previous Scan Centroids', s=100)
            else:
                plt.scatter(data[0], data[1], c='orange', label='Current Scan Centroids', s=100)

    # Labeling and formatting
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cartesian Point Cloud with Centroids')
    #plt.legend()
    plt.grid(True)
    plt.show()  

def plot(point , waypoints=None):
    # Create x-values as indices
    # Convert to x and y lists
    x_coords = point[::2]  # Every even-indexed element
    y_coords = point[1::2]  # Every odd-indexed element
    x_way = waypoints[::2] 
    y_way = waypoints[1::2]

    # Plot the coordinates
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o')  # Connect points with lines and show points
    plt.plot(x_way, y_way, marker='x', color='red')  # Plot waypoints in red
    plt.title('Cartesian Coordinates Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')  # Equal scaling on both axes
    plt.show()





