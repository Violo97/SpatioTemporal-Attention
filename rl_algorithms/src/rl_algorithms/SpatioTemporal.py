import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
from typing import List
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray



"""
network_config:
  spatial_key_mlp_layers: [256, 128, 64]        # Embedding network for attention keys
  spatial_value_mlp_layers: [80, 50, 30]        # Feature network for attention values
  spatial_attention_mlp_layers: [60, 50, 1]     # Score network for attention weights
  waypoint_mlp_layers: [64, 40, 30]             # Waypoint embedding MLP (unused in A2)
  action_mlp_layers: [128, 64, 64]              # Output layers for actor/critic

"""


# ================================
# Network Configuration
# ================================
network_config = {
    'spatial_key_mlp_layers': rospy.get_param('/network_config/spatial_key_mlp_layers'),        # Embedding network for attention keys
    'spatial_value_mlp_layers': rospy.get_param('/network_config/spatial_value_mlp_layers'),        # Feature network for attention values
    'spatial_attention_mlp_layers': rospy.get_param('/network_config/spatial_attention_mlp_layers'),     # Score network for attention weights
    'waypoint_mlp_layers': rospy.get_param('/network_config/waypoint_mlp_layers'),             # Waypoint embedding MLP (unused in A2)
    'action_mlp_layers': rospy.get_param('/network_config/action_mlp_layers'),              # Output layers for actor/critic
}


# ================================
# Base Model for Spatial Stream
# ================================
class SpatioTemporalBase(nn.Module):
    """
    Implements the spatial stream of the architecture. Applies location-based attention
    (score+value) to lidar sectors concatenated with waypoints. Omits temporal TAGDs.
    """
    def __init__(self):
        super().__init__()
        self.attention_model = rospy.get_param('/Training/attention_method')
        # Input dimensionality setup
        self.total_lidar_points = rospy.get_param('/Spatial_Attention/n_rays')                # N lidar points
        self.lidar_point_dim = 2                  # 2D Cartesian input per ray
        #self.robot_state_dim = kwargs["robot_state"]                 # Robot state vector (e.g., velocity)
        self.num_waypoints = rospy.get_param('/Training/n_waypoint')                 # Number of path waypoints
        self.waypoint_dim = 2                 # Dimensionality per waypoint
        self.temporal_seq_len = 1                                   # Temporal window (A2 uses 1)
        self.tagd_points = 20  # Number of TAGD points (A2 uses 0)
        self.batch_size = rospy.get_param('/Training/batch_size')  # Batch size for training

        # Sector configuration (Nc sectors)
        self.num_sectors = rospy.get_param('/Spatial_Attention/input_spatial_size')
        self.num_sectors_temporal = 20
        self.points_per_sector = self.total_lidar_points // self.num_sectors
        assert self.points_per_sector * self.num_sectors == self.total_lidar_points

        # Input dimension per sector
        '''spatial_input_dim = (
            self.points_per_sector * self.lidar_point_dim +
            self.num_waypoints * self.waypoint_dim +
            self.robot_state_dim
        )'''
        """spatial_input_dim = (
            self.points_per_sector * self.lidar_point_dim +
            self.num_waypoints * self.waypoint_dim 
        )"""

        spatial_input_dim = (
            self.points_per_sector * self.lidar_point_dim +
            self.num_waypoints * 2 
        )

        temporal_input_dim = (
            4 + 
            self.num_waypoints * 2
        )

        """spatial_input_dim = (
            self.points_per_sector * self.lidar_point_dim
        )"""

        #path network 
        """self.mlp_path = build_mlp(
            10,
            network_config['waypoint_mlp_layers'],
            activate_last_layer=True
        )"""
        #-----------------------------------------
        # SPATIAL ATTENTION NETWORK
        #-----------------------------------------
        # Embedding MLP (key)
        self.mlp_spatial_key = build_mlp(
            spatial_input_dim,
            network_config["spatial_key_mlp_layers"],
            activate_last_layer=True
        )

        # Feature MLP (value)
        self.mlp_spatial_value = build_mlp(
            network_config["spatial_key_mlp_layers"][-1] ,
            network_config["spatial_value_mlp_layers"],
            activate_last_layer=False
        )

        # Score MLP (attention weight)
        self.mlp_spatial_attention = build_mlp(
            network_config["spatial_key_mlp_layers"][-1],
            network_config["spatial_attention_mlp_layers"],
            activate_last_layer=False
        )

        #-----------------------------------------
        # TEMPORAL ATTENTION NETWORK
        #-----------------------------------------

        # Embedding MLP (key)
        self.mlp_temporal_key = build_mlp(
            temporal_input_dim,
            network_config["spatial_key_mlp_layers"],
            activate_last_layer=True
        )

        # Feature MLP (value)
        self.mlp_temporal_value = build_mlp(
            network_config["spatial_key_mlp_layers"][-1] ,
            network_config["spatial_value_mlp_layers"],
            activate_last_layer=False
        )

        # Score MLP (attention weight)
        self.mlp_temporal_attention = build_mlp(
            network_config["spatial_key_mlp_layers"][-1],
            network_config["spatial_attention_mlp_layers"],
            activate_last_layer=False
        )
    def forward(self, lidar_scan , waypoints , tagd):
        spatial_input = input_split( lidar_scan, waypoints , self.num_sectors)  # Reshape lidar data
        temporal_input = input_split( tagd, waypoints , self.num_sectors_temporal)  # Reshape TAGD data
        # Compute key, value, attention
        key_features = self.mlp_spatial_key(spatial_input)          # Embedding per sector
        key_temporal_features = self.mlp_temporal_key(temporal_input)  # Embedding per TAGD
        #key_features = torch.cat([key_features, waypoint], dim=2)  # Concatenate waypoints
        value_features = self.mlp_spatial_value(key_features)       # Sector features
        attention_scores = self.mlp_spatial_attention(key_features) # Attention weights

        # Reshape for attention-weighted sum
        value_features = value_features.view(spatial_input.shape[0], self.num_sectors, -1)
        attention_scores = attention_scores.view(spatial_input.shape[0] , self.num_sectors, 1)

        value_temporal_features = self.mlp_temporal_value(key_temporal_features)       # Sector features
        attention_temporal_scores = self.mlp_temporal_attention(key_temporal_features) # Attention

        # Reshape for attention-weighted sum
        value_temporal_features = value_temporal_features.view(temporal_input.shape[0], self.num_sectors_temporal, -1)
        attention_temporal_scores = attention_temporal_scores.view(temporal_input.shape[0] , self.num_sectors_temporal, 1)
        
        #rospy.loginfo(str(attention_weights))
        """if self.debug and eval:
            attention_weights = softmax(attention_scores, dim=1)  # Normalize across sectors
            with open(self.rospack.get_path('tiago_navigation') + "/data/" + str(self.algorithm_name) + "_attention_score.txt", 'a') as file:  
                for i in range(attention_weights.shape[1]):  # Batch dimension
                    file.write(str(attention_weights[0, i, 0].item()))
                    if i < attention_weights.shape[1] - 1:
                        file.write(",")
                file.write("\n")
        """
        # Weighted sum over sectors using softmax attention
        weighted_features, attention_weights = compute_spatial_weighted_feature(attention_scores, value_features)
        weighted_temporal_features, attention_temporal_weights = compute_spatial_weighted_feature(attention_temporal_scores, value_temporal_features)
        if eval is False:
            self.publish_scores(attention_weights)
        #rospy.loginfo("Weighted features shape: %s", str(weighted_features))
        if self.attention_model == "Spatial" or self.attention_model == "Depth":
            features = weighted_features
        elif self.attention_model == "Temporal":
            features = weighted_temporal_features
        else:
            features = torch.cat([weighted_features, weighted_temporal_features], dim=1)

        return features
        


# ================================
# Actor Network
# ================================
class SpatioTemporalActor(SpatioTemporalBase):
    """
    Actor for A2 ablation. Processes a single lidar scan with spatial attention,
    fuses features, and outputs actions.
    """
    def __init__(self, action_dim ):
        super().__init__()

        self.att_score_pub = rospy.Publisher('/spatial_attention_scores', Float32MultiArray, queue_size=1)
        # Action output size
        self.num_actions = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output dim: mean + std for SAC, otherwise just action vector
        output_dim = self.num_actions

        # Final MLP projecting aggregated features to action logits
        self.mlp_action_output = build_mlp(
            2*network_config["spatial_value_mlp_layers"][-1] * self.temporal_seq_len,
            network_config["action_mlp_layers"] + [output_dim],
            activate_last_layer=True,
            last_layer_activate_func=nn.Tanh()
        )

        #linear and angular velocity raange
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')

        self.debug = rospy.get_param('/Training/debug')
        self.algorithm_name = rospy.get_param("/Training/algorithm")

        self.rospack = rospkg.RosPack()


    def forward(self, lidar_scan , waypoints , tagd ,  eval=False):
        #rospy.loginfo("Lidar : " + str(lidar_scan) + " Waypoints : " + str(waypoints))
        spatial_input = input_split( lidar_scan, waypoints , self.num_sectors)  # Reshape lidar data
        temporal_input = input_split( tagd, waypoints , self.num_sectors_temporal)  # Reshape TAGD data
        #rospy.loginfo("Spatial input shape: %s", str(spatial_input))
        # Compute key, value, attention
        key_features = self.mlp_spatial_key(spatial_input)          # Embedding per sector
        key_temporal_features = self.mlp_temporal_key(temporal_input)  # Embedding per TAGD
        #key_features = torch.cat([key_features, waypoint], dim=2)  # Concatenate waypoints
        value_features = self.mlp_spatial_value(key_features)       # Sector features
        attention_scores = self.mlp_spatial_attention(key_features) # Attention weights

        # Reshape for attention-weighted sum
        value_features = value_features.view(spatial_input.shape[0], self.num_sectors, -1)
        attention_scores = attention_scores.view(spatial_input.shape[0] , self.num_sectors, 1)

        value_temporal_features = self.mlp_temporal_value(key_temporal_features)       # Sector features
        attention_temporal_scores = self.mlp_temporal_attention(key_temporal_features) # Attention

        # Reshape for attention-weighted sum
        value_temporal_features = value_temporal_features.view(temporal_input.shape[0], self.num_sectors_temporal, -1)
        attention_temporal_scores = attention_temporal_scores.view(temporal_input.shape[0] , self.num_sectors_temporal, 1)
        
        #rospy.loginfo(str(attention_weights))
        """if self.debug and eval:
            attention_weights = softmax(attention_scores, dim=1)  # Normalize across sectors
            with open(self.rospack.get_path('tiago_navigation') + "/data/" + str(self.algorithm_name) + "_attention_score.txt", 'a') as file:  
                for i in range(attention_weights.shape[1]):  # Batch dimension
                    file.write(str(attention_weights[0, i, 0].item()))
                    if i < attention_weights.shape[1] - 1:
                        file.write(",")
                file.write("\n")
        """
        # Weighted sum over sectors using softmax attention
        weighted_features, attention_weights = compute_spatial_weighted_feature(attention_scores, value_features)
        weighted_temporal_features, attention_temporal_weights = compute_spatial_weighted_feature(attention_temporal_scores, value_temporal_features)
        if eval is False:
            self.publish_scores(attention_weights)
        #rospy.loginfo("Weighted features shape: %s", str(weighted_features))
        features = torch.cat([weighted_features, weighted_temporal_features], dim=1)
        #features = weighted_features.view(spatial_input.shape[0], -1)
        # Ensure waypoints has batch dimension
        #if len(waypoints.shape) == 1:
        #    waypoints = waypoints.unsqueeze(0)  # Shape becomes (1, input_dim) 
        #path_features = self.mlp_path(waypoints)
        # Final action output
        #output = self.mlp_action_output(torch.cat([features, path_features], dim=1))
        output = self.mlp_action_output(features)
        #if len(lidar_scan.shape) == 1:
            #lidar_scan = lidar_scan.unsqueeze(0)  # Shape becomes (1, input_dim) 
        #output = self.mlp_action_output(torch.cat([lidar_scan, waypoints], dim=1))  # Concatenate lidar scan and waypoints
        #rospy.loginfo("Actor output shape: %s", str(torch.cat([lidar_scan, waypoints], dim=1)))
        # Bound the outputs
        #rospy.loginfo("Actor output before scaling: %s", str(output))
        output1 = ((output[:,0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity).unsqueeze(1)
        output2 = ((output[:,1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity).unsqueeze(1)
        #rospy.loginfo("Actor output after scaling: %s", str(torch.cat([output1, output2], dim=1)))
          
        #rospy.loginfo(str(output))
        return torch.cat([output1, output2], dim=1)
        #return output

    def publish_scores(self, attention_scores):
        # Create a Float32MultiArray message
        score_msg = Float32MultiArray()
        score_msg.data = attention_scores.detach().cpu().squeeze().numpy()

        # Publish the message
        self.att_score_pub.publish(score_msg)

# ================================
# Critic Network
# ================================
class SpatioTemporalCritic(SpatioTemporalBase):
    """
    Critic for A2 ablation. Computes spatial attention over lidar and evaluates Q-value
    of observation-action pair.
    """
    def __init__(self,  action_space ):
        super().__init__()

        # Final Q-value regressor MLP
        self.num_actions = action_space
        self.q_value_mlp = build_mlp(
            #network_config["spatial_value_mlp_layers"][-1] * self.temporal_seq_len + self.num_actions + 30,
            2*network_config["spatial_value_mlp_layers"][-1] * self.temporal_seq_len + self.num_actions,
            #190 + self.num_actions,
            network_config["action_mlp_layers"] + [1],
            activate_last_layer=False
        )
        

    def forward(self, lidar_scan , waypoints , tagd , action):

        #rospy.loginfo("Lidar : " + str(lidar_scan) + " Waypoints : " + str(waypoints))
        spatial_input = input_split( lidar_scan, waypoints , self.num_sectors)  # Reshape lidar data
        #rospy.loginfo("Spatial input shape: %s", str(spatial_input))# Apply attention block
        key_features = self.mlp_spatial_key(spatial_input)
        #key_features = torch.cat([key_features, waypoint], dim=2)  # Concatenate waypoints
        value_features = self.mlp_spatial_value(key_features)
        attention_scores = self.mlp_spatial_attention(key_features)

        value_features = value_features.view(spatial_input.shape[0], self.num_sectors, -1)
        attention_scores = attention_scores.view(spatial_input.shape[0], self.num_sectors, 1)

        # Attention-based sector aggregation
        weighted_spatial_features, _ = compute_spatial_weighted_feature(attention_scores, value_features)
        
        # -------- Temporal branch (TAGD) --------
        temporal_input = input_split(tagd, waypoints, self.num_sectors_temporal)  # [B, num_sectors_temporal, ...]
        key_temporal_features = self.mlp_temporal_key(temporal_input)
        value_temporal_features = self.mlp_temporal_value(key_temporal_features)
        attention_temporal_scores = self.mlp_temporal_attention(key_temporal_features)

        value_temporal_features = value_temporal_features.view(
            temporal_input.shape[0], self.num_sectors_temporal, -1
        )
        attention_temporal_scores = attention_temporal_scores.view(
            temporal_input.shape[0], self.num_sectors_temporal, 1
        )

        weighted_temporal_features, _ = compute_spatial_weighted_feature(
            attention_temporal_scores, value_temporal_features
        )


        #concatenate spatial + temporal along feature dim
        fused_features = torch.cat([weighted_spatial_features, weighted_temporal_features], dim=1)
        # Ensure action is [B, num_actions]
        if action.dim() == 1:
            action = action.unsqueeze(0)  # handle single-sample case
        critic_input = torch.cat([fused_features, action], dim=1)

        return self.q_value_mlp(critic_input)
    
    

# ================================
# Utilities: MLP and Attention
# ================================
def build_mlp(
    input_dim: int,
    mlp_dims: List[int],
    activate_last_layer: bool = False,
    activate_func=nn.ReLU(),
    last_layer_activate_func=None
) -> nn.Sequential:
    """
    Constructs a feedforward MLP module using a list of dimensions.
    Applies Xavier uniform initialization to all hidden layers, and
    small uniform initialization to final layer if using Tanh activation.

    Args:
        input_dim (int): Dimensionality of the input vector.
        mlp_dims (List[int]): Sizes of subsequent MLP layers.
        activate_last_layer (bool): Whether to activate the last layer.
        activate_func: Activation between hidden layers (default: ReLU).
        last_layer_activate_func: Override for the last layer activation.

    Returns:
        nn.Sequential: Assembled and initialized MLP block.
    """
    layers = []
    layer_dims = [input_dim] + mlp_dims

    for i in range(len(layer_dims) - 1):
        in_dim = layer_dims[i]
        out_dim = layer_dims[i + 1]
        linear = nn.Linear(in_dim, out_dim)

        is_last = (i == len(layer_dims) - 2)
        # Initialize weights
        if is_last and activate_last_layer and isinstance(last_layer_activate_func, nn.Tanh):
            # Small init range for final Tanh layer
            nn.init.uniform_(linear.weight, -3e-3, 3e-3)
        else:
            # Xavier uniform for all other layers
            nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

        layers.append(linear)
        # Append activation
        if not is_last:
            layers.append(activate_func)
        elif activate_last_layer:
            final_act = last_layer_activate_func if last_layer_activate_func else activate_func
            layers.append(final_act)

    return nn.Sequential(*layers)

    """layers = []
    layer_dims = [input_dim] + mlp_dims
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        torch.nn.init.xavier_uniform_(layers[-1].weight)  # Ensures stable initialization

        is_last = (i == len(layer_dims) - 2)
        if not is_last:
            layers.append(activate_func)
        elif activate_last_layer:
            final_act = last_layer_activate_func if last_layer_activate_func else activate_func
            layers.append(final_act)

    return nn.Sequential(*layers)"""


def compute_spatial_weighted_feature(attention_scores: torch.Tensor, features: torch.Tensor):
    """
    Computes the attention-weighted sum over sectors for a batch.

    Args:
        attention_scores (Tensor): Raw attention logits [B, Nc, 1].
        features (Tensor): Corresponding sector features [B, Nc, D].

    Returns:
        weighted_feature (Tensor): Aggregated feature [B, D].
        attention_weights (Tensor): Normalized attention weights [B, Nc, 1].
    """
    attention_weights = softmax(attention_scores, dim=1)  # Normalize across sectors
    weighted_feature = torch.sum(attention_weights * features, dim=1)  # Weighted sum
    return weighted_feature, attention_weights

def input_split(lidar_data , waypoints , num_sectors):
        # Ensure input has batch dimension
        if len(lidar_data.shape) == 1:
            lidar_data = lidar_data.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        # Ensure waypoints has batch dimension
        if len(waypoints.shape) == 1:
            waypoints = waypoints.unsqueeze(0)  # Shape becomes (1, input_dim) 
        #rospy.loginfo("Lidar data shape: %s", str(lidar_data.shape))
        #rospy.loginfo(str(lidar_data))
        lidar_data = lidar_data.reshape(lidar_data.shape[0], num_sectors, -1)
        waypoints = waypoints.view(lidar_data.shape[0], 1, -1).repeat(1, num_sectors, 1)

        # Concatenate sector data and waypoint input
        spatial_input = torch.cat([lidar_data, waypoints], dim=2)
        
        #return lidar_data
        return spatial_input