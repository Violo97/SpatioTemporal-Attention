import rospy
from geometry_msgs.msg import Point, Quaternion , Pose
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState , GetModelState , GetModelStateRequest
from gazebo_msgs.msg import ModelState
import random
import numpy as np
import math


import rospy
from geometry_msgs.msg import Point, Quaternion , Pose
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState , GetModelState , GetModelStateRequest
from gazebo_msgs.msg import ModelState
import random
import numpy as np
import math


import rospy
from geometry_msgs.msg import Point, Quaternion , Pose
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState , GetModelState , GetModelStateRequest
from gazebo_msgs.msg import ModelState
import random
import numpy as np
import math
from tf.transformations import quaternion_from_euler


class Environment_Management():

    def __init__(self):
        super().__init__()
        self.phase = '/' + rospy.get_param("/Curriculum_param/curriculum_phase")
        self.environment = rospy.get_param("/Curriculum_param/environment")
        self.environment_path = '/' + self.environment
        self.static_obstacles = ['sobs_1' , 'sobs_2']#rospy.get_param(self.phase + "/static_obstacles")
        self.dynamic_obstacles = ['dobs_1' , 'dobs_2' , 'dobs_3' , 'dobs_4' ]#rospy.get_param(self.phase + "/dynamic_obstacles")
        #[[x_first_zone],[x_second_zone]]
        self.zone_coord = rospy.get_param(self.environment_path + "/env_coord")
        rospy.logdebug("Zone coord : " + str(self.zone_coord) + " phase : " + str(self.phase) + " environment : " + str(self.environment) + " static obs : " + str(self.static_obstacles) + " dynamic obs : " + str(self.dynamic_obstacles))
        self.n_zones = 2  # Number of zones in the corridor
        #define start and goal position areas , give a margin to y coordinates in this way 
        #avoid conpenetration of robot with wall
        self.start_zone = [[self.zone_coord[0][0], self.zone_coord[0][0] + 2.5],
                            [self.zone_coord[1][0] + 0.6, self.zone_coord[1][1] - 0.6]]
        self.goal_zone = [[self.zone_coord[0][1] - 2.5, self.zone_coord[0][1]],
                            [self.zone_coord[1][0] + 0.6, self.zone_coord[1][1] - 0.6]]
        self.initial_pos_x = rospy.get_param(self.phase + "/initial_pos_x")
        self.initial_pos_y = rospy.get_param(self.phase + "/initial_pos_y")
        self.circular_goal = rospy.get_param(self.phase + "/circular")
        if self.circular_goal:
            self.radius = rospy.get_param(self.phase + "/radius")
            self.angles = rospy.get_param(self.phase + "/angles")
        else:
            self.goal_x = rospy.get_param(self.phase + "/goal_pos_x")
            self.goal_y = rospy.get_param(self.phase + "/goal_pos_y")

        self.count_obs = rospy.get_param(self.phase + "/n_obs")
        
        self.start_coord = []
        self.goal_coord = []
        
        rospy.logdebug("Start zone : " + str(self.start_zone) + " Goal zone : " + str(self.goal_zone))
        #define the zones of corridor in remaining part of environment
        self.zones = self.generate_zones(self.start_zone[0][1] , 
                                         self.goal_zone[0][0] ,
                                         self.n_zones)
        rospy.logdebug("Zones : " + str(self.zones))
        
        self.dynamic_path_coords = []
        self.room_zone = []
        self.random_start = rospy.get_param(self.phase + "/dyn_start")
        self.random_goal = rospy.get_param(self.phase + "/dyn_goal")
        if len(self.dynamic_obstacles) > 0:
            self.speed = rospy.get_param(self.phase + "/dyn_obs_speed")#0.3  # Speed of dynamic obstacles
        self.update_obs_rate = 5
        self.position = []
        self.dir = []
        self.forward = []
        self.static_obs_pos = []
        self.dyn_obs_pos = []
        #gazebo control
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        #dynamic obstacles topic
        self.get_dynamic_obstacles_pos = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_dynamic_obstacles_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.translation_obs_x = rospy.get_param(self.phase + "/translation_obs_x")
        if self.phase != '/Phase_0':
            self.static_obs_type = rospy.get_param(self.phase + "/static_obs_type")
            self.static_obs_angle = rospy.get_param(self.phase + "/static_obs_angle")
    def remove_obstacles(self):

        self.delete_model_from_gazebo(self.static_obstacles)
        
        self.delete_model_from_gazebo(self.dynamic_obstacles)

        #static and dynamic obstacles lists reset
        self.static_obstacles = []
        self.dynamic_obstacles = []

        # reset state
        self.position = []
        self.dynamic_path_coords = []
        self.dir = []
        self.forward = []
        self.static_obs_pos = []
        self.dyn_obs_pos = []
    
    def generate_coords(self):
        if self.random_start:
            if len(self.initial_pos_x) > 1:
                x_start = random.uniform(self.initial_pos_x[0] , self.initial_pos_x[1]) 
            else:
                x_start = self.initial_pos_x[0]
            if len(self.initial_pos_y) > 1:
                y_start = random.uniform(self.initial_pos_y[0] , self.initial_pos_y[1])
            else:
                y_start = self.initial_pos_y[0]
        self.start_coord = [x_start, y_start]
        if self.random_goal:
            if self.circular_goal:
                angle = random.uniform(self.angles[0], self.angles[1])
                x_goal = self.radius * math.cos(angle)
                y_goal = self.radius * math.sin(angle)
            else:
                if len(self.goal_x) > 1:
                    x_goal = random.uniform(self.goal_x[0] , self.goal_x[1])                
                else:
                    x_goal = self.goal_x[0]
                if len(self.goal_y) > 1:
                    y_goal = random.uniform(self.goal_y[0] , self.goal_y[1])
                else:
                    y_goal = self.goal_y[0]
        self.goal_coord = [x_goal, y_goal]

        return x_start, y_start, x_goal, y_goal
    
    def generate_zones(self , start_limit_zone , goal_limit_zone , n_zones):
        zones = []
        x_dim_zone = (goal_limit_zone - start_limit_zone) / n_zones
        for i in range(n_zones):
            x_start = start_limit_zone + i * x_dim_zone
            x_end = x_start + x_dim_zone
            zones.append([x_start, x_end])
        return zones
    
    def obstacles_generation(self , path):
        if self.environment == 'Corridor':
            self.spawn_corridor_obs(path)
        else:
            self.spawn_room_obs()

    def spawn_corridor_obs(self , path ):
        n_obs = 0

        #generation of obstacles configuration
        if self.count_obs >= 2 :
            n_static = 1 #random.randint(1 , 2)
            for i in range(n_static):
                self.static_obstacles.append('sobs_' + str(i+1))
            for i in range(self.count_obs - n_static):
                self.dynamic_obstacles.append('dobs_' + str(i+1))
            static_obs = '/home/violo/tesi_project/src/model/obstacles/model.sdf'
            speeds = [0.5 , 0.6, 0.7, 0.8]
            self.speed = random.choice(speeds)
            
        elif self.count_obs == 1:
            self.static_obstacles.append('sobs_1')
            static_obs = '/home/violo/tiago_public_ws/src/pal_gazebo_worlds/models/lab_table/lab_table.sdf'


        # --- STATIC OBSTACLES ---
        for i, obs in enumerate(self.static_obstacles):
            while True:
                idx = random.randint(0, path.shape[0] - 1)
                distance_from_start = np.linalg.norm(path[idx] - np.array(self.start_coord))
                distance_from_goal = np.linalg.norm(path[idx] - np.array(self.goal_coord))
                if distance_from_start >= 2.0 and distance_from_goal >= 2.0:
                    x = path[idx, 0]
                    y = path[idx,1]
                    
                    if self.translation_obs_x:
                        x_noise = random.uniform(-1.0 , 1.0)
                        x = path[idx, 0] + x_noise
                        if x <= self.zone_coord[0][0] :
                            x = self.zone_coord[0][0] + 0.2
                        elif x >= self.zone_coord[0][1]:
                            x = self.zone_coord[0][1] - 0.2
                    else:
                        y_noise = random.uniform(-0.6 , 0.6)
                        y = path[idx, 1] + y_noise
                        if y <= self.zone_coord[1][0] :
                            y = self.zone_coord[1][0] + 0.2
                        elif y >= self.zone_coord[1][1]:
                            y = self.zone_coord[1][1] - 0.2
                    break
            rospy.logdebug("Static obstacle " + str(obs) + " at position (" + str(x) + " , " + str(y) + ")")
            self.static_obs_pos.append([x, y])
            self.spawn_model_from_sdf(
                obs,
                self.static_obs_type,
                #'/home/violo/tesi_project/src/model/obstacles/model.sdf',
                #'/home/violo/tiago_public_ws/src/pal_gazebo_worlds/models/kitchen_table/kitchen_table.sdf', #table
                x=x,
                y=y,
                yaw=self.static_obs_angle
                )
            n_obs += 1


        # --- DYNAMIC OBSTACLES ---
        for i, obs in enumerate(self.dynamic_obstacles):
            A, B = self.dyn_obs_coord_generation(path)
            dx = B[0] - A[0]
            dy = B[1] - A[1]
            dist = math.sqrt(dx*dx + dy*dy)
            self.dir.append([dx/dist, dy/dist])
            start_from_A = random.choice([True, False]) # randomizza partenza
            self.forward.append(start_from_A) # True: va verso B; False: verso A
            start_pos = [A[0], A[1]] if start_from_A else [B[0], B[1]]


            rospy.logdebug(f"Dyn obstacle {obs}: A=({A[0]:.2f},{A[1]:.2f}) -> B=({B[0]:.2f},{B[1]:.2f})")


            self.dynamic_path_coords.append([[A[0], A[1]] , [B[0], B[1]]])
            self.position.append(start_pos)


            self.spawn_model_from_sdf(
                obs,
                '/home/violo/tesi_project/src/model/obstacles/model.sdf',
                x=start_pos[0],
                y=start_pos[1]
                )


            n_obs += 1

    def spawn_room_obs(self):
        return

    def _clamp_to_corridor(self, x, y, margin=0.2):
        y_min = self.zone_coord[1][0] + margin
        y_max = self.zone_coord[1][1] - margin
        return x, min(max(y, y_min), y_max)

    def dyn_obs_coord_generation(self, path,
                                 min_length=4.0,
                                 min_static_clear=1.5,
                                 min_dyn_clear=2.0, #1.5 if 4 dyn obs
                                 min_from_start_goal=2.0, #1.5 if 4 dyn obs
                                 y_margin=0.2,
                                 edge_guard=5,
                                 max_tries=150):
        """
        Versione minimale: genera A e B verticali (stesso x, y diversi) a partire da un punto del path (x,y).
        - lunghezza verticale >= min_length
        - distanza dal centro rispetto agli ostacoli statici >= min_static_clear
        - distanza dal centro rispetto agli altri dinamici >= min_dyn_clear
        - lontano da start/goal di almeno min_from_start_goal
        """
        n_pts = path.shape[0]
        if n_pts < 2:
            rospy.logwarn("Path troppo corto; impossibile generare ostacoli dinamici.")
            return (0.0, 0.0), (0.0, 0.0)

        # corridoio utile in y con margine
        y_lo = self.zone_coord[1][0] + y_margin
        y_hi = self.zone_coord[1][1] - y_margin
        avail_span = max(0.0, y_hi - y_lo)
        if avail_span < min_length:
            rospy.logwarn(f"Corridoio utile ({avail_span:.2f} m) più corto di min_length ({min_length:.2f} m). Adeguo min_length.")
            min_length = avail_span
            if min_length <= 0.0:
                # fallback senza movimento
                P = path[n_pts // 2]
                x0 = float(P[0])
                yc = float(np.clip(P[1], y_lo, y_hi))
                A = (x0, yc)
                B = (x0, yc)
                self.dyn_obs_pos.append([A[0], A[1]])
                return A, B

        tries = 0
        while tries < max_tries:
            tries += 1
            # 1) scegli un indice interno
            idx = random.randint(edge_guard, max(edge_guard, n_pts - 1 - edge_guard))
            P = path[idx]
            x0, yc = float(P[0]), float(P[1])

            # clamp del centro dentro al corridoio lasciando spazio per metà segmento ai lati
            half = min_length / 2.0
            yc = float(np.clip(yc, y_lo + half, y_hi - half))

            # 2) distanze da start/goal
            if (np.linalg.norm(P - np.array(self.start_coord)) < min_from_start_goal or
                np.linalg.norm(P - np.array(self.goal_coord)) < min_from_start_goal):
                continue

            # 3) centro lontano dagli statici
            ok = True
            for spos in self.static_obs_pos:
                if self.translation_obs_x:
                    if math.hypot(yc - spos[1]) < min_static_clear:
                        ok = False
                        break
                else:
                    if math.hypot(x0 - spos[0]) < min_static_clear:
                        ok = False
                        break
            if not ok:
                continue

            # 4) centro lontano dagli altri dinamici
            for dpos in self.dyn_obs_pos:
                if self.translation_obs_x:
                    if math.hypot(yc - dpos[1]) < min_dyn_clear:
                        ok = False
                        break
                else:
                    if math.hypot(x0 - dpos[0]) < min_dyn_clear:
                        ok = False
                        break
            if not ok:
                continue

            # 5) costruisci A,B verticali e registra
            if self.translation_obs_x:
                A = (x0 - half , yc )
                B = (x0 + half , yc )
            else:
                A = (x0 , yc - half)
                B = (x0 , yc + half)
            self.dyn_obs_pos.append([A[0], A[1]])
            return A, B

        # Fallback: usa il centro del path
        rospy.logwarn("dyn_obs_coord_generation: fallback verticale al centro del path")
        idx = n_pts // 2
        P = path[idx]
        x0, yc = float(P[0]), float(np.clip(P[1], y_lo + min_length/2, y_hi - min_length/2))
        A = (x0, yc - min_length/2)
        B = (x0, yc + min_length/2)
        self.dyn_obs_pos.append([A[0], A[1]])
        return A, B

    def dynamic_obstacles_movement(self):
        if len(self.dynamic_obstacles) == 0:
            return
        step = self.speed / self.update_obs_rate
        for i, obs in enumerate(self.dynamic_obstacles):
            sign = 1.0 if self.forward[i] else -1.0
            position = self.gazebo_obstacle_state(obs)
            if position is None:
                continue
            position[0] += sign * self.dir[i][0] * step
            position[1] += sign * self.dir[i][1] * step

            if self.forward[i]:
                if self._projected_distance(position, i) >= self._total_distance(i):
                    position = [self.dynamic_path_coords[i][1][0], self.dynamic_path_coords[i][1][1]]
                    self.forward[i] = False
            else:
                if self._projected_distance(position, i) <= 0.0:
                    position = [self.dynamic_path_coords[i][0][0], self.dynamic_path_coords[i][0][1]]
                    self.forward[i] = True

            self.gazebo_obstacles_update(obs, position)

    def _total_distance(self , idx):
        dx = self.dynamic_path_coords[idx][0][0] - self.dynamic_path_coords[idx][1][0]
        dy = self.dynamic_path_coords[idx][0][1] - self.dynamic_path_coords[idx][1][1]
        return math.sqrt(dx*dx + dy*dy)

    def _projected_distance(self, pos , idx):
        vx = pos[0] - self.dynamic_path_coords[idx][0][0]
        vy = pos[1] - self.dynamic_path_coords[idx][0][1]
        return vx*self.dir[idx][0] + vy*self.dir[idx][1]

    def spawn_model_from_sdf(self, model_name, file_path, x=0, y=0, z=0.0, 
                           roll=0, pitch=0, yaw=0, reference_frame="world"):
        try:
            with open(file_path, 'r') as f:
                sdf_content = f.read()
        except FileNotFoundError:
            rospy.logerr(f"SDF file not found: {file_path}")
            return None
        except Exception as e:
            rospy.logerr(f"Error reading SDF file: {e}")
            return None
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        try:
            pose = Pose()
            pose.position = Point(x=x, y=y, z=z)
            pose.orientation = Quaternion(x=qx , y=qy , z=qz, w=qw)
            response = self.spawn_model(
                model_name=model_name,
                model_xml=sdf_content,
                robot_namespace="",
                initial_pose=pose,
                reference_frame=reference_frame
            )
            if response.success:
                rospy.logdebug(f"Successfully spawned model '{model_name}' at position ({x}, {y}, {z})")
                return True
            else:
                rospy.logerr(f"Failed to spawn model: {response.status_message}")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
        
    def delete_model_from_gazebo(self, model_name):
        try:
            for model in model_name:
                rospy.logdebug(f"Attempting to delete model '{model}'")
                response = self.delete_model(model)
                rospy.sleep(0.5)
                if response.success:
                    rospy.logdebug(f"Successfully deleted model '{model}'")
                else:
                    rospy.logerr(f"Failed to delete model: {response.status_message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
        
    def gazebo_obstacle_state(self , obs_name):
        try:
            req = GetModelStateRequest()
            req.model_name = obs_name
            req.relative_entity_name = 'world'
            resp = self.get_dynamic_obstacles_pos(req)  
            if not resp.success:
                rospy.logwarn("Failed to get model state for '%s'" % obs_name)
                return None
            return [resp.pose.position.x,resp.pose.position.y]
        except rospy.ServiceException as e:
            rospy.logerr("Service call /gazebo/get_model_state failed: %s" % e)
            return None
        
    def gazebo_obstacles_update(self, model_name, obs_coord):
        try:
            obs_pose = ModelState()
            obs_pose.model_name = model_name
            obs_pose.pose.position.x = obs_coord[0]
            obs_pose.pose.position.y = obs_coord[1]
            obs_pose.pose.position.z = 0.5
            obs_pose.pose.orientation.x = 0.0
            obs_pose.pose.orientation.y = 0.0
            obs_pose.pose.orientation.z = 0.0
            obs_pose.pose.orientation.w = 1.0
            obs_pose.reference_frame = "world"
            self.set_dynamic_obstacles_pos(obs_pose)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False




### OLD VERSION WITH MOVEMENT OF DYNAMIC OBSTACLES MORE COMPLEX
"""class Environment_Management():

    def __init__(self):
        super().__init__()
        self.phase = '/' + rospy.get_param("/Curriculum_param/curriculum_phase")
        self.environment = rospy.get_param("/Curriculum_param/environment")
        self.environment_path = '/' + self.environment
        self.static_obstacles = rospy.get_param(self.phase + "/static_obstacles")
        self.dynamic_obstacles = rospy.get_param(self.phase + "/dynamic_obstacles")
        #[[x_first_zone],[x_second_zone]]
        self.zone_coord = rospy.get_param(self.environment_path + "/env_coord")
        rospy.loginfo("Zone coord : " + str(self.zone_coord) + " phase : " + str(self.phase) + " environment : " + str(self.environment) + " static obs : " + str(self.static_obstacles) + " dynamic obs : " + str(self.dynamic_obstacles))
        self.n_zones = 2  # Number of zones in the corridor
        #define start and goal position areas , give a margin to y coordinates in this way 
        #avoid conpenetration of robot with wall
        self.start_zone = [[self.zone_coord[0][0], self.zone_coord[0][0] + 2.5],
                            [self.zone_coord[1][0] + 0.6, self.zone_coord[1][1] - 0.6]]
        self.goal_zone = [[self.zone_coord[0][1] - 2.5, self.zone_coord[0][1]],
                            [self.zone_coord[1][0] + 0.6, self.zone_coord[1][1] - 0.6]]
        
        self.start_coord = []
        self.goal_coord = []
        
        rospy.loginfo("Start zone : " + str(self.start_zone) + " Goal zone : " + str(self.goal_zone))
        #define the zones of corridor in remaining part of environment
        self.zones = self.generate_zones(self.start_zone[0][1] , 
                                         self.goal_zone[0][0] ,
                                         self.n_zones)
        rospy.loginfo("Zones : " + str(self.zones))
        
        self.dynamic_path_coords = []
        self.room_zone = []
        self.random_start = rospy.get_param(self.phase + "/dyn_start")
        self.random_goal = rospy.get_param(self.phase + "/dyn_goal")
        self.speed = 0.3  # Speed of dynamic obstacles
        self.update_obs_rate = 5
        self.position = []
        self.dir = []
        self.forward = []
        self.static_obs_pos = []
        self.dyn_obs_pos = []
        #gazebo control
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        #dynamic obstacles topic
        self.get_dynamic_obstacles_pos = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_dynamic_obstacles_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def remove_obstacles(self):

        self.delete_model_from_gazebo(self.static_obstacles)
        
        self.delete_model_from_gazebo(self.dynamic_obstacles)

        # reset state
        self.position = []
        self.dynamic_path_coords = []
        self.dir = []
        self.forward = []
        self.static_obs_pos = []
        self.dyn_obs_pos = []
    
    def generate_coords(self):
        
        #Generate random coordinates for the start and goal positions of the robot.
        #If random_start is True, generate a random start position within the start zone.
        #If random_goal is True, generate a random goal position within the goal zone.
        
        if self.random_start:
            x_start = -3.5
            y_start = 2.5
        self.start_coord = [x_start, y_start]
        if self.random_goal:
            angle = random.uniform(-1.74, 1.74)  # random angle in radians [+- 100 degrees]
            x_goal = 3.5
            y_goal = random.uniform(-4.0 ,4.0)
        self.goal_coord = [x_goal, y_goal]
        return x_start, y_start, x_goal, y_goal
    
    def generate_zones(self , start_limit_zone , goal_limit_zone , n_zones):
        zones = []
        x_dim_zone = (goal_limit_zone - start_limit_zone) / n_zones
        for i in range(n_zones):
            x_start = start_limit_zone + i * x_dim_zone
            x_end = x_start + x_dim_zone
            zones.append([x_start, x_end])
        return zones
    
    def obstacles_generation(self , path):
        #spawn obstacles if evironment is corridor 
        if self.environment == 'Corridor':
            self.spawn_corridor_obs(path)
        else:
            self.spawn_room_obs()

    def spawn_corridor_obs(self , path ):
        # random.shuffle returns None; keep it simple and just shuffle in-place if needed
        # zones_idx = list(range(0, self.n_zones))
        # random.shuffle(zones_idx)
        n_obs = 0

        # --- STATIC OBSTACLES ---
        for i, obs in enumerate(self.static_obstacles):
            while True:
                idx = random.randint(0, path.shape[0] - 1)
                # check if path coordinates are at least 1.5 m from start and goal pos
                distance_from_start = np.linalg.norm(path[idx] - np.array(self.start_coord))
                distance_from_goal = np.linalg.norm(path[idx] - np.array(self.goal_coord))
                if distance_from_start >= 2.5 and distance_from_goal >= 1.5:
                    x = path[idx, 0]
                    y_noise = random.uniform(-1.5 , 1.5)
                    y = path[idx, 1] + y_noise 
                    if y <= self.zone_coord[1][0] :
                        y = self.zone_coord[1][0] + 0.2
                    elif y >= self.zone_coord[1][1]:
                        y = self.zone_coord[1][1] - 0.2
                    break
            rospy.loginfo("Static obstacle " + str(obs) + " at position (" + str(x) + " , " + str(y) + ")")
            self.static_obs_pos.append([x, y])
            self.spawn_model_from_sdf(
                obs,
                '/home/violo/tesi_project/src/model/obstacles/model.sdf',
                x=x,
                y=y
            )
            n_obs += 1

        # --- DYNAMIC OBSTACLES ---
        for i, obs in enumerate(self.dynamic_obstacles):
            A, B = self.dyn_obs_coord_generation(path)
            dx = B[0] - A[0]
            dy = B[1] - A[1]
            dist = math.sqrt(dx*dx + dy*dy)
            self.dir.append([dx/dist, dy/dist])
            self.forward.append(True)  # Initially moving forward

            rospy.loginfo(f"Dyn obstacle {obs}: A=({A[0]:.2f},{A[1]:.2f}) -> B=({B[0]:.2f},{B[1]:.2f})")

            self.dynamic_path_coords.append([[A[0], A[1]] , [B[0], B[1]]])
            self.position.append([A[0] , A[1]])

            self.spawn_model_from_sdf(
                obs,
                '/home/violo/tesi_project/src/model/obstacles/model.sdf',
                x=A[0],
                y=A[1]
            )

            n_obs += 1


    def spawn_room_obs(self):
        # not already implement 
        return
    
    # =====================
    #  GEOMETRY HELPERS
    # =====================
    def _point_segment_distance(self, p, a, b):
        #Distanza tra punto p e segmento ab (2D).
        px, py = p
        ax, ay = a
        bx, by = b
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx*abx + aby*aby
        if ab2 == 0.0:
            # a==b
            dx, dy = px - ax, py - ay
            return math.hypot(dx, dy)
        t = max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))
        cx = ax + t*abx
        cy = ay + t*aby
        return math.hypot(px - cx, py - cy)

    def _clamp_to_corridor(self, x, y, margin=0.2):
        #Mantieni il punto dentro alle pareti y del corridoio.
        y_min = self.zone_coord[1][0] + margin
        y_max = self.zone_coord[1][1] - margin
        return x, min(max(y, y_min), y_max)

    # =====================
    #  DYNAMIC PATH (A,B)
    # =====================
    def dyn_obs_coord_generation(self, path, *,
                                 min_from_start_goal=1.5,
                                 min_dyn_clear=2.0,
                                 min_static_clear=2.0,
                                 y_margin=0.2,
                                 d_range=(0.8, 1.6),
                                 along_range=(1.2, 3.8),
                                 min_length=4.0,
                                 edge_guard=5,
                                 max_tries=200):
        
        #Genera (A,B) per un ostacolo dinamico:
        #- A e B sono sui due lati del path (normale opposta)
        #- B è spostato lungo -tangente, così A->B è opposto al moto del robot
        #- Rispetta clearance da start/goal, ostacoli statici/dinamici e pareti
        
        n_pts = path.shape[0]
        tries = 0

        while tries < max_tries:
            tries += 1

            # 1) scegli un indice "interno" per evitare estremi del path
            idx = random.randint(edge_guard, max(edge_guard, n_pts - 1 - edge_guard))
            P = path[idx]
            Pm1 = path[idx - 1]
            Pp1 = path[idx + 1]

            # 2) distanze da start/goal
            if (np.linalg.norm(P - np.array(self.start_coord)) < min_from_start_goal or
                np.linalg.norm(P - np.array(self.goal_coord)) < min_from_start_goal):
                continue

            # 3) tangente e normale locali (finite differences)
            tx, ty = Pp1[0] - Pm1[0], Pp1[1] - Pm1[1]
            norm_t = math.hypot(tx, ty)
            if norm_t < 1e-6:
                continue
            tx, ty = tx / norm_t, ty / norm_t  # tangente unitaria
            nx, ny = -ty, tx                   # normale unitaria (rotazione +90°)

            # 4) ampiezze laterali e offset lungo -t (verso lo start)
            d = random.uniform(*d_range)
            alpha = random.uniform(*along_range)
            beta  = random.uniform(*along_range)

            A = np.array([P[0] + d*nx + beta*tx,  P[1] + d*ny + beta*ty])
            B = np.array([P[0] - d*nx - alpha*tx, P[1] - d*ny - alpha*ty])

            # 5) clamp dentro al corridoio
            A = np.array(self._clamp_to_corridor(A[0], A[1], margin=y_margin))
            B = np.array(self._clamp_to_corridor(B[0], B[1], margin=y_margin))

            # vincolo di lunghezza minima del segmento AB
            if np.linalg.norm(A - B) < min_length:
                continue

            # 6) clearance da statici/dinamici (punti A,B e segmento AB)
            ok = True

            for pos in self.static_obs_pos:
                pos = np.array(pos)
                if (np.linalg.norm(A - pos) < min_static_clear or
                    np.linalg.norm(B - pos) < min_static_clear or
                    self._point_segment_distance(pos, A, B) < min_static_clear):
                    ok = False
                    break
            if not ok:
                continue

            for pos in self.dyn_obs_pos:
                pos = np.array(pos)
                if (np.linalg.norm(A - pos) < min_dyn_clear or
                    np.linalg.norm(B - pos) < min_dyn_clear or
                    self._point_segment_distance(pos, A, B) < min_dyn_clear):
                    ok = False
                    break
            if not ok:
                continue

            # 7) anche A/B devono restare lontani da start/goal
            if (np.linalg.norm(A - np.array(self.start_coord)) < min_from_start_goal or
                np.linalg.norm(A - np.array(self.goal_coord)) < min_from_start_goal or
                np.linalg.norm(B - np.array(self.start_coord)) < min_from_start_goal or
                np.linalg.norm(B - np.array(self.goal_coord)) < min_from_start_goal):
                continue

            # tutto ok -> registra la posizione di spawn (A) per future clearance
            self.dyn_obs_pos.append(A.tolist())
            return tuple(A.tolist()), tuple(B.tolist())

        rospy.logwarn("dyn_obs_coord_generation: non ho trovato una coppia (A,B) valida; ripiego su offset casuale corto.")
        # fallback: piccolo offset casuale attorno a un punto interno
        idx = max(edge_guard, min(n_pts - 1 - edge_guard, n_pts // 2))
        P = path[idx]
        A = (P[0], P[1] + 0.8)
        B = (P[0], P[1] - 0.8)
        A = self._clamp_to_corridor(A[0], A[1], margin=y_margin)
        B = self._clamp_to_corridor(B[0], B[1], margin=y_margin)
        self.dyn_obs_pos.append([A[0], A[1]])
        return A, B
            

    def dynamic_obstacles_movement(self):
        # Calcolo del passo in metri per update
        step = self.speed / self.update_obs_rate
        for i, obs in enumerate(self.dynamic_obstacles):
            # Avanza o indietreggia
            sign = 1.0 if self.forward[i] else -1.0
            position = self.gazebo_obstacle_state(obs)
            if position is None:
                continue
            # Calcolo nuova posizione lungo la direzione normalizzata
            position[0] += sign * self.dir[i][0] * step #need to modify for every dynamic obj
            position[1] += sign * self.dir[i][1] * step #idem

            # Controllo oltrepassamento di fine tratta
            if self.forward[i]:
                # proiezione del vettore corrente su AB >= lunghezza AB?
                if self._projected_distance(position, i) >= self._total_distance(i):
                    position = [self.dynamic_path_coords[i][1][0], self.dynamic_path_coords[i][1][1]]
                    self.forward[i] = False
            else:
                # distanza da A <= 0 -> si torna a inizio tratta
                if self._projected_distance(position, i) <= 0.0:
                    position = [self.dynamic_path_coords[i][0][0], self.dynamic_path_coords[i][0][1]]
                    self.forward[i] = True
                        
            self.gazebo_obstacles_update(obs, position)

    def _total_distance(self , idx):
        dx = self.dynamic_path_coords[idx][0][0] - self.dynamic_path_coords[idx][1][0]
        dy = self.dynamic_path_coords[idx][0][1] - self.dynamic_path_coords[idx][1][1]
        return math.sqrt(dx*dx + dy*dy)

    def _projected_distance(self, pos , idx):
        # distanza scalare di pos lungo AB rispetto ad A
        vx = pos[0] - self.dynamic_path_coords[idx][0][0]
        vy = pos[1] - self.dynamic_path_coords[idx][0][1]
        return vx*self.dir[idx][0] + vy*self.dir[idx][1]

    def spawn_model_from_sdf(self, model_name, file_path, x=0, y=0, z=1, 
                           roll=0, pitch=0, yaw=0, reference_frame="world"):
        
        #Load sdf model
        try:
            with open(file_path, 'r') as f:
                sdf_content = f.read()
        except FileNotFoundError:
            rospy.logerr(f"SDF file not found: {file_path}")
            return None
        except Exception as e:
            rospy.logerr(f"Error reading SDF file: {e}")
            return None
        #Spawn model in Gazebo from SDF content
        try:
            # Create pose for the model
            pose = Pose()
            pose.position = Point(x=x, y=y, z=z)
            
            # Convert RPY to quaternion (simplified for basic rotations)
            pose.orientation = Quaternion(x=roll , y=pitch , z=yaw, w=1)
            
            # Call the spawn service
            response = self.spawn_model(
                model_name=model_name,
                model_xml=sdf_content,
                robot_namespace="",
                initial_pose=pose,
                reference_frame=reference_frame
            )
            
            if response.success:
                rospy.loginfo(f"Successfully spawned model '{model_name}' at position ({x}, {y}, {z})")
                return True
            else:
                rospy.logerr(f"Failed to spawn model: {response.status_message}")
                return False
                
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
        
    def delete_model_from_gazebo(self, model_name):
        #Delete a model from Gazebo
        try:
            for model in model_name:
                rospy.loginfo(f"Attempting to delete model '{model}'")
                response = self.delete_model(model)
                rospy.sleep(0.5)
                if response.success:
                    rospy.loginfo(f"Successfully deleted model '{model}'")
                else:
                    rospy.logerr(f"Failed to delete model: {response.status_message}")
                
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
        
    def gazebo_obstacle_state(self , obs_name):
        
        #Calls /gazebo/get_model_state and returns a GetModelStateResponse,
        #which contains `.pose` and `.twist` of the specified model.
        

        try:
            # Create a proxy and request object
            req = GetModelStateRequest()
            req.model_name = obs_name
            req.relative_entity_name = 'world'

            # Call the service
            resp = self.get_dynamic_obstacles_pos(req)  

            if not resp.success:
                rospy.logwarn("Failed to get model state for '%s'" % obs_name)
                return None

            # Return the full response (pose + twist + success flag)
            return [resp.pose.position.x,resp.pose.position.y]

        except rospy.ServiceException as e:
            rospy.logerr("Service call /gazebo/get_model_state failed: %s" % e)
            return None
        
    def gazebo_obstacles_update(self, model_name, obs_coord):
        #Update robot pose after spawning
        try:
            obs_pose = ModelState()
            obs_pose.model_name = model_name
            # Set position for start_pose
            obs_pose.pose.position.x = obs_coord[0]  # Starting x-coordinate
            obs_pose.pose.position.y = obs_coord[1]  # Starting y-coordinate
            obs_pose.pose.position.z = 0.5  # Often 0 for a 2D navigation

            # Set orientation for start_pose (quaternion values for no rotation)
            obs_pose.pose.orientation.x = 0.0
            obs_pose.pose.orientation.y = 0.0
            obs_pose.pose.orientation.z = 0.0
            obs_pose.pose.orientation.w = 1.0  # w = 1 means no rotation

            obs_pose.reference_frame = "world"

            self.set_dynamic_obstacles_pos(obs_pose)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
"""
