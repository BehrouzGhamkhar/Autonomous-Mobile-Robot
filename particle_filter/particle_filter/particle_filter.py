import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist, PoseArray, Pose, PoseWithCovarianceStamped
from tf2_msgs.msg import TFMessage
import numpy as np
import math
import tf_transformations, tf2_ros, tf2_geometry_msgs
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import random
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math
from numpy import pi
import sys
from scipy.stats import norm


class ParticleFilter(Node):
    def __init__(self,node_name: str):
        super().__init__(node_name)
        #self.map_width=7.5
        #self.map_height=6.5
        #self.map_resolution=0.05
        self.no_of_init_hypotheses=10
        self.dist_thresh=0.3#Minimum movement required to update particles
        self.theta_thresh=0.4

        self.particles=[]

        self.last_used_pose: Pose = None
        self.last_pose: Pose = None
        self.last_scan: LaserScan = None
        self.receive_new_scan_pose=False
        self.init_pose_received=False


        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        depth=10
        )

        self.pub_hypotheses = self.create_publisher(PoseArray, "/hypotheses", 10)
        self.curr_pose_sub = self.create_subscription(Odometry, "/odom", self.odomCallback, 10)
        self.laser_scan_sub = self.create_subscription(LaserScan, "/scan", self.scanCallback, 10)
        self.init_pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initposeCallback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.mapCallback, qos_profile)
        
        self.timer_block=self.create_timer(0.1,self.timerCallback)

        #self.generate_hypotheses()

    def initposeCallback(self,msg):

        self.init_pose_received=True


        self.last_used_pose= msg.pose.pose
        self.last_pose= msg.pose.pose

        for i in range(self.no_of_init_hypotheses):

            init_pose_x=msg.pose.pose.position.x
            init_pose_y=msg.pose.pose.position.y
            init_pose_orientation=tf_transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
            
            x=random.uniform(init_pose_x-0.5, init_pose_x+0.5)
            y=random.uniform(init_pose_y-0.5, init_pose_y+0.5)
            yaw=random.uniform(init_pose_orientation,init_pose_orientation)
            weight=1/self.no_of_init_hypotheses #Assigning uniform weight at start       

            self.particles.append((x,y,yaw,weight))
            #print(f"{i}th point is {x, y, yaw, weight}")
        
        self.publish_particles(self.particles)

    def publish_particles(self,particles):
        hypothese=PoseArray()
        hypothese.header.frame_id='odom'
        for i,data in enumerate(particles):
            point=Pose()

            point.position.x=data[0]
            point.position.y=data[1]
            point.position.z=0.0

            x,y,z,w=tf_transformations.quaternion_from_euler(0.0,0.0,data[2])
            point.orientation.x=x
            point.orientation.y=y
            point.orientation.z=z
            point.orientation.w=w

            hypothese.poses.append(point)
        
        self.pub_hypotheses.publish(hypothese)

    def odomCallback(self,msg):
        if self.receive_new_scan_pose==True:
            self.last_pose=msg.pose.pose


    def mapCallback(self, msg):
        self.map_published=True
        self.get_logger().info('Received map data')

        # Store the occupancy grid data in a 2D array
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution=msg.info.resolution
        self.map_data = msg.data
        self.map_origin=[msg.info.origin.position.x, msg.info.origin.position.y]

        '''
        self.occupancy_grid = np.array([[0 for _ in range(self.map_height)] for _ in range(self.map_width)])

        for i in range(self.map_height):
            for j in range(self.map_width):
                self.occupancy_grid[j][i] = self.map_data[i * self.map_width + j]

        self.occupancy_grid = np.rot90(self.occupancy_grid, 2)

        self.occupancy_origin = [msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z]
        self.map_scale = 1 / round(self.map_resolution, 4)
        '''

    def scanCallback(self,msg):
        if self.receive_new_scan_pose==True:
            self.last_scan=msg

    def prediction_step(self, current_pose,last_pose,last_used_pose):
        dx=last_pose.position.x-last_used_pose.position.x
        dy=last_pose.position.y-last_used_pose.position.y
        #print(dx,dy)
        theta=tf_transformations.euler_from_quaternion([last_used_pose.orientation.x, last_used_pose.orientation.y, last_used_pose.orientation.z, last_used_pose.orientation.w])[2]
        theta_final=tf_transformations.euler_from_quaternion([last_pose.orientation.x,last_pose.orientation.y,last_pose.orientation.z,last_pose.orientation.w])[2]

        dtrans=np.sqrt(dx*dx+dy*dy)
        #print(f"d_trans:{dtrans}")
        drot1=np.arctan2(dy,dx)-theta
        drot2=theta_final-theta-drot1


        #With Gaussian Noise
        t_d1 = drot1 + np.random.normal(0, 0.01)
        t_dt = dtrans + np.random.normal(0, 0.01)
        t_d2 = drot2 + np.random.normal(0, 0.1)
        #print(f"Received data in function: {t_d1,t_dt,t_d2}")

        x_new=current_pose[0] + t_dt*math.cos(current_pose[2]+t_d1)
        y_new=current_pose[1] + t_dt*math.sin(current_pose[2]+t_d1)
        theta_new=current_pose[2]+ t_d1 + t_d2

        return [x_new, y_new, theta_new]
    
    def calculate_reference_index(self, range_len):
        selected_scan_indices=[]
        selected_scan_indices.append(0) #Appending first index
        for i in range(9,range_len,10):
            selected_scan_indices.append(i)

        return selected_scan_indices
    
    def to_xy(self, msg, pose, ref_ind):
        m_array = []
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        for index in ref_ind:
            if math.isinf(msg.ranges[index]):
                m_array.append([msg.range_max, pose[2]+angle_min + index * angle_increment])
            else:
                m_array.append([msg.ranges[index], pose[2]+angle_min + index * angle_increment])
        np_array = np.array(m_array)

        #print(np_array)

        #converting polar to cartesian
        """
        np_polar: 2-dimenisional numpy array, where each row consists of the distance to the obstacle and the
                corresponding angle in radians. Thus, number of rows will be equal to the number of scan data points
        Return:
        np_cart: 2-dimenisional numpy array, where each row consists of X and Y coordinates in Cartesian system
        """
        np_cart = []

        for i, (radius, angle) in enumerate(np_array):
            coordinates = [radius * math.cos(angle), radius * math.sin(angle)]
            np_cart.append(coordinates)

        #print(np_array)

        return np_cart
    
    def to_grid_pt(self,xy_pts):
        grid_pts=[]

        for pts in xy_pts:
            i = np.round((-self.map_origin[0]+pts[0])/self.map_resolution)
            j = np.round((-self.map_origin[1]+pts[1])/self.map_resolution)
            grid_pts.append([i,j])

        return grid_pts

    def find_objects(self):
        object_index_list = []
        for x in range(self.map_width):
            for y in range(self.map_height):
                index = y * self.map_width + x
                if index >= len(self.map_data) or index < -len(self.map_data):
                    continue
                if self.map_data[index] >= 70:
                    object_index_list.append((x, y, index))
        return object_index_list


    def closest_object_dist(self, x, y):
        dist = sys.float_info.max
        #self._object_list=self.find_objects()
        for object in self._object_list:
            current_dist = np.sqrt(((object[0] - x) ** 2) + ((object[1] - y) ** 2))+np.random.normal(0, 0.01)#Adding gaussian noise for sensor measurement
            if current_dist < dist:
                dist = current_dist
        return dist

    def resample(self,particles):
        resampled_particles=[]
        
        cum_sum=particles[0][3]
        r=random.uniform(0,1/len(particles))
        i=0

        for j in range(len(particles)):
            U=r+j*1/len(particles)
            while U>cum_sum:
                i=i+1
                #if(i>=len(particles)):
                #    return resampled_particles
                cum_sum=cum_sum+particles[i][3]

            resampled_particles.append(particles[i])

        return resampled_particles
    
    def cal_rel_position_of_laser_pts(self,scan_msg,curr_pose):
        ref_indices=[]
        laser_pts_in_xy=[]
        self._object_list=self.find_objects()
        ref_indices=self.calculate_reference_index(len(scan_msg.ranges))
        #print(f"Length of scanned elements: {selected_scan_elements}")

        laser_pts_in_xy=self.to_xy(scan_msg, curr_pose,ref_indices)
        laser_pts_in_grid_pts=self.to_grid_pt(laser_pts_in_xy)
        #print(laser_pts_in_grid_pts)
        dist=[]
        for grid_pt in laser_pts_in_grid_pts:
            dist.append(self.closest_object_dist(grid_pt[0],grid_pt[1]))
        
        return dist
    
    def calc_particle_weight(self, ref_rel_pos,curr_particle_rel_pos):
        weight=0
        for i in range(len(ref_rel_pos)):
            weight=weight+abs(ref_rel_pos[i]-curr_particle_rel_pos[i])

        
        return weight
    
    def normalize_particles(self,particles):
        sum_of_weights=0
        normalized_weights=[]
        normalized_particles=[]
        for particle in particles:
            sum_of_weights=sum_of_weights+particle[3]

        print(f"Particle sum:{sum_of_weights}")

        for i,particle in enumerate(particles):
            norm_wt=(particle[3]/sum_of_weights)
            normalized_particles.append([particle[0], particle[1], particle[2],norm_wt])

        norm_wt_sum=0
        for i,particle in enumerate(normalized_particles):
            norm_wt_sum=norm_wt_sum+particle[3]

        print(f"Norm_wt_sum:{norm_wt_sum}")
        

        return normalized_particles
        
    def timerCallback(self):
        #if self.last_used_pose and self.last_pose:
        if self.init_pose_received==False:
            return
        

        self.receive_new_scan_pose=True
        
        distance_from_last_pose=np.sqrt(np.square(self.last_used_pose.position.x-self.last_pose.position.x)+np.square(self.last_used_pose.position.y-self.last_pose.position.y))
        #print(distance_from_last_pose)
        theta_last_used_pose=tf_transformations.euler_from_quaternion([self.last_used_pose.orientation.x, self.last_used_pose.orientation.y, self.last_used_pose.orientation.z, self.last_used_pose.orientation.w])[2]
        theta_last_pose=tf_transformations.euler_from_quaternion([self.last_pose.orientation.x,self.last_pose.orientation.y,self.last_pose.orientation.z,self.last_pose.orientation.w])[2]
        rotation_from_last_pose=theta_last_used_pose-theta_last_pose
        current_pose=[self.last_pose.position.x, self.last_pose.position.y, theta_last_pose]
        #print(f"Rotation_from_last_pose: {rotation_from_last_pose}")
        new_particles=[]
        relative_position_data_for_particles=[]
        particle_weight=[]
        max = 0.5
        w = []
        c = []
        for i in range(len(self.particles)):
            #print(i)
            w.append(random.uniform(0, max))
            if i == 0:
                c.append(w[i])
            else :
                c.append(c[i-1]+w[i])
            max = (1-c[i])/2
        w.append(1-c[len(self.particles)-2])

        
        
        if distance_from_last_pose<self.dist_thresh and abs(rotation_from_last_pose)<self.theta_thresh:
            #print("Movement is not enough")
            return
        else:
            self._object_list=self.find_objects()
            ref_relative_position=self.cal_rel_position_of_laser_pts(self.last_scan,current_pose)
            print(f"Relative_position:{ref_relative_position}")

            for i, particle in enumerate(self.particles):
                #print("Movement is enough")
                #current_particle_pose=particle
                #print(f"Current_pose: {current_particle_pose}")

                predicted_pose=self.prediction_step(particle,self.last_pose,self.last_used_pose)
                #print(f"Predicted_pose: {predicted_pose}")
                #prob=self.measurement_update(self.last_scan, self.last_pose)
                #print(f"Prob:{prob}")

                rel_position_for_each_particle=self.cal_rel_position_of_laser_pts(self.last_scan,predicted_pose)
                relative_position_data_for_particles.append(rel_position_for_each_particle)
                weight=self.calc_particle_weight(ref_relative_position,rel_position_for_each_particle)
                if (weight<0.0001):
                    weight=0.001
                
                new_particles.append([predicted_pose[0], predicted_pose[1], predicted_pose[2], 1/weight])

                
                #self.measurement_update()
            self.particles.clear()
            print(f"New_particle:{new_particles}")
            #print(f"Length:{len(relative_position_data_for_particles)}")
            #print(f"Particle_weight:{len(particle_weight)}")
            normalized_particles = self.normalize_particles(new_particles)
            print(f"normalized_particles:{normalized_particles}")
            '''
            
            if normalized_particles is None:
                print("No normalized particles")
            '''
            self.particles=self.resample(normalized_particles)
            print(f"Number of particles:{len(self.particles)}")

            self.publish_particles(self.particles)
            self.last_used_pose=self.last_pose
            self.receive_new_scan_pose=False



def main(args=None):
    rclpy.init(args=args)
    sub = ParticleFilter("particle_filter")
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()