import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist, PoseArray, Pose, PoseWithCovarianceStamped, TransformStamped
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
import tf2_ros
from tf2_ros import TransformBroadcaster


class ParticleFilter(Node):
    def __init__(self,node_name: str):
        super().__init__(node_name)
        self.no_of_init_hypotheses=10
        self.dist_thresh=0.3#Minimum movement required to update particles
        self.theta_thresh=0.4#Minimum rotation required to update particles

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
        self.pub_estimated_pose = self.create_publisher(PoseStamped, "/estimated_pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.curr_pose_sub = self.create_subscription(Odometry, "/odom", self.odomCallback, 10)
        self.laser_scan_sub = self.create_subscription(LaserScan, "/scan", self.scanCallback, 10)
        self.init_pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initposeCallback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.mapCallback, qos_profile)
        
        self.timer_block=self.create_timer(0.1,self.timerCallback)

    def initposeCallback(self,msg):

        self.particles.clear()
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
        
        self.publish_particles(self.particles)
        self.publish_estimated_pose(self.particles)

    def publish_particles(self,particles):
        #This function publishes /hypotheses topic 
        hypothese=PoseArray()
        hypothese.header.frame_id='map'
        for data in particles:
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

    def publish_estimated_pose(self,particles):
        #This function publishes /estimated_pose topic
        estimated_pose=PoseStamped()
        x=0
        y=0
        yaw=0

        estimated_pose.header.frame_id='map'
        for data in particles:
            x+=data[0]
            y+=data[1]
            yaw+=data[2]

        x=x/len(particles)
        y=y/len(particles)
        yaw=yaw/len(particles)

        estimated_pose.pose.position.x=x
        estimated_pose.pose.position.y=y
        estimated_pose.pose.position.z=0.0

        rx,ry,rz,w=tf_transformations.quaternion_from_euler(0.0,0.0,yaw)
        estimated_pose.pose.orientation.x=rx
        estimated_pose.pose.orientation.y=ry
        estimated_pose.pose.orientation.z=rz
        estimated_pose.pose.orientation.w=w
       
        self.pub_estimated_pose.publish(estimated_pose)

    
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


    def scanCallback(self,msg):
        if self.receive_new_scan_pose==True:
            self.last_scan=msg

    def prediction_step(self, current_pose,last_pose,last_used_pose):
        dx=last_pose.position.x-last_used_pose.position.x
        dy=last_pose.position.y-last_used_pose.position.y
        theta=tf_transformations.euler_from_quaternion([last_used_pose.orientation.x, last_used_pose.orientation.y, last_used_pose.orientation.z, last_used_pose.orientation.w])[2]
        theta_final=tf_transformations.euler_from_quaternion([last_pose.orientation.x,last_pose.orientation.y,last_pose.orientation.z,last_pose.orientation.w])[2]

        dtrans=np.sqrt(dx*dx+dy*dy)
        drot1=np.arctan2(dy,dx)-theta
        drot2=theta_final-theta-drot1

        #Addding Gaussian Noise in yaw and 
        t_d1 = drot1 + np.random.normal(0, 0.01)
        t_dt = dtrans + np.random.normal(0, 0.01)
        t_d2 = drot2 + np.random.normal(0, 0.1)

        x_new=current_pose[0] + t_dt*math.cos(current_pose[2]+t_d1)
        y_new=current_pose[1] + t_dt*math.sin(current_pose[2]+t_d1)
        theta_new=current_pose[2]+ t_d1 + t_d2

        return [x_new, y_new, theta_new]
    
    def calculate_reference_index(self, range_len):
        #This function calculates reference indexes for sensor measurement and stores them in selected_scan_indices
        #Here I am selecting first element, then every 10th index.
        selected_scan_indices=[]
        selected_scan_indices.append(0) #Appending first index
        for i in range(9,range_len,10):
            selected_scan_indices.append(i)

        return selected_scan_indices
    
    def to_xy(self, msg, pose, ref_ind):
        #Extracting scan data and arranging it in an array as range, angle
        m_array = []
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        for index in ref_ind:
            if math.isinf(msg.ranges[index]):
                m_array.append([msg.range_max, pose[2]+angle_min + index * angle_increment])
            else:
                m_array.append([msg.ranges[index], pose[2]+angle_min + index * angle_increment])
        np_array = np.array(m_array)

        #Converting polar to cartesian
        np_cart = []

        for i, (radius, angle) in enumerate(np_array):
            coordinates = [radius * math.cos(angle), radius * math.sin(angle)]
            np_cart.append(coordinates)

        return np_cart
    
    def to_grid_pt(self,xy_pts):
        #Converting from x-y coo-rdinate to grid location
        grid_pts=[]

        for pts in xy_pts:
            i = np.round((pts[0]-self.map_origin[0])/self.map_resolution)
            j = np.round((pts[1]-self.map_origin[1])/self.map_resolution)
            grid_pts.append([i,j])

        return grid_pts

    def find_objects(self):
        #Scanning the map data and finding indexes of occupied cells
        object_index_list = []
        for x in range(self.map_width):
            for y in range(self.map_height):
                index = y * self.map_width + x
                if index >= len(self.map_data) or index < -len(self.map_data):
                    continue
                if self.map_data[index] >= 50:
                    object_index_list.append((x, y, index))
        return object_index_list


    def closest_object_dist(self, x, y):
        #This function finds nearest object to a grid pt x,y
        dist = sys.float_info.max
        for object in self.object_list:
            current_dist = np.sqrt(((object[0] - x) ** 2) + ((object[1] - y) ** 2))+np.random.normal(0, 0.01)#Adding gaussian noise in sensor measurement
            if current_dist < dist:
                dist = current_dist
        return dist

    def resample(self,particles):
        #This function performs stochastic universal sampling algorithms to resample particles based on their importance weight
        #This algorithm is referred from S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics (Intelligent Robotics and Autonomous Agents). The MIT Press, 2005
        resampled_particles=[]
        
        cum_sum=particles[0][3]
        r=random.uniform(0,1/len(particles))
        i=0

        for j in range(len(particles)):
            U=r+j*1/len(particles)
            while U>cum_sum:
                i=i+1
                cum_sum=cum_sum+particles[i][3]

            resampled_particles.append(particles[i])

        return resampled_particles
    
    def cal_rel_position_of_laser_pts(self,scan_msg,curr_pose):

        ref_indices=[]
        laser_pts_in_xy=[]
        self.object_list=self.find_objects()
        ref_indices=self.calculate_reference_index(len(scan_msg.ranges))

        laser_pts_in_xy=self.to_xy(scan_msg, curr_pose,ref_indices)
        laser_pts_in_grid_pts=self.to_grid_pt(laser_pts_in_xy)

        dist=[]
        for grid_pt in laser_pts_in_grid_pts:
            dist.append(self.closest_object_dist(grid_pt[0],grid_pt[1]))
        
        return dist
    
    def calc_particle_weight(self, ref_rel_pos,curr_particle_rel_pos):
        #This function calculates importance weight of the particle by taking sum of difference between reference scan and particle scan
        weight=0
        for i in range(len(ref_rel_pos)):
            weight=weight+abs(ref_rel_pos[i]-curr_particle_rel_pos[i])

        return weight
    
    def normalize_particles(self,particles):
        # Normalising particle weights between 0-1
        sum_of_weights=0
        normalized_weights=[]
        normalized_particles=[]
        for particle in particles:
            sum_of_weights=sum_of_weights+particle[3]

        for i,particle in enumerate(particles):
            norm_wt=(particle[3]/sum_of_weights)
            normalized_particles.append([particle[0], particle[1], particle[2],norm_wt])

        norm_wt_sum=0
        for i,particle in enumerate(normalized_particles):
            norm_wt_sum=norm_wt_sum+particle[3]

        return normalized_particles
    
    def resample_around_top3(self, resampled_particles):

        sorted_tuples = sorted(resampled_particles, key=lambda x: x[3], reverse=True)

        # Extract unique tuples based on the weight value
        unique_top_particles = []
        seen_third_values = set()

        for t in sorted_tuples:
            third_value = t[2]
            if third_value not in seen_third_values:
                unique_top_particles.append(t)
                seen_third_values.add(third_value)
            if len(unique_top_particles) == 3:  # Stop once we have the top 3 unique elements
                break

        self.particles.clear()
        for particle in unique_top_particles:
            for j in range(3):
                self.particles.append([np.random.normal(particle[0], 0.01), np.random.normal(particle[1], 0.01), np.random.normal(particle[2], 0.01), 1/(len(unique_top_particles))])
        
    def timerCallback(self):

        if self.init_pose_received==False:
            return
        
        self.receive_new_scan_pose=True
        
        distance_from_last_pose=np.sqrt(np.square(self.last_used_pose.position.x-self.last_pose.position.x)+np.square(self.last_used_pose.position.y-self.last_pose.position.y))
        theta_last_used_pose=tf_transformations.euler_from_quaternion([self.last_used_pose.orientation.x, self.last_used_pose.orientation.y, self.last_used_pose.orientation.z, self.last_used_pose.orientation.w])[2]
        theta_last_pose=tf_transformations.euler_from_quaternion([self.last_pose.orientation.x,self.last_pose.orientation.y,self.last_pose.orientation.z,self.last_pose.orientation.w])[2]
        rotation_from_last_pose=theta_last_used_pose-theta_last_pose
        current_pose=[self.last_pose.position.x, self.last_pose.position.y, theta_last_pose]

        new_particles=[]
        relative_position_data_for_particles=[]
        #This condition checks whether robot has moved/rotated more than dist_thresh and theta_thresh
        if distance_from_last_pose<self.dist_thresh and abs(rotation_from_last_pose)<self.theta_thresh:
            return
        else:
            self._object_list=self.find_objects()
            
            ref_relative_position=self.cal_rel_position_of_laser_pts(self.last_scan,current_pose)#Calculating how far measured laser points are from surrounding obstacles/occupied cells

            for i, particle in enumerate(self.particles):

                predicted_pose=self.prediction_step(particle,self.last_pose,self.last_used_pose)
                
                rel_position_for_each_particle=self.cal_rel_position_of_laser_pts(self.last_scan,predicted_pose)#Calculating how far measured laser points are from surrounding obstacles/occupied cells
                relative_position_data_for_particles.append(rel_position_for_each_particle)
                weight=self.calc_particle_weight(ref_relative_position,rel_position_for_each_particle)
                
                if (weight<0.0001):#this condition prevents shooting of 1/weight value
                    weight=0.001

                #Here 1/weight is appended because particle whose scan doesn't match well with reference scan will have higher weight but it should have lower importance weight
                new_particles.append([predicted_pose[0], predicted_pose[1], predicted_pose[2], 1/weight])

            
            self.particles.clear()
            #Normalizing importance weights of the particles
            normalized_particles = self.normalize_particles(new_particles)
         
            self.particles=self.resample(normalized_particles)
            
            self.publish_particles(self.particles)
            self.publish_estimated_pose(self.particles)

            self.resample_around_top3(self.particles)

            self.last_used_pose=self.last_pose
            self.receive_new_scan_pose=False



def main(args=None):
    rclpy.init(args=args)
    sub = ParticleFilter("particle_filter")
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()