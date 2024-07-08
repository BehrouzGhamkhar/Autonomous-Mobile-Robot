import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from tf2_msgs.msg import TFMessage
import numpy as np
import math
import sys
import tf_transformations, tf2_ros, tf2_geometry_msgs
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import Odometry,OccupancyGrid
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class grid_map(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.map_resolution=0.1
        self.p_prior=0.5
        self.p_occ=0.9
        self.p_free=0.2
        self.thresh_p_occ=0.6
        self.thresh_p_free=0.3
        self.map_height=20.0
        self.map_width=20.0
        self.map_scale=math.floor(1/self.map_resolution)

        self.l_occ= self.probability_to_log_odd(self.p_occ)
        self.l_free = self.probability_to_log_odd(self.p_free)
        self.l_prior = self.probability_to_log_odd(self.p_prior)

        self.env_data=[]
        self.curr_pose_gridpt=[0,0]
        self.vehicle_moving=False

        #Creating Occupancy grid array
        map_rows = int(self.map_height / self.map_resolution)
        map_cols = int(self.map_width/ self.map_resolution)
        self.occupancy_grid_arr = self.l_prior * np.ones((map_rows, map_cols))


        #Declaring publishers and suscribers
        self.laser_scan_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        self.curr_pose_sub = self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.curr_pose_sub = self.create_subscription(Twist,"/cmd_vel",self.cmd_vel_callback,10)
        self.pub_occ_grid=self.create_publisher(OccupancyGrid,"/map",10)   

        #TF_buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)  

    def cmd_vel_callback(self, msg):
        if(msg.linear.x==0):
            self.vehicle_moving=True
        else:
            self.vehicle_moving=False
    
    def log_odd_to_probability(self,l):
        return 1 - (1/(1+np.exp(l)))


    def probability_to_log_odd(self,p):
        return np.log(p/(1-p))
    
    def extract_ranges(self, msg, yaw):
        m_array = []
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        for index, distance in enumerate(msg.ranges):
            m_array.append([distance, yaw+angle_min + index * angle_increment])

        np_array = np.array(m_array)
        return np_array
    
    def np_polar2cart(self, np_polar):
        """
        np_polar: 2-dimenisional numpy array, where each row consists of the distance to the obstacle and the
                corresponding angle in radians. Thus, number of rows will be equal to the number of scan data points
        Return:
        np_cart: 2-dimenisional numpy array, where each row consists of X and Y coordinates in Cartesian system
        """
        np_cart = []

        for i, (radius, angle) in enumerate(np_polar):
            coordinates = [radius * math.cos(angle), radius * math.sin(angle)]
            np_cart.append(coordinates)

            # test if it is working: length = sqrt(x^2 + y^2)
            # if (math.isclose(math.sqrt(coordinates[0] ** 2 + coordinates[1] ** 2), radius)):
            #    print("True")

        return np.array(np_cart)
    
    def to_grid_pt(self,x,y):

        i = np.round(y*self.map_scale)
        j = np.round(x*self.map_scale)

        return i, j
    
    #This is taken from https://github.com/salihmarangoz/robot_laser_grid_mapping/blob/main/scripts/grid_mapping.py
    def is_inside (self, i, j):
        return i<self.occupancy_grid_arr.shape[0] and j<self.occupancy_grid_arr.shape[1] and i>=0 and j>=0
    
    #This is taken from https://github.com/salihmarangoz/robot_laser_grid_mapping/blob/main/scripts/grid_mapping.py
    def bresenham (self, i0, j0, i1, j1, d,debug=False):   # i0, j0 (starting point)

        dx = np.absolute(j1-j0)
        dy = -1 * np.absolute(i1-i0)
        sx = -1
        if j0<j1:
            sx = 1
        sy = -1
        if i0<i1:
            sy = 1
        jp, ip = j0, i0
        err = dx+dy                     # error value e_xy
        while True:                     # loop
            if (jp == j1 and ip == i1) or (np.sqrt((jp-j0)**2+(ip-i0)**2) >= d) or not self.is_inside(ip, jp):
                return ip, jp, False
            elif self.occupancy_grid_arr[int(ip),int(jp)]==100:
                return ip, jp, True

            if self.is_inside(ip, jp):
                # miss:
                self.occupancy_grid_arr[int(ip),int(jp)] += self.l_free - self.l_prior

            e2 = 2*err
            if e2 >= dy:                # e_xy+e_x > 0 
                err += dy
                jp += sx
            if e2 <= dx:                # e_xy+e_y < 0
                err += dx
                ip += sy

    
    def laser_callback(self, msg):
        self.env_data.clear()

        self.range_max=msg.range_max
        self.range_min=msg.range_min

        np_polar_array = self.extract_ranges(msg,self.robot_pose[2])
        np_cart_array = self.np_polar2cart(np_polar_array)

        #Getting laser_link w.r.t. map
        try:
            self.trans_base_laser_to_map = self.tf_buffer.lookup_transform('map', 'base_laser_front_link', rclpy.time.Time())

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("Failed to transform")
            return

        #Now I am storing environment data as [angle,range,cartesian_x, cartesian_y]
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min

        for i,distance in enumerate(msg.ranges):
            self.env_data.append([angle_min + i * angle_increment,distance,np_cart_array[i][0],np_cart_array[i][1]])

        i0,j0=self.to_grid_pt(self.trans_base_laser_to_map.transform.translation.x,self.trans_base_laser_to_map.transform.translation.y)

        for index,data in enumerate(self.env_data):
            if not math.isinf(data[1]) and self.vehicle_moving==True:
                laser_pt_in_x=data[2]+self.trans_base_laser_to_map.transform.translation.x
                laser_pt_in_y=data[3]+self.trans_base_laser_to_map.transform.translation.y

                i,j=self.to_grid_pt(laser_pt_in_x,laser_pt_in_y)
                ip, jp, is_hit = self.bresenham(i0,j0,i, j,data[1]*self.map_scale)  
                # Updating the grid occupancy
                if not np.isnan(data[1]) and data[1] != self.range_max and self.is_inside(int(ip),int(jp)):
                    self.occupancy_grid_arr[int(ip),int(jp)] += self.l_occ- self.l_prior

        self.gridmap_p = self.log_odd_to_probability(self.occupancy_grid_arr).flatten()

        self.gridmap_int8 = (self.gridmap_p*100).astype(dtype=np.int8)
        self.gridmap_int8=self.gridmap_int8.tolist()

        # Publish map
        map_msg = OccupancyGrid()

        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = int(self.map_width/ self.map_resolution)
        map_msg.info.height = int(self.map_height / self.map_resolution)
        map_msg.info.origin.position.x = 0.0
        map_msg.info.origin.position.y = 0.0

        map_msg.data = self.gridmap_int8
        map_msg.header.stamp = msg.header.stamp
        self.pub_occ_grid.publish(map_msg)
              
    def odom_callback(self, msg):

        yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        #Storing robot pose as [pose_x,pose_y, yaw_angle]
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]


def main(args=None):
    rclpy.init(args=args)
    sub = grid_map("Occ_grid_mapper")
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
