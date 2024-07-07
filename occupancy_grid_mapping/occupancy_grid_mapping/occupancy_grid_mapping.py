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

# p(x) = 1 - \frac{1}{1 + e^l(x)}
def l2p(l):
    return 1 - (1/(1+np.exp(l)))

# l(x) = log(\frac{p(x)}{1 - p(x)})
def p2l(p):
    return np.log(p/(1-p))


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

        self.sensor_model_l_occ = p2l(self.p_occ)
        self.sensor_model_l_free = p2l(self.p_free)
        self.sensor_model_l_prior = p2l(self.p_prior)

        self.env_data=[]
        self.curr_pose_gridpt=[0,0]

        #Creating Occupancy grid array
        #self.occupancy_grid_arr=np.ones(((self.map_height/self.map_resolution),(self.map_height/self.map_resolution)), dtype=int)*self.p_prior
        map_rows = int(self.map_height / self.map_resolution)
        map_cols = int(self.map_width/ self.map_resolution)
        self.occupancy_grid_arr = self.sensor_model_l_prior * np.ones((map_rows, map_cols))
        #self.occupancy_grid_arr=np.ones((200,200), dtype=int)*self.p_prior

        #Declaring publishers and suscribers
        self.laser_scan_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        self.curr_pose_sub = self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.pub_occ_grid=self.create_publisher(OccupancyGrid,"/map_new",10)   

        #TF_buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)  

    def extract_ranges(self, msg):
        m_array = []
        # calculate Mikel Arteta
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        for index, distance in enumerate(msg.ranges):
            m_array.append([distance, angle_min + index * angle_increment])

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
            coordinates = [radius * math.sin(angle), radius * math.cos(angle)]
            np_cart.append(coordinates)

            # test if it is working: length = sqrt(x^2 + y^2)
            # if (math.isclose(math.sqrt(coordinates[0] ** 2 + coordinates[1] ** 2), radius)):
            #    print("True")

        return np.array(np_cart)
    
    def to_grid_pt(self,x,y):
        #i = math.floor((y-self.map_center_y)/ self.map_resolution)
        #j = math.floor((x-self.map_center_x)/ self.map_resolution)
        
        i = 200-np.round(y*self.map_scale)
        j = np.round((20-x)*self.map_scale)

        #i = np.round(y*self.map_scale)
        #j = np.round((20-x)*self.map_scale)
        return i, j
    
    def is_inside (self, i, j):
        return i<self.occupancy_grid_arr.shape[0] and j<self.occupancy_grid_arr.shape[1] and i>=0 and j>=0
    
    def bresenham (self, i1, j1, d,debug=False):   # i0, j0 (starting point)

        i0, j0=self.curr_pose_gridpt[0],self.curr_pose_gridpt[0]
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
                self.occupancy_grid_arr[int(ip),int(jp)] += self.sensor_model_l_free - self.sensor_model_l_prior

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

        np_polar_array = self.extract_ranges(msg)
        np_cart_array = self.np_polar2cart(np_polar_array)

        #Getting laser_link w.r.t. map
        try:
            trans_base_laser_to_map = self.tf_buffer.lookup_transform('map', 'base_laser_front_link', rclpy.time.Time())

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("Failed to transform")
            return
        

        #self.get_logger().info(f'Transform laser frame to map{trans_base_laser_to_map}')


        #Now I am storing environment data as [angle,range,cartesian_x, cartesian_y]
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min

        for i,distance in enumerate(msg.ranges):
            self.env_data.append([angle_min + i * angle_increment,distance,np_cart_array[i][0],np_cart_array[i][1]])

        for index,data in enumerate(self.env_data):
            if not math.isinf(data[1]):
                laser_pt_in_x=data[2]+trans_base_laser_to_map.transform.translation.x
                laser_pt_in_y=data[3]+trans_base_laser_to_map.transform.translation.y

                i,j=self.to_grid_pt(laser_pt_in_x,laser_pt_in_y)
                #self.get_logger().info(f'Laser_pt in map frame at{index}: {i},{j}')   
                ip, jp, is_hit = self.bresenham(i, j,data[1]*self.map_scale)  
                #self.get_logger().info(f'Is grid {ip}{jp} hit: {is_hit}') 
                if not np.isnan(data[1]) and data[1] != self.range_max and self.is_inside(int(ip),int(jp)):
                # Hit!
                    self.occupancy_grid_arr[int(ip),int(jp)] += self.sensor_model_l_occ - self.sensor_model_l_prior

                '''
                if is_hit:
                    # Hit!
                    #self.occupancy_grid_arr[int(ip),int(jp)] += self.sensor_model_l_occ - self.sensor_model_l_prior
                    self.occupancy_grid_arr[int(ip),int(jp)]=-1
                else:
                    #self.occupancy_grid_arr[int(ip),int(jp)]=-1
                    self.occupancy_grid_arr[int(ip),int(jp)] += self.sensor_model_l_occ - self.sensor_model_l_prior
                '''
        '''

        laser_pt_in_x=self.env_data[0][2]+trans_base_laser_to_map.transform.translation.x
        laser_pt_in_y=self.env_data[0][3]+trans_base_laser_to_map.transform.translation.y
        i,j=self.to_grid_pt(laser_pt_in_x,laser_pt_in_y)
        ip, jp, is_hit = self.bresenham(i, j,self.env_data[0][1])  
        
        if is_hit:
            # Hit!
            self.occupancy_grid_arr[int(150),int(150)] += self.sensor_model_l_occ - self.sensor_model_l_prior
        else:
            self.occupancy_grid_arr[int(150),int(150)]=-1


        np.set_printoptions(threshold=sys.maxsize)
        print(f'Grid pts are: {i} {j},{ip} {jp}') 
        '''
        self.gridmap_p = l2p(self.occupancy_grid_arr).flatten()
        #self.get_logger().info(f'Grid is: {gridmap_p}') 
        #unknown_mask = (gridmap_p == self.sensor_model_p_prior)  # for setting unknown cells to -1
        self.gridmap_int8 = (self.gridmap_p*100).astype(dtype=np.int8)
        self.gridmap_int8=self.gridmap_int8.tolist()

        #gridmap_int8[unknown_mask] = -1  # for setting unknown cells to -1

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
        #rospy.loginfo_once("Published map!")

                



    def odom_callback(self, msg):
        #Getting odom center w.r.t. map
        try:
            trans = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            #self.get_logger().warn(trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("Failed to transform")
            return
        
        curr_pose_wrt_map_x=trans.transform.translation.x
        curr_pose_wrt_map_y=trans.transform.translation.y

        i_curr,j_curr=self.to_grid_pt(curr_pose_wrt_map_x,curr_pose_wrt_map_y)
        self.curr_pose_gridpt=[i_curr,j_curr]

        #self.get_logger().info(f'Curr pose wrt map{curr_pose_wrt_map_x},{curr_pose_wrt_map_y}')






def main(args=None):
    rclpy.init(args=args)
    sub = grid_map("Occ_grid_mapper")
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
