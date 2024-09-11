import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from tf2_msgs.msg import TFMessage
import numpy as np
import math
import tf_transformations, tf2_ros, tf2_geometry_msgs
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import Odometry, Path


class Subscriber(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.goal_pose = PoseStamped()
        self.goal_pose_laser_link = PoseStamped()
        self.goal_reached = False

        self.curr_pose = PoseStamped()
        self.curr_pose_base_link = PoseStamped()
        self.goal_pose_received = False
        self.path = []

        self.nearest_path_point = None
        self.goal_index = 0


        self.env_data = []
        self.k_attraction = -0.5
        self.k_replusion = 0.5
        self.threshold_dist = 1.0
        self.MAX_LINEAR_VELOCITY = 0.5  # Maximum linear velocity
        self.MAX_ANGULAR_VELOCITY = 1.0  # Maximum angular velocity

        self.laser_scan_sub = self.create_subscription(LaserScan, "/scan", self.laserCallback, 10)
        self.goal_pose_sub = self.create_subscription(Path, "/path", self.goalPoseCallback, 10)
        # self.goal_pose_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goalPoseCallback, 10)
        self.curr_pose_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def odom_callback(self, msg):

        self.curr_pose = msg.pose.pose

        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())

            self.curr_pose_base_link.pose = do_transform_pose(self.curr_pose, trans)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("Failed to transform current pose to base_laser_front_ink frame")
            return


    def goalPoseCallback(self, msg):

        self.goal_pose_received = True

        self.path = msg.poses
        self.nearest_path_point = self.path[0]


    def calc_rep_force_x(self, data):

        force = self.k_replusion * (
                ((1 / data[1]) - (1 / self.threshold_dist)) * (0 - data[2]) / (data[1] * data[1] * data[1]))

        return force

    def calc_rep_force_y(self, data):

        force = self.k_replusion * (
                ((1 / data[1]) - (1 / self.threshold_dist)) * (0 - data[3]) / (data[1] * data[1] * data[1]))
        return force

    def calc_attractive_force_x(self):

        dx = self.curr_pose_base_link.pose.position.x - self.goal_pose_laser_link.pose.position.x
        dy = self.curr_pose_base_link.pose.position.y - self.goal_pose_laser_link.pose.position.y
        force = 0

        dist_to_goal = math.sqrt(dx ** 2 + dy ** 2)

        if dist_to_goal >= 0.2 and self.goal_pose_received == True:
            force = self.k_attraction * (dx) / dist_to_goal
        else:
            force = 0.0

        if dist_to_goal < 0.2 and self.goal_pose_received == True:
            self.goal_reached = True
            current_index = self.path.index(self.nearest_path_point)
            if current_index < len(self.path) - 1:
                self.nearest_path_point = self.path[current_index + 1]
        else:
            self.goal_reached = False
        return force

    def calc_attractive_force_y(self):
        dx = self.curr_pose_base_link.pose.position.x - self.goal_pose_laser_link.pose.position.x
        dy = self.curr_pose_base_link.pose.position.y - self.goal_pose_laser_link.pose.position.y
        force = 0

        dist_to_goal = math.sqrt(dx ** 2 + dy ** 2)

        if dist_to_goal > 0.2 and self.goal_pose_received == True:
            force = self.k_attraction * (dy) / dist_to_goal

        else:
            force = 0.0

        if dist_to_goal < 0.2 and self.goal_pose_received == True:
            self.goal_reached = True
        else:
            self.goal_reached = False

        return force

    def laserCallback(self, msg):

        # Transforming current pose from 'odom' to 'base_laser_front_link'
        self.curr_pose_base_link.header.frame_id = 'base_link'
        if self.curr_pose is None:
            # self.get_logger().info('Goal pose not received yet.')
            return

        if (len(self.path) == 0):
            #print("Path is not received")
            pass
        else:

            if not self.nearest_path_point:
                return

            self.goal_pose.pose = self.nearest_path_point.pose
            self.goal_pose_laser_link.header.frame_id = 'base_link'
            try:
                trans = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
                self.goal_pose_laser_link.pose = do_transform_pose(self.goal_pose.pose, trans)
                # self.get_logger().info(f'Goal_pt after transform is {self.goal_pose_laser_link.pose}')

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().warn("Failed to transform goal pose to base_laser_front_ink frame")
                return

        rep_force_data, attr_force = self.calculate_forces(msg)
        self.move_robile(rep_force_data, attr_force)

    def calculate_forces(self, msg):
        self.env_data.clear()
        np_polar_array = self.extract_ranges(msg)
        np_cart_array = self.np_polar2cart(np_polar_array)

        # Now I am storing environment data as [angle,range,cartesian_x, cartesian_y]
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min

        for i, distance in enumerate(msg.ranges):
            self.env_data.append([angle_min + i * angle_increment, distance, np_cart_array[i][0], np_cart_array[i][1]])

        obstacle_points = []

        rep_force_x_overall = []
        rep_force_y_overall = []
        index_start = -1

        found_start = False
        for i, data in enumerate(self.env_data):
            if math.isinf(data[1]):
                if found_start == True:
                    index_end = i
                    found_start = False
                    obstacle_points.append(self.env_data[index_start:index_end])
                pass
            else:
                if found_start == False:
                    index_start = i
                    found_start = True

        # Clustering lidar(i.e. obstacle points). Here neighbouring points are clustered into one group and repulsive force corresponding to each point is calculated
        for i in obstacle_points:
            rep_force_x = []
            rep_force_y = []

            for j in i:
                rep_force_x.append(self.calc_rep_force_x(j))
                rep_force_y.append(self.calc_rep_force_y(j))

            rep_force_x_overall.append(rep_force_x)
            rep_force_y_overall.append(rep_force_y)

        average_rep_force_x = []
        average_rep_force_y = []

        # In this algorithm, addition and subtraction of the attractive and repulsive forces is done at base_laser_link

        # Calculating average repulsive force in X direction. This average is for one cluster
        for count, data1 in enumerate(rep_force_x_overall):
            average_rep_force_x.append(sum(data1) / len(data1))

        # Calculating average repulsive force in Y direction. This average is for one cluster
        for int, data2 in enumerate(rep_force_y_overall):
            average_rep_force_y.append(sum(data2) / len(data2))

        rep_force_data = []

        for i in range(len(average_rep_force_x)):
            rep_force_data.append([average_rep_force_x[i], average_rep_force_y[i]])

        # Calculating attractive forces in
        attr_force_x = self.calc_attractive_force_x()
        attr_force_y = self.calc_attractive_force_y()

        return rep_force_data, [attr_force_x, attr_force_y]


    def move_robile(self, rep_force_data, attr_force):
        velocity_x = 0.0
        velocity_y = 0.0

        # Adding repulsive forces
        for force_x, force_y in rep_force_data:
            velocity_x = velocity_x + force_x
            velocity_y = velocity_y + force_y

        # Adding the attractive force
        if self.goal_pose_received == False or self.goal_reached == True:
            velocity_x = 0.0
            velocity_y = 0.0
        else:
            velocity_x = velocity_x + attr_force[0]
            velocity_y = velocity_y + attr_force[1]

        # # Capping the velocity
        # if (velocity_x > 1.0):
        #     velocity_x = 1.0
        # elif (velocity_x <= 0):
        #     velocity_x = 0.0

        # if (velocity_y > 0.5):
        #     velocity_y = 0.5

        # As addition and subtraction of forces is done at the base_laser_front_link, I am converting corrsponding velocities to base_link.
        # v = r * omega

        distance_baselink_to_laserfront = 0.45
        base_link_x_vel = velocity_x
        base_link_WZ_vel = velocity_y / distance_baselink_to_laserfront

        # Publishing velocity data into cmd_vel
        vel_cmd = Twist()
        vel_cmd.linear.x = np.clip(base_link_x_vel, 0.0, self.MAX_LINEAR_VELOCITY)
        vel_cmd.angular.z = np.clip(base_link_WZ_vel, -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)

        self.pub_cmd_vel.publish(vel_cmd)


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


def main(args=None):
    rclpy.init(args=args)
    sub = Subscriber("pf_planner")
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
