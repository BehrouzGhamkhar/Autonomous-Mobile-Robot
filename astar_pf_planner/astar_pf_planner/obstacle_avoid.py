import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import tf2_ros
import tf_transformations
from tf2_geometry_msgs import PointStamped
from nav_msgs.msg import Odometry, Path
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


class PotentialFieldAvoidance(Node):
    def __init__(self):
        super().__init__('potential_field_avoidance')
        self.scan_subscriber = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, 'ground_truth_pose', self.pose_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_sub = self.create_subscription(Path, "/path", self.path_callback, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.robot_speed = 0.9
        self.MAX_LINEAR_VELOCITY = 1.0  # Maximum linear velocity
        self.MAX_ANGULAR_VELOCITY = 2.0  # Maximum angular velocity
        self.k_rep = 0.5  # Repulsive coefficient
        self.k_att = 1  # Attractive coefficient
        # self.goal = np.array([4.0, 10.0, -1.0])  # Goal position (x, y, theta) in odom frame
        # self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}  # Placeholder for the current pose
        self.current_pose = [0.0, 0.0, 0.0]
        self.path = []  # initialising path array
        self.nearest_path_point = None

        # Max velocities
        self.MAX_LINEAR_VELOCITY = 1.0  # Maximum linear velocity
        self.MAX_ANGULAR_VELOCITY = 2.0  # Maximum angular velocity

        self.goal_pose = PoseStamped()
        self.goal_pose_base_link = PoseStamped()
        self.goal_pose_received = False

    def pose_callback(self, msg):
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w
        self.current_pose[2] = 2.0 * np.arctan2(orientation_z, orientation_w)

    def path_callback(self, msg):
        self.goal_pose_received = True
        self.path = msg.poses
        print("path is: ", [x.pose.position for x in self.path])
        self.nearest_path_point = self.path[0]

    def scan_callback(self, msg):
        if self.current_pose is None:
            return  # Wait until we have the current pose

        if (len(self.path) == 0):
            # print("Path is not received")
            pass
        else:

            if not self.nearest_path_point:
                return

            self.goal_pose.pose = self.nearest_path_point.pose
            self.goal_pose_base_link.header.frame_id = 'base_link'
            try:
                trans = self.tf_buffer.lookup_transform('base_link', 'odom', rclpy.time.Time())
                self.goal_pose_base_link.pose = do_transform_pose(self.goal_pose.pose, trans)

                goal_base_link = np.array(
                    [self.goal_pose_base_link.pose.position.x, self.goal_pose_base_link.pose.position.x])

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().warn("Failed to transform goal pose to base_laser_front_ink frame")
                return


            # Transform goal position to base_link frame
            print(f"near pose: {self.nearest_path_point.pose}")
            print(f"current pose: {self.current_pose}")
            print(f"different pose: {self.current_pose[0] - self.nearest_path_point.pose.position.x} , {self.current_pose[1] - self.nearest_path_point.pose.position.y}")


            obstacle_forces = self.calculate_repulsive_forces(msg)
            attractive_force = self.calculate_attractive_force(goal_base_link)
            total_force = attractive_force + obstacle_forces

            self.get_logger().info(
                f'Attractive force: {attractive_force}, Repulsive force: {obstacle_forces}, Total force: {total_force}')

            self.move_robot(total_force, goal_base_link)

    def calculate_repulsive_forces(self, scan):
        forces = np.zeros(2)
        for i, range in enumerate(scan.ranges):
            if np.isinf(range) or range >= scan.range_max:
                continue  # Skip if the range is infinite or beyond the maximum range
            angle = scan.angle_min + i * scan.angle_increment
            force_direction = np.array([np.cos(angle), np.sin(angle)])
            force_magnitude = self.k_rep * (1.0 / range - 1.0 / scan.range_max) / (range ** 2)
            forces -= force_magnitude * force_direction

        return forces

    def calculate_attractive_force(self, goal_base_link):
        direction_to_goal = goal_base_link
        distance_to_goal = np.linalg.norm(direction_to_goal)
        if distance_to_goal < 0.5:
            # Control orientation when close to goal
            force_magnitude = 0.0

            # ?
            quarternion = (self.goal_pose_base_link.pose.orientation.x, self.goal_pose_base_link.pose.orientation.y,
                           self.goal_pose_base_link.pose.orientation.z, self.goal_pose_base_link.pose.orientation.w)
            Rz = tf_transformations.euler_from_quaternion(quarternion)[2]
            # Rx,Ry,Rz=tf_transformations.euler_from_quaternion(self.goal_pose_base_link.pose.orientation)
            theta_goal = Rz  # self.goal[2]
            current_yaw = self.current_pose[2]
            angular_error = self.normalize_angle(theta_goal - current_yaw)
            return np.array([0.0, angular_error])
        else:
            # Normal attractive force
            force_magnitude = self.k_att * distance_to_goal
            force_direction = direction_to_goal / (distance_to_goal + 1e-6)  # Avoid division by zero
            return force_magnitude * force_direction

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def move_robot(self, force, goal_base_link):
        msg = Twist()
        force_magnitude = np.linalg.norm(force)
        distance_to_goal = np.linalg.norm(goal_base_link)

        if distance_to_goal < 0.05:
            # Stop the robot if it is very close to the goal
            print("stop robile")
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            current_index = self.path.index(self.nearest_path_point)
            if current_index < len(self.path) - 1:
                self.nearest_path_point = self.path[current_index + 1]

        elif force_magnitude < 0.1:  # When close to goal, just rotate to desired orientation
            msg.linear.x = 0.0
            msg.angular.z = np.clip(force[1], -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)
        else:
            msg.linear.x = min(self.MAX_LINEAR_VELOCITY, distance_to_goal)
            msg.angular.z = np.clip(np.arctan2(force[1], force[0]), -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)  # Clipping angular velocity

        self.get_logger().info(f'Moving with linear x: {msg.linear.x} angular z: {msg.angular.z}')
        self.cmd_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
