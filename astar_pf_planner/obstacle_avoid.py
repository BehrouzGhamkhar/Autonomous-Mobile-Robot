import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
import numpy as np
import tf2_ros
import tf_transformations
from std_msgs.msg import String
from tf2_geometry_msgs import PointStamped
import time

class PathFollowingNode(Node):
    def __init__(self):
        super().__init__('path_following_node')
        self.scan_subscriber = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.slam_pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, 'pose', self.slam_pose_callback, 10)
        self.progress_result_pub = self.create_publisher(String, "/progress_result", 10)
        self.path_subscriber = self.create_subscription(Path, 'path', self.path_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Repulsive and attractive coefficients
        self.k_rep = 0.3  # Repulsive coefficient
        self.k_att = 1.0  # Attractive coefficient
        self.min_obs_dist = 0.5 # Minimum distance from obstacle

        self.goal_threshold = 0.5

        # Max velocities
        self.MAX_LINEAR_VELOCITY = 0.3  # Maximum linear velocity
        self.MAX_ANGULAR_VELOCITY = 0.5  # Maximum angular velocity

        self.waypoints = []  # List to store waypoints
        self.current_waypoint_index = 0  # Index of the current waypoint
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}  # Placeholder for the current pose
        self.last_position = None  # Store the robot's last known position
        self.stuck_time_threshold = 2.0  # Time in seconds before we consider the robot stuck
        self.last_movement_time = 0  # Timestamp of when the robot last moved


    def path_callback(self, msg):
        self.waypoints = [pose.pose for pose in msg.poses]
        self.current_waypoint_index = 0  # Start from the first waypoint
        self.get_logger().info(f'Received path with {len(self.waypoints)} waypoints.')

    def slam_pose_callback(self, msg):
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        euler = tf_transformations.euler_from_quaternion(quaternion)
        self.current_pose['theta'] = euler[2]  # Yaw


    def scan_callback(self, msg):
        
        if self.current_pose is None or not self.waypoints:
            return  # Wait until we have the current pose and waypoints

        # Get the current target waypoint
        if self.current_waypoint_index < len(self.waypoints):
            current_waypoint = self.waypoints[self.current_waypoint_index]
            # Transform waypoint position to base_link frame
            try:
                transform = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
                waypoint_point = PointStamped()
                waypoint_point.header.frame_id = 'map'
                waypoint_point.point.x = current_waypoint.position.x
                waypoint_point.point.y = current_waypoint.position.y
                waypoint_point.point.z = 0.0
                transformed_waypoint = self.tf_buffer.transform(waypoint_point, 'base_link')
                waypoint_base_link = np.array([transformed_waypoint.point.x, transformed_waypoint.point.y])
            except Exception as e:
                self.get_logger().warn(f'Failed to transform waypoint to base_link frame: {e}')
                return

            # Calculate angle to waypoint
            angle_to_waypoint = np.arctan2(waypoint_base_link[1], waypoint_base_link[0])

            # Check if the robot is stuck
            self.check_robile_stuck()

            # If the waypoint is more than 90 degrees away from the front of the robot, rotate first
            if abs(angle_to_waypoint) > np.pi / 4:
                self.rotate_towards_goal(angle_to_waypoint)
            else:
                obstacle_forces = self.calculate_repulsive_forces(msg)
                attractive_force = self.calculate_attractive_force(waypoint_base_link)
                total_force = attractive_force + obstacle_forces

                self.get_logger().info(
                    f'Attractive force: {attractive_force}, Repulsive force: {obstacle_forces}, Total force: {total_force}')
                self.move_robot(total_force, waypoint_base_link)

    def check_robile_stuck(self):

        current_position = np.array([self.current_pose['x'] , self.current_pose['y'] ])

        if self.last_position is not None:
            distance_moved = np.linalg.norm(current_position - self.last_position)
            time_since_last_move = time.time() - self.last_movement_time

            # Check if the robot has been stuck for more than the threshold
            if distance_moved < 0.02 and time_since_last_move > self.stuck_time_threshold:  # 0.05 is a small threshold
                self.get_logger().warn(f'Robot is stuck! Not moved for {time_since_last_move:.2f} seconds.')
                self.handle_getting_stuck()
            else:
                # Reset the last movement time if the robot has moved
                if distance_moved >= 0.05:
                    self.last_movement_time = time.time()

        # Update the last known position
        self.last_position = current_position


    def rotate_towards_goal(self, angle_to_goal):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = np.clip(angle_to_goal, -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)
        self.publish_cmd_vel(msg)

    def calculate_repulsive_forces(self, scan):
        forces = np.zeros(2)
        for i, range in enumerate(scan.ranges):
            if np.isinf(range) or range >= scan.range_max or range > self.min_obs_dist or range < scan.range_min or np.isnan(range):  # Only consider obstacles within 1m
                continue  # Skip if the range is infinite, beyond the maximum range, or beyond 1m

            angle = scan.angle_min + i * scan.angle_increment
            force_direction = np.array([np.cos(angle), np.sin(angle)])
            force_magnitude = self.k_rep * (1.0 / range - 1.0 / scan.range_max) / (range ** 2)
            forces -= force_magnitude * force_direction

        return forces

    def calculate_attractive_force(self, waypoint_base_link):
        direction_to_goal = waypoint_base_link
        distance_to_goal = np.linalg.norm(direction_to_goal)

        force_magnitude = self.k_att * distance_to_goal
        force_direction = direction_to_goal / (distance_to_goal + 1e-6)  # Avoid division by zero
        return force_magnitude * force_direction

    def move_robot(self, force, waypoint_base_link):
        msg = Twist()
        force_magnitude = np.linalg.norm(force)
        distance_to_goal = np.linalg.norm(waypoint_base_link)

        if (len(self.waypoints) - 1) == (self.current_waypoint_index):
            self.goal_threshold = 0.1
        else:
            self.goal_threshold = 0.5

        if distance_to_goal < self.goal_threshold:
            # Stop the robot if it is very close to the goal
            msg.linear.x = 0.0
            msg.angular.z = 0.0

            self.get_logger().info(f'Reached waypoint {self.current_waypoint_index} / {len(self.waypoints)}')
            self.current_waypoint_index += 1  # Move to the next waypoint

        else:
            msg.linear.x = np.clip((force[0] + force[1]), 0.0, self.MAX_LINEAR_VELOCITY) #min(distance_to_goal, self.MAX_LINEAR_VELOCITY)
            msg.angular.z = np.clip(np.arctan2(force[1], force[0]), -self.MAX_ANGULAR_VELOCITY,
                                    self.MAX_ANGULAR_VELOCITY)

        self.publish_cmd_vel(msg)


    def publish_cmd_vel(self, msg):
       
        self.get_logger().info(f'Moving with linear x: {msg.linear.x} angular z: {msg.angular.z}')
        self.cmd_publisher.publish(msg)


    def handle_getting_stuck(self):
        self.last_movement_time = time.time()
        self.publish_progres_result()
        

    def publish_progres_result(self, result: bool):
        result_msg = String()
        result_msg.data = str(result)
        self.progress_result_pub.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PathFollowingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()