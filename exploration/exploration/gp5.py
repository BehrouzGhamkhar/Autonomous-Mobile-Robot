#Generating goals on fringe between explored and unexplored regions

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import random
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class FrontierGoalGenerator(Node):
    def __init__(self):
        super().__init__('frontier_goal_generator')

        # Subscribers for map and odometry
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        
        # self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # Publisher for goal poses on the consistent topic "/goal_pose"
        self.goal_publisher = self.create_publisher(
            PoseStamped, 'goal_pose', 10
        )
        
        self.astar_result_sub = self.create_subscription(String, "astar_result", self.astar_result_callback, 10)
        self.slam_pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, 'pose', self.slam_pose_callback, 10)

        self.map_data = None # Stores the most recent map data received from the /map topic
        self.robot_pose = None # Robot's current position and orientation received from the /odom topic
        self.current_goal = None # Current goal pose that the robot is moving towards
        self.goal_tolerance = 1.0  # Distance tolerance in meters
        self.goal_active = False # Flag indicating if an active goal is being pursued

    # Called whenever a new OccupancyGrid message is received
    def map_callback(self, msg):
        self.map_data = msg
        self.get_logger().info('Map received.')
        
        # Generate a goal if there isn't an active one
        if not self.goal_active:
            self.generate_frontier_goal()

        # Check if the robot has reached the goal
        if self.current_goal and self.is_goal_reached():
            self.get_logger().info('Goal reached. Ready to generate a new goal...')
            self.goal_active = False # Resets the goal flag to generate a new goal

    # Called whenever a new Odometry message is received
    # def odom_callback(self, msg):
    #     self.robot_pose = msg.pose.pose
        
    #     # Generate a goal if there isn't an active one
    #     if not self.goal_active:
    #         self.generate_frontier_goal()

    #     # Check if the robot has reached the goal
    #     if self.current_goal and self.is_goal_reached():
    #         self.get_logger().info('Goal reached. Ready to generate a new goal...')
    #         self.goal_active = False # Resets the goal flag to generate a new goal

    def slam_pose_callback(self, msg):
        self.robot_pose = msg.pose.pose


    def astar_result_callback(self, result):
        
        astar_result = result.data

        if astar_result == "False":
            self.goal_active = False
            self.generate_frontier_goal()
            

    def generate_frontier_goal(self):
        if self.map_data is None:
            self.get_logger().warn('Map not received yet. Cannot generate goal.')
            return

        # if self.robot_pose is None:
        #     self.get_logger().warn('Robot pose not received yet. Cannot generate goal.')
        #     return

        # Extract map info
        map_width = self.map_data.info.width
        map_height = self.map_data.info.height
        map_resolution = self.map_data.info.resolution
        map_origin = self.map_data.info.origin.position
        
        # Convert occupancy grid to 2D array
        map_array = list(self.map_data.data)
        map_2d = [map_array[i:i + map_width] for i in range(0, len(map_array), map_width)]

        # Find frontier cells
        frontier_cells = self.find_frontier_cells(map_2d, map_width, map_height)

        if len(frontier_cells) < 50:
            self.get_logger().warn('No frontier cells found. Finishing Exploration!!  YAY!!.')
            self.get_logger().error('You have served your duty Soldier. You may Leave now')
            rclpy.shutdown()
            return

        # Randomly select a frontier cell
        goal_cell = random.choice(frontier_cells)

        # Converts the selected cell's grid coordinates to real-world coordinates
        goal_x = map_origin.x + (goal_cell[1] * map_resolution)
        goal_y = map_origin.y + (goal_cell[0] * map_resolution)

        # Random orientation for the goal
        goal_yaw = random.uniform(-math.pi, math.pi)

        # Create PoseStamped message
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "odom"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        goal_pose.pose.orientation.w = math.cos(goal_yaw / 2.0)

        # Store and publish the goal pose
        self.current_goal = goal_pose
        self.get_logger().info(f'Publishing goal: {goal_pose}')
        self.goal_publisher.publish(goal_pose)  # Publish on /goal_pose
        self.goal_active = True # Sets the goal active flag to prevent generating new goals until the current one is reached

    def find_frontier_cells(self, map_2d, map_width, map_height):
        """
        Find frontier cells that are on the boundary between explored (free) and unexplored (unknown) areas.
        """
        frontier_cells = []
        for y in range(1, map_height - 1):
            for x in range(1, map_width - 1):
                # Check if the current cell is free space
                if map_2d[y][x] == 0:  # Free space
                    # Check the 8-connected neighbors for unknown space
                    if any(map_2d[ny][nx] == -1 for nx, ny in self.get_neighbors(x, y)):
                        frontier_cells.append((y, x))
        return frontier_cells

    def get_neighbors(self, x, y):
        """
        Get the 8-connected neighbors for a given cell.
        """
        return [
            (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y),               (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
        ]

    #Checks if the robot is within goal_tolerance distance of the current goal
    def is_goal_reached(self):
        if self.robot_pose is None or self.current_goal is None:
            return False
        
        dx = self.current_goal.pose.position.x - self.robot_pose.position.x
        dy = self.current_goal.pose.position.y - self.robot_pose.position.y
        distance = math.sqrt(dx**2 + dy**2) # Euclidean distance between the robot's position and the goal position

        return distance < self.goal_tolerance

def main(args=None):
    rclpy.init(args=args)
    frontier_goal_generator = FrontierGoalGenerator()
    rclpy.spin(frontier_goal_generator)
    frontier_goal_generator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
