import numpy as np
from heapq import heappop, heappush
from typing import List, Tuple
import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Header
import tf_transformations
from std_msgs.msg import String
import math
from nav_msgs.msg import Odometry

from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from modules.timeout_module import *


class AStarNode:
    def __init__(self, state: Tuple, g: int, f: int, parent=None):
        self.state = state
        self.g = g  # Cost to reach this state
        self.f = f  # Estimated total cost. (g + h) for A* and h for best first search
        self.parent = parent  # Parent node

    def __lt__(self, other):
        return self.f < other.f


class AStarPathPlanner(Node):
    def __init__(self, node_name: str, start, goal, goal_threshold, min_threshold, grid_pivot):
        super().__init__(node_name)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.start_pose = start

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.goal_pose_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.slam_pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, 'pose', self.slam_pose_callback, 10)
        self.astar_result_pub = self.create_publisher(String, "/astar_result", 10)

        self.curr_pose = None
        self.goal = None
        self.goal_pose = goal
        self.goal_orientation = None
        self.goal_threshold = goal_threshold
        self.min_threshold = min_threshold
        self.map_scale = 20
        self.occupancy_origin = grid_pivot
        self.astar_path = []
        self.occupancy_grid = []
        self.grid_width = 0
        self.grid_height = 0

    def map_callback(self, msg):

        self.get_logger().info('Received map data')

        # Store the occupancy grid data in a 2D array
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        data = msg.data

        self.occupancy_grid = np.array([[0 for _ in range(self.grid_height)] for _ in range(self.grid_width)])

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                self.occupancy_grid[j][i] = data[i * self.grid_width + j]

        self.occupancy_grid = np.rot90(self.occupancy_grid, 2)

        self.occupancy_origin = [msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z]
        self.map_scale = 1 / round(msg.info.resolution, 4)

        # Log some information for debugging
        self.get_logger().info(f'Map size: {self.grid_width}x{self.grid_height}')
        self.get_logger().info(f'Origin: {self.occupancy_origin}')
        self.get_logger().info(f'Resolution: {self.map_scale}')

    def goal_callback(self, data):

        pose_stamp = data.pose.position
        self.goal = data
        self.goal_orientation = data.pose.orientation
        self.goal_pose = self.rescale_pose_to_mapsize(pose_stamp, self.occupancy_grid)
        self.get_logger().info(f'Goal_pose in grid:{self.goal_pose}')
        self.get_logger().info(f"Getting goal pose: {pose_stamp}")

        self.plan(self.occupancy_grid)


    def slam_pose_callback(self, msg):
        self.curr_pose = msg.pose.pose.position
        self.start_pose = self.rescale_pose_to_mapsize(self.curr_pose, self.occupancy_grid)

    def rescale_pose_to_mapsize(self, pose, grid):
        return (math.floor(self.grid_width - ((pose.x - self.occupancy_origin[0]) * self.map_scale)),
                math.floor(self.grid_height - ((pose.y - self.occupancy_origin[1]) * self.map_scale)))

    def heuristic(self, current, goal):
        # Euclidian distance
        return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

    def check_threshold(self, grid, new_position):
        if new_position[0] + self.min_threshold >= self.grid_width:
            return False
        if new_position[1] + self.min_threshold >= self.grid_height:
            return False
        if new_position[0] - self.min_threshold < 0:
            return False
        if new_position[1] - self.min_threshold < 0:
            return False

        for i in range(-self.min_threshold, self.min_threshold + 1, self.min_threshold):
            for j in range(-self.min_threshold, self.min_threshold + 1, self.min_threshold):

                if grid[new_position[0] + i][new_position[1] + j]  > 0:
                    return False

        return True

    def get_possible_moves(self, grid, current: Tuple, min_threshold: int = 0) -> List[Tuple]:
        possible_moves = []
        neighbors = [(i, j) for i in range(-2, 4, 2) for j in range(-2, 4, 2)]

        for neighbor in neighbors:
            new_position = (current[0] + neighbor[0], current[1] + neighbor[1])
            if self.check_valid_position(grid, new_position):
                possible_moves.append(new_position)

        return possible_moves

    def reached_goal(self, current_state):
        distance = math.sqrt((current_state[0] - self.goal_pose[0]) ** 2 + (current_state[1] - self.goal_pose[1]) ** 2)
        return distance <= self.goal_threshold

    def astar(self, grid, start, goal, min_threshold) -> Tuple[List[int], int]:
        explored_set = set()
        start_node = AStarNode(start, 0, self.heuristic(start, goal))
        fringe = [start_node]

        while fringe:
            current_node = heappop(fringe)
            current_state = current_node.state

            if self.reached_goal(current_state):
                # Reconstruct the solution path
                solution_path = [current_state]
                while current_node.parent:
                    current_node = current_node.parent
                    solution_path.append(current_node.state)
                solution_path.reverse()
                return solution_path

            explored_set.add(current_state)

            for new_state in self.get_possible_moves(grid, current_state, min_threshold):
                if new_state not in explored_set:
                    new_g = current_node.g + 1
                    new_f = new_g + self.heuristic(new_state, goal)
                    new_node = AStarNode(new_state, new_g, new_f, parent=current_node)
                    heappush(fringe, new_node)

        return []


    def astarpath_to_rospath(self, grid, astar_path = None):

        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"

        if not astar_path:
            path.poses.append(self.goal)
            print("path is: ", [x.pose.position for x in path.poses])
            return path

        for i, point in enumerate(astar_path):
            pose = PoseStamped()
            pose.header = Header()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'

            # rescale points back to sim scale
            pose.pose.position.x = ((len(grid) - point[0]) / self.map_scale) + self.occupancy_origin[0]
            pose.pose.position.y = ((len(grid[:][0]) - point[1]) / self.map_scale) + self.occupancy_origin[1]

            pose.pose.position.z = 0.0  # Assuming a 2D path

            if i < len(astar_path) - 1:  # Calculate orientation for all points except the last one
                next_point = astar_path[i + 1]
                dx = next_point[0] - point[0]
                dy = next_point[1] - point[1]
                yaw = math.atan2(dy, dx)
                quaternion = tf_transformations.quaternion_from_euler(0, 0, -yaw)
                pose.pose.orientation.x = quaternion[0]
                pose.pose.orientation.y = quaternion[1]
                pose.pose.orientation.z = quaternion[2]
                pose.pose.orientation.w = quaternion[3]

            else:  # For the last point, use the default orientation
                pose.pose.orientation.z = self.goal_orientation.z
                pose.pose.orientation.w = self.goal_orientation.w

            path.poses.append(pose)

        print("path is: ", [x.pose.position for x in path.poses])
        return path
    

    def check_valid_position(self, grid, pose):
        if pose[0] >= self.grid_width or pose[1] >= self.grid_height:
            return False
        if grid[pose[0]][pose[1]] != 0:
            return False
        if not self.check_threshold(grid, pose):
            return False
        return True
    

    def plan(self, grid):
        if not self.check_valid_position(grid, self.goal_pose):
            self.get_logger().info(f"Goal Pose is not valid!")
            self.publish_astar_result(False)
            return False

        timeout_duration = 2
        astar_path = run_with_timeout(timeout_duration, self.astar, grid, self.start_pose, self.goal_pose, self.min_threshold)
        
        if astar_path:
            self.astar_path = astar_path[2::5] if len(astar_path) > 2 else astar_path[::5]
            self.astar_path.append(astar_path[-1])
            path = self.astarpath_to_rospath(grid, self.astar_path)
            self.path_pub.publish(path)
            self.publish_astar_result(True)

            self.get_logger().info(f"A star path: {self.astar_path}")

        else:
            self.get_logger().info(f"A star path: path not found!")
            self.get_logger().info(f"Returning goal pose as a path!")

            # At the beginning of the launch, current pose is not updated yet.
            if not self.curr_pose:
                path = self.astarpath_to_rospath(grid)
                self.path_pub.publish(path)
                self.publish_astar_result(True)
                return
                    
            dx = self.goal.pose.position.x - self.curr_pose.x
            dy = self.goal.pose.position.y - self.curr_pose.y
            goal_distance = math.sqrt(dx**2 + dy**2) # Euclidean distance between the robot's position and the goal position

            # Did not find a path but the goal pos is close enough.
            if goal_distance < 3.0:
                path = self.astarpath_to_rospath(grid)
                self.path_pub.publish(path)
                self.publish_astar_result(True)

            else:
                self.publish_astar_result(False)


    def publish_astar_result(self, result: bool):
        result_msg = String()
        result_msg.data = str(result)
        self.astar_result_pub.publish(result_msg)



def main(args=None):
    grid_pivot = [0, 0, 0]
    start = (0, 0)
    goal = (100, 100)
    goal_threshold = 2
    min_threshold = 7
    rclpy.init(args=args)
    sub = AStarPathPlanner("path_planner", start, goal, goal_threshold, min_threshold, grid_pivot)
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
