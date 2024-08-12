import numpy as np
import cv2
from heapq import heappop, heappush
import matplotlib.pyplot as plt
from typing import List, Tuple
import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Header
import tf_transformations
from std_msgs.msg import String
import math
from nav_msgs.msg import Odometry
import yaml


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.full_load(f)
        return data


def load_pgm(pgm_file):
    with open(pgm_file, 'rb') as pgmf:
        img = plt.imread(pgmf)

        # plt.imshow(np.array(img))
        # plt.show()

        return np.array(img)


def visualize_path(grid, path, output_image_path):
    path_image = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    for point in path:
        path_image[point[0], point[1]] = (0, 50, 255)  # (B,G,R) orange
    cv2.imwrite(output_image_path, path_image)


class AStarNode:
    def __init__(self, state: Tuple, g: int, f: int, parent=None):
        self.state = state
        self.g = g  # Cost to reach this state
        self.f = f  # Estimated total cost. (g + h) for A* and h for best first search
        self.parent = parent  # Parent node

    def __lt__(self, other):
        return self.f < other.f


class AStarPathPlanner(Node):
    def __init__(self, node_name: str, pgm_file, map_scale, start, goal, min_threshold, output_image_path, grid_pivot):
        super().__init__(node_name)
        self.start_pose = start
        self.goal_pose_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.pgm_file_path = pgm_file
        self.goal_pose = goal
        self.min_threshold = min_threshold
        self.output_image_path = output_image_path
        self.astar_path = []
        self.map_scale = map_scale
        self.grid_pivot = grid_pivot
        self.grid = load_pgm(self.pgm_file_path)

    def goal_callback(self, data):

        self.get_logger().info(f'Grid Shape:{self.grid.shape}, Grid Length:{len(self.grid)}')

        pose_stamp = data.pose.position

        self.goal_pose = self.rescale_pose_to_mapsize(pose_stamp, self.grid)
        self.get_logger().info(f'Goal_pose in grid:{self.goal_pose}')
        self.get_logger().info(f"Getting goal pose: {pose_stamp}")

        self.plan(self.grid)

    def odom_callback(self, msg):

        self.curr_pose = msg.pose.pose.position
        self.start_pose = self.rescale_pose_to_mapsize(self.curr_pose, self.grid)

    def rescale_pose_to_mapsize(self, pose, grid):
        return (math.floor(len(grid) - ((pose.y - self.grid_pivot[1]) * self.map_scale)),
                math.floor((pose.x - self.grid_pivot[0]) * self.map_scale))

    def heuristic(self, current, goal):
        # manhattan distance
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def check_threshold(self, grid, new_position):
        len_x = len(grid)
        len_y = len(grid[:][0])

        for i in range(self.min_threshold + 1):
            if new_position[0] + i > len_x:
                return False
            if new_position[1] + i > len_y:
                return False
            if new_position[0] - i < 0:
                return False
            if new_position[1] - i < 0:
                return False

        for i in range(-self.min_threshold, self.min_threshold + 1):
            for j in range(-self.min_threshold, self.min_threshold + 1):
                if grid[new_position[0] + i][new_position[1] + j] < 254.0:
                    return False

        return True

    def get_possible_moves(self, grid, current: Tuple, min_threshold: int = 0) -> List[Tuple]:
        possible_moves = []
        neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]

        for neighbor in neighbors:
            new_position = (current[0] + neighbor[0], current[1] + neighbor[1])

            if grid[new_position[0]][new_position[1]] >= 254.0:
                if self.check_threshold(grid, new_position):
                    possible_moves.append(new_position)

        return possible_moves

    def astar(self, grid, start, goal, min_threshold) -> Tuple[List[int], int]:
        explored_set = set()
        start_node = AStarNode(start, 0, self.heuristic(start, goal))
        fringe = [start_node]

        while fringe:
            current_node = heappop(fringe)
            current_state = current_node.state

            if current_state == goal:
                # Reconstruct the solution path
                solution_path = [current_state]
                while current_node.parent:
                    current_node = current_node.parent
                    solution_path.append(current_node.state)
                solution_path.reverse()
                return solution_path, len(explored_set)

            explored_set.add(current_state)

            for new_state in self.get_possible_moves(grid, current_state, min_threshold):
                if new_state not in explored_set:
                    new_g = current_node.g + 1
                    new_f = new_g + self.heuristic(new_state, goal)
                    new_node = AStarNode(new_state, new_g, new_f, parent=current_node)
                    heappush(fringe, new_node)

        return [], len(explored_set)

    def astarpath_to_rospath(self, astar_path, grid):

        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "odom"

        for i, point in enumerate(astar_path):
            pose = PoseStamped()
            pose.header = Header()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'odom'

            # rescale points back to sim scale
            pose.pose.position.x = ((point[1]) / self.map_scale) + self.grid_pivot[0]
            pose.pose.position.y = ((len(grid) - point[0]) / self.map_scale) + self.grid_pivot[1]

            pose.pose.position.z = 0.0  # Assuming a 2D path

            if i < len(astar_path) - 1:  # Calculate orientation for all points except the last one
                next_point = astar_path[i + 1]
                dx = next_point[0] - point[0]
                dy = next_point[1] - point[1]
                yaw = math.atan2(dy, dx)
                quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)
                pose.pose.orientation.x = quaternion[0]
                pose.pose.orientation.y = quaternion[1]
                pose.pose.orientation.z = quaternion[2]
                pose.pose.orientation.w = quaternion[3]
            else:  # For the last point, use the default orientation
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0

            path.poses.append(pose)

        return path

    def plan(self, grid):
        if not self.check_threshold(grid, self.goal_pose):
            self.get_logger().info(f"Goal Pose is not valid!")
            return False

        astar_path, _ = self.astar(grid, self.start_pose, self.goal_pose, self.min_threshold)
        print(astar_path)

        if astar_path:
            self.astar_path = astar_path[::10]
            self.astar_path.append(astar_path[-1])
            path = self.astarpath_to_rospath(self.astar_path, grid)
            self.path_pub.publish(path)

            visualize_path(grid, astar_path, self.output_image_path)
            self.get_logger().info(f"A star path: {self.astar_path}")

        else:
            self.get_logger().info(f"A star path: path not found!")


def main(args=None):
    pgm_file = 'maps/closed_walls_map.pgm'
    yaml_file = 'maps/closed_walls_map.yaml'
    # pgm_file = 'src/Autonomous-Mobile-Robot/astar_pf_planner/astar_pf_planner/maps/closed_walls_map.pgm'

    yaml_data = load_yaml(yaml_file)

    output_image_path = 'path_outputs/map_A_star_path_planning.jpg'
    grid_pivot = yaml_data.get('origin')[:2]
    start = (0, 0)
    goal = (100, 100)
    min_threshold = 12
    map_scale = 1/yaml_data.get('resolution')
    rclpy.init(args=args)
    sub = AStarPathPlanner("path_planner", pgm_file, map_scale, start, goal, min_threshold, output_image_path,
                           grid_pivot)
    rclpy.spin(sub)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
