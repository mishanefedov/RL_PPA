from typing import Tuple
import numpy as np


def reach(current_pos, goal_pos, gripper_closed, env_dimension, obstacles=None, step_size= 0.01) -> [[float]]:
    trajectory_planner = FindTrajectory(current_pos, goal_pos,
                                        step_size=step_size, obstacles=obstacles, safety_margin=0.045,
                                        gripper_closed=gripper_closed, env_dimension=env_dimension)
    return trajectory_planner.a_star_search()


class Obstacles:
    def __init__(self, obs: dict, dt: float):
        self.obstacles = []
        # change depending on number of obstacles ["obstacle1", "obstacle2", "obstacle3"]
        for o in ["obstacle1", "obstacle2",]:
            self.obstacles.append(Obstacle(pos=obs[o][:3], size=obs[o][3:6], vel=obs[o][6], direction=[1,0,0], dt=dt))
        self.obstacles[1].vel *= -1

    def __str__(self):
        obst_str = ""
        for i, o in enumerate(self.obstacles):
            obst_str += f"o{i}: {o}\n"
        return obst_str

    def collides_at_time_step(self, pos: Tuple[float, float, float], time_step: int, safety_margin: float) -> bool:
        for i in self.obstacles:
            if i.collides_at_time_step(pos, time_step, safety_margin):
                return True
        return False


class Obstacle:
    def __init__(self, pos: Tuple[float, float, float], size: Tuple[float, float, float],
                 vel: float, direction: [int], dt: float):
        self.pos = pos
        self.size = size
        self.vel = vel  # velocity
        self.direction = direction
        self.dt = dt
        self.future_pos = self._future_obstacle_positions()

    def __str__(self):
        return "Obastcle with: pos: %s,  size: %s, vel: %s, direction: %s, dt: %s" % \
               (self.pos, self.size, self.vel, self.direction, self.dt)

    def collides_at_time_step(self, pos: Tuple[float, float, float], time_step: int, safety_margin: float) -> bool:
        f_obs_pos = self.future_pos[time_step]
        # gripper dimension
        g_x_width = 0.03
        g_y_depth = 0.01
        g_z_up = 0.0
        g_z_down = 0.0
        # xmax1 >= xmin2 and xmin1 <= xmax2
        if not ((pos[0] + g_x_width + safety_margin >= f_obs_pos[0] - self.size[0]) and
                (pos[0] - g_x_width - safety_margin <= f_obs_pos[0] + self.size[0])):
            return False
        # ymax1 >= ymin2 and ymin1 <= ymax2
        if not ((pos[1] + g_y_depth + safety_margin >= f_obs_pos[1] - self.size[1]) and
                (pos[1] - g_y_depth - safety_margin <= f_obs_pos[1] + self.size[1])):
            return False
        # zmax1 >= zmin2 and zmin1 <= zmax2
        if not (pos[2] + g_z_up + safety_margin >= f_obs_pos[2] - self.size[2] and
                pos[2] - g_z_down - safety_margin <= f_obs_pos[2] + self.size[2]):
            return False
        return True

    def _future_obstacle_positions(self) -> Tuple[float, float, float]:
        future_pos = []
        for _ in range(20):
            new_pos = [0, 0, 0]
            for i in range(3):
                new_pos[i] = self.pos[i] + (self.direction[i] * i * self.dt * self.vel)
            future_pos.append(tuple(new_pos))
        return future_pos


class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent, pos: Tuple[float, float, float]):
        self.parent = parent
        self.pos = pos
        self.depth = parent.depth + 1 if parent else 0
        self.g = 0  # G is the distance between the current node and the start node.
        self.h = 0  # is the heuristic — estimated distance from the current node to the end node.
        self.f = 0  # F is the total cost of the node.

    def __eq__(self, other) -> bool:
        for i in range(3):
            if round(self.pos[i], 3) != round(other.pos[i], 3):
                return False
        return True

    def __str__(self):
        return "Pos: %s  Parent: %s>" % (self.pos, self.parent)

    def __lt__(self, other):
        return self.f < other.f

    def index(self):
        x = round(self.pos[0], 3)
        y = round(self.pos[1], 3)
        z = round(self.pos[2], 3)
        return tuple([x, y, z])


class FindTrajectory:

    def __init__(self, start_pos, goal_position, step_size, obstacles: Obstacles,
                 safety_margin, gripper_closed,env_dimension):
        self.start_pos = start_pos
        self.goal_pos = goal_position
        self.step_size = step_size
        self.obstacles = obstacles
        self.og_safety_margin = safety_margin
        self.safety_margin = safety_margin
        self.gripper_closed = gripper_closed
        self.env_dimension = env_dimension

    def is_nearest_node_to_goal(self, node: Node) -> bool:
        return np.linalg.norm(np.asarray(node.pos) - np.asarray(self.goal_pos), axis=-1) < (self.step_size - 0.00)

    def node_is_reachable(self, node: Node) -> bool:
        for i in range(3):
            if node.pos[i] < self.env_dimension[i][0] or node.pos[i] > self.env_dimension[i][1]:
                return False
        return True

    def get_possible_adjacent_nodes(self, node: Node, time_step) -> [Node]:
        adjacent_nodes = []
        steps = [-self.step_size, 0, self.step_size]
        for i in steps:
            for j in steps:
                for k in steps:
                    if i == j == k == 0:
                        continue
                    new_node_pos = (node.pos[0] + i, node.pos[1] + j, node.pos[2] + k)
                    new_node = Node(node, new_node_pos)
                    if self.node_is_reachable(new_node) and \
                            (self.obstacles is None
                             or not self.obstacles.collides_at_time_step(new_node.pos, time_step, self.safety_margin)):
                        adjacent_nodes.append(new_node)
        return adjacent_nodes

    def heuristic(self, node: Node) -> float:
        for i in range(3):
            node.h += abs(node.pos[i] - self.goal_pos[i])
        return node.h

    def trajectory_to_actions(self, nodes: [Node]) -> [[float]]:
        optimal_trajectory = nodes
        actions = []
        while len(optimal_trajectory) != 1:
            cur = optimal_trajectory.pop(0).pos
            next_node_pos = optimal_trajectory[0].pos
            action = [(next_node_pos[0] - cur[0]) / self.step_size, (next_node_pos[1] - cur[1]) / self.step_size,
                      (next_node_pos[2] - cur[2]) / self.step_size]
            if self.gripper_closed:
                action.append(1)
            else:
                action.append(-1)
            actions.append([round(i, 5) for i in action])
        return actions

    def a_star_search(self):
        # Create start  node
        start_node = Node(None, self.start_pos)
        self.safety_margin = self.og_safety_margin
        if self.obstacles and self.obstacles.collides_at_time_step(start_node.pos, 0, self.safety_margin):
            self.safety_margin = 0.02

        # Initialize both open and closed list
        open_dic = {}
        closed_dic = {}

        # Add the start node
        open_dic[start_node.index()] = start_node

        # Loop until you find the end
        c = 0
        while len(open_dic) > 0:
            c += 1
            # Get the current node (find the best Node)
            min_key = min(open_dic, key=open_dic.get)  # get key with maximum value
            current_node = open_dic[min_key]

            # Pop current off open list, add to closed list
            open_dic.pop(min_key)
            closed_dic[min_key] = current_node

            # Found the goal or maximum depth reached
            if self.is_nearest_node_to_goal(current_node) or current_node.depth > 15:
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = current.parent
                return self.trajectory_to_actions(path[::-1])  # Return reversed path

            # Generate children
            children = self.get_possible_adjacent_nodes(current_node, time_step=current_node.depth)

            # Loop through children
            for child in children:

                # Child is on the closed list
                if child.pos in closed_dic:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + self.step_size
                child.h = self.heuristic(child)
                child.f = child.g + child.h

                if child.pos in open_dic and child > open_dic[child.pos]:
                    continue

                # Add the child to the open list
                open_dic[child.index()] = child
