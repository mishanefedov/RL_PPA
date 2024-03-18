import gripper_control_module
from GripperControl import reach

import time

def test():
    start = [0.0, 0.1, 0.2]
    goal = [0.0, 0.4, 0.2]
    step_size = 0.01
    gripper_closed = True

    env_dym = [(0.0, 10.31), (0.0, 10.91), (0.0, 10.31)]

    # Measuring Python implementation time
    py_start_time = time.time()
    py_path = reach(start, goal, True, env_dym, None, step_size)
    py_end_time = time.time()

    print("PYTHON PATH: ", py_path)

    obstacles_list = []
    mock_obstacle = gripper_control_module.Obstacle(
        [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], 0.0
    )
    obstacles_list.append(mock_obstacle)

    safety_margin = 0.045

    # Measuring C++ implementation time
    cpp_start_time = time.time()
    trajectory_planner = gripper_control_module.FindTrajectory(
        start,
        goal,
        obstacles_list,
        env_dym,
        step_size,
        safety_margin,
        gripper_closed
    )
    cpp_path = trajectory_planner.aStarSearch()
    cpp_end_time = time.time()

    print("CPP PATH: ", cpp_path)
    print(f"\n\nPython implementation took {py_end_time - py_start_time} seconds.")
    print(f"C++ implementation took {cpp_end_time - cpp_start_time} seconds.")

    # print("python length: ", len(py_path), ", cpp length: ", len(cpp_path))

test()
