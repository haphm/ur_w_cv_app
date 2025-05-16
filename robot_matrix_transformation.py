import os
import sys
import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from scipy.spatial.transform import Rotation as R
import time

ROBOT_IP = "192.168.50.23"
ROBOT_PORT = 30004

SPEED = 0.1
ACCELERATION = 0.1

initial_pose = [-0.05206830786462389, -0.27606108393635814, 0.4008758870663998, -0.03809314044235728, 2.5927831777220005, -1.7110561426440825]

pick_pose = [-0.13878618032424306, -0.8131839750148216, 0.3059486631405733, -0.04854812125397555, 2.180853477135878, -2.2203822783851366]

eye_in_hand_transform = np.array([[0.9984471, -0.033222, 0.0447185, -52.1405],
                                  [0.0320330, 0.9991208, 0.0270476, -93.1971],
                                  [-0.045577, -0.025573, 0.9986334, 54.81567],
                                  [0, 0, 0, 1]])

def pose_to_homogeneous_matrix(pose):
    # Convert a pose (x, y, z, rx, ry, rz) to a homogeneous transformation matrix.
    translation = np.array(pose[:3])
    rotation_vector = np.array(pose[3:6])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()

    # Create the homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation

    return T

def transform_point_to_camera(point_in_camera_frame):
    # Get the current pose of the robot
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)
    robot_transform = pose_to_homogeneous_matrix(rtde_receive.getActualTCPPose())

    flange_to_camera_transform = eye_in_hand_transform
    print(f"Camera pose in flange reference frame (result of eye-in-hand calibration): \n {flange_to_camera_transform}")

    base_to_flange_transform = robot_transform
    print(f"Flange (end-effector) pose in robot base reference frame: \n {base_to_flange_transform}")

    print("Computing camera pose in robot base reference frame")
    base_to_camera_transform = np.matmul(base_to_flange_transform, flange_to_camera_transform)
    print(f"Camera pose in robot base reference frame: \n {base_to_camera_transform}")

    point_in_base_frame = np.matmul(base_to_camera_transform, point_in_camera_frame)
    print(f"Transforming (picking) point from camera to robot base reference frame: \n {point_in_base_frame}")

    return point_in_base_frame[:3]

def move_robot_to_position(position):
    rtde_control = RTDEControlInterface(ROBOT_IP)
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)
    rtde_control.moveL(initial_pose, SPEED, ACCELERATION)

    try:
        current_pose = rtde_receive.getActualTCPPose()
        print(f"Current Pose:\n{current_pose}")

        target_pose = pick_pose.copy()
        target_pose[0] = position[0] * 0.001 - 0.055
        target_pose[1] = position[1] * 0.001 - 0.05
        target_pose[2] = position[2] * 0.001 + 0.4

        print(f"Target Pose: {target_pose}")
        success = rtde_control.moveL(target_pose, SPEED, ACCELERATION)

        if success:
            print(f"Moved to position:\n{target_pose}")
            time.sleep(2)
        else:
            print("Failed to move to position")

    finally:
        rtde_control.moveL(initial_pose, SPEED, ACCELERATION)
        rtde_control.stopScript()
        print("Disconnected from robot")


def _main():
    file_name = "test/xyz_coordinate.txt"
    if not os.path.isfile(file_name):
        print(f"File {file_name} does not exits. Program stopped.")
        sys.exit()
    if os.stat(file_name).st_size == 0:
        print(f"File {file_name} is empty. Program stopped.")
        sys.exit()

    # Pick all
    with open(file_name, "r") as f:
        for line in f:
            if 'nan' in line:
                continue
            point_in_camera_frame = np.array(list(map(float, line.split())))
            print(f"Object coordinate reference camera {point_in_camera_frame}")
            robot_coordinate_to_move = transform_point_to_camera(point_in_camera_frame)

            move_robot_to_position(robot_coordinate_to_move[0:3])

    # # Pick 1
    # with open(file_name, "r") as f:
    #     line = f.readline()
    #     point_in_camera_frame = np.array(list(map(float, line.split())))
    #     print(f"Object coordinate reference camera {point_in_camera_frame}")
    #     robot_coordinate_to_move = transform_point_to_camera(point_in_camera_frame)
    #
    #     move_robot_to_position(robot_coordinate_to_move[0:3])


if __name__ == "__main__":
    _main()

