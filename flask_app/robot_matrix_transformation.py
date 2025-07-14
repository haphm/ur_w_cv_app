import os
import sys
import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from scipy.spatial.transform import Rotation as R
import time
import robotiq_gripper

ROBOT_IP = "192.168.50.23"
ROBOT_PORT = 30004

SPEED = 0.1
ACCELERATION = 0.1

initial_pose = [-0.052120238377700065, -0.2760688724179189, 0.40696261118541904, -0.03782026574893962, 2.5915557209775604, -1.71312026416318]

pick_pose = [-0.13878618032424306, -0.8131839750148216, 0.3059486631405733, -0.04854812125397555, 2.180853477135878, -2.2203822783851366]

place_pose = [-0.06485233894868425, -0.641655439538598, -0.10081181242741, -0.03313596432691149, -3.088821948569065, 0.10733373101289995]

place_approach_pose = [-0.06488994697924241, -0.6416594993545568, 0.06799634250921342, -0.03329855180851168, -3.088776671577564, 0.10734188057096894]

place_pre_approach_pose = [-0.06483457378766654, -0.6416708746173843, 0.06796595880392561, 0.016856204624654196, 2.2033355950485167, -2.2246352752883305]

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

def picking_process(position):
    rtde_control = RTDEControlInterface(ROBOT_IP)
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)
    gripper_control(0)
    rtde_control.moveL(initial_pose, SPEED, ACCELERATION)

    try:
        current_pose = rtde_receive.getActualTCPPose()
        print(f"Current Pose:\n{current_pose}")

        target_pose = pick_pose.copy()
        target_pose[0] = position[0] * 0.001 - 0.065
        target_pose[1] = position[1] * 0.001 - 0.06
        target_pose[2] = position[2] * 0.001 + 0.4

        print(f"Target Pose: {target_pose}")
        success = rtde_control.moveL(target_pose, SPEED, ACCELERATION)
        gripper_control(1)

        if success:
            print(f"Moved to position:\n{target_pose}")
            time.sleep(1)
        else:
            print("Failed to move to position")
        exit_pose = target_pose
        exit_pose[2] += 0.01
        rtde_control.moveL(exit_pose, SPEED, ACCELERATION)
        exit_pose[1] += 0.3
        rtde_control.moveL(exit_pose, SPEED, ACCELERATION)

        rtde_control.moveL(place_pre_approach_pose, SPEED, ACCELERATION)
        rtde_control.moveL(place_approach_pose, SPEED, ACCELERATION)
        rtde_control.moveL(place_pose, SPEED, ACCELERATION)

        gripper_control(0)
        time.sleep(2)

        rtde_control.moveL(place_pre_approach_pose, SPEED, ACCELERATION)

    finally:
        rtde_control.moveL(initial_pose, SPEED, ACCELERATION)
        rtde_control.stopScript()
        print("Disconnected from robot")

def gripper_control(status):
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ROBOT_IP, 63352)
    if status==0:
        gripper.move_and_wait_for_pos(0, 255, 255)  # Open gripper
    elif status==1:
        gripper.move_and_wait_for_pos(255, 255, 255)  # Close gripper

def _main():
    # Reading coordinates
    file_name = "test/xyz_coordinate.txt"
    if not os.path.isfile(file_name):
        print(f"File {file_name} does not exits. Program stopped.")
        sys.exit()
    if os.stat(file_name).st_size == 0:
        print(f"File {file_name} is empty. Program stopped.")
        sys.exit()

    # Initializing gripper
    print("Connecting to gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ROBOT_IP, 63352)
    gripper.activate()

    # Picking program
    if sys.argv[1] == "all":
        # Pick all
        with open(file_name, "r") as f:
            for line in f:
                l = line.strip().split()
                point_in_camera_frame = np.array(list(map(float, l[:4])))
                print(f"Object coordinate reference camera {point_in_camera_frame}")
                robot_coordinate_to_move = transform_point_to_camera(point_in_camera_frame)
                picking_process(robot_coordinate_to_move[0:3])
    else:
        # Pick by color
        with open(file_name, "r") as f:
            lines = f.readlines()
            color_list = []
            for line in lines:
                color_list.append(line.strip().split()[-1])
            color_to_pick = sys.argv[1]
            for line in lines:
                l = line.strip().split()
                if l[4] == color_to_pick:
                    point_in_camera_frame = np.array(list(map(float, l[:4])))
                    print(f"Object coordinate reference camera {point_in_camera_frame}")
                    robot_coordinate_to_move = transform_point_to_camera(point_in_camera_frame)
                    picking_process(robot_coordinate_to_move[0:3])
                else:
                    continue

    print("Program stopped.")

if __name__ == "__main__":
    _main()