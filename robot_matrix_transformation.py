import argparse
import zivid
import numpy as np
from numpy.ma.testutils import approx
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
from scipy.spatial.transform import Rotation as R
from save_load_matrix import load_and_assert_affine_matrix

ROBOT_IP = "192.168.50.23"
ROBOT_PORT = 30004

SPEED = 0.1
ACCELERATION = 0.1

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

def transform_point_to_camera(P_camera, T_camera_to_flange, T_tcp_to_flange):
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)

    # Get the current pose of the robot
    tcp_pose = rtde_receive.getActualTCPPose() #[-0.046898353029691586, -0.4316070248077163, 0.4008185198236556, -0.037900070432067556, 2.5929860963701654, -1.7110011727817565]
    T_base_to_tcp = pose_to_homogeneous_matrix(tcp_pose)

    # Compute flange pose using TCP -> flange transformation
    T_tcp_to_flange_inv = np.linalg.inv(T_tcp_to_flange)
    T_base_to_flange = T_base_to_tcp @ T_tcp_to_flange_inv

    # Invert flange -> base
    T_flange_to_base_inv = np.linalg.inv(T_camera_to_flange)
    T_base_to_camera = T_base_to_flange @ T_flange_to_base_inv

    object_h = np.append(P_camera, 1.0)
    point_in_camera = T_base_to_camera @ object_h

    return point_in_camera[:3]

def move_robot_to_position(position):
    rtde_control = RTDEControlInterface(ROBOT_IP)
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)

    try:
        current_pose = rtde_receive.getActualTCPPose()
        print(f"Current Pose: {current_pose}")

        target_pose = current_pose.copy()
        target_pose[0] = position[0] * 0.001 - 0.055
        target_pose[1] = position[1] * 0.001 - 0
        target_pose[2] = position[2] * 0.001 + 0.5

        print(f"Target Pose: {target_pose}")
        input("Press Enter to continue...")
        success = rtde_control.moveL(target_pose, SPEED, ACCELERATION)

        if success:
            print(f"Moved to position: {target_pose}")
        else:
            print("Failed to move to position")

    finally:
        rtde_control.stopScript()
        print("Disconnected from robot")


def main():
    """

    tool_base_to_tool_tip_transform = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 40],
                                                [0, 0, 0, 1]])

    flange_to_tcp_transform = tool_base_to_tool_tip_transform

    rtde_receive = RTDEReceiveInterface(ROBOT_IP)

    # Get the current pose of the robot
    tcp_pose = rtde_receive.getActualTCPPose()  # [-0.046898353029691586, -0.4316070248077163, 0.4008185198236556, -0.037900070432067556, 2.5929860963701654, -1.7110011727817565]
    capture_pose = pose_to_homogeneous_matrix(tcp_pose)
    print(capture_pose)

    camera_to_calibration_object_transform = np.array([[1, 0, 0, -103],
                                                        [0, 1, 0, -119],
                                                        [0, 0, 1, 555],
                                                        [0, 0, 0, 1]])      # -103.97682189941406 -119.36602783203125 555.1683959960938

    base_to_flange_transform = capture_pose.T
    flange_to_camera_transform = np.array([ [0.9987439, -0.04133826, 0.02831647, -37.62662],
                                                        [0.04059724, 0.9988304, 0.0262629, -93.9603],
                                                        [-0.02936901, -0.02508034, 0.9992539, -96.43546],
                                                        [0, 0, 0, 1]])

    base_to_calibration_object_transform = base_to_flange_transform @ flange_to_camera_transform @ camera_to_calibration_object_transform
    print(base_to_calibration_object_transform)

    touch_pose = base_to_calibration_object_transform @ np.linalg.inv(flange_to_tcp_transform)

    touch_pose_offset = np.identity(4)
    touch_pose_offset[2, 3] = 0
    approach_pose = touch_pose @ touch_pose_offset

    translation = approach_pose[0:3, 3] * 0.001
    rotation_matrix = approach_pose[0:3, 0:3]
    rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()

    pose = np.concatenate((translation, rotation_vector))
    print(f"Pose: {pose}")


    XYZ value at pixel (1334, 200): 131.99740600585938 -156.32672119140625 683.0321655273438
    XYZ value at pixel (702, 188): -113.58340454101562 -162.91226196289062 692.2449340820312
    XYZ value at pixel (1224, 960): 108.98149108886719 167.17172241210938 836.2149047851562
    XYZ value at pixel (968, 951): -11.484528541564941 162.7566375732422 836.2740478515625
    XYZ value at pixel (763, 949): -107.38746643066406 161.2161102294922 832.4418334960938
    XYZ value at pixel (1023, 198): 11.73599624633789 -156.30308532714844 681.6798095703125

    """
    point_in_camera_frame = np.array(
        [
            -107,
            161,
            832,
            1,
        ]
    )

    eye_in_hand_transform = np.array([[0.9984471, -0.03322203, 0.04471851, -52.14058],
                                        [0.03203303, 0.9991208, 0.02704765, -93.1971],
                                        [-0.04557777, -0.02557317, 0.9986334, 54.81567],
                                        [0, 0, 0, 1]])

    rtde_receive = RTDEReceiveInterface(ROBOT_IP)

    # Get the current pose of the robot
    tcp_pose = rtde_receive.getActualTCPPose()
    print(f"tcp_pose {tcp_pose}")
    capture_pose = pose_to_homogeneous_matrix(tcp_pose)
    print(f"capture_pose {capture_pose}")
    robot_transform = capture_pose

    print(
        "Reading camera pose in flange (end-effector) reference frame (result of eye-in-hand calibration)"
    )
    flange_to_camera_transform = eye_in_hand_transform

    print("Reading flange (end-effector) pose in robot base reference frame")
    base_to_flange_transform = robot_transform

    print("Computing camera pose in robot base reference frame")
    base_to_camera_transform = np.matmul(base_to_flange_transform, flange_to_camera_transform)
    print(base_to_camera_transform)



    print(f"Point coordinates in camera reference frame: {point_in_camera_frame[0:3]}")

    print("Transforming (picking) point from camera to robot base reference frame")
    point_in_base_frame = np.matmul(base_to_camera_transform, point_in_camera_frame)
    print(point_in_base_frame)


    print(f"Point coordinates in robot base reference frame: {point_in_base_frame[0:3]}")

    input("Press Enter to continue...")
    move_robot_to_position(point_in_base_frame)


if __name__ == "__main__":
    main()