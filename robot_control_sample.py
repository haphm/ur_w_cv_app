from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "192.168.50.23"
ROBOT_PORT = 30004

SPEED = 0.1
ACCELERATION = 0.1

def move_to_position(position):
    rtde_control = RTDEControlInterface(ROBOT_IP)
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)

    try:
        current_pose = rtde_receive.getActualTCPPose()
        print(f"Current Pose: {current_pose}")
        input("Press enter to continue...")

        target_pose = current_pose.copy()
        target_pose[0] = position[0]
        target_pose[1] = position[1]
        target_pose[2] = position[2]

        success = rtde_control.moveL(target_pose, SPEED, ACCELERATION)

        if success:
            print(f"Moved to position: {target_pose}")
        else:
            print("Failed to move to position")

    finally:
        rtde_control.stopScript()
        print("Disconnected from robot")

if __name__ == "__main__":
    # Example position to move to
    target_position = [-0.046, -0.704, 0.388]

    move_to_position(target_position)