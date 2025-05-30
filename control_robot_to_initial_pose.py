from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "192.168.50.23"
ROBOT_PORT = 30004

SPEED = 0.1
ACCELERATION = 0.1

def move_to_position(position):
    rtde_control = RTDEControlInterface(ROBOT_IP)
    try:
        success = rtde_control.moveL(position, SPEED, ACCELERATION)
        if success:
            print(f"Moved to initial position: {position}")
        else:
            print("Failed to move to initial position")

    finally:
        rtde_control.stopScript()
        print("Disconnected from robot")

def read_current_position():
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)
    current_pose = rtde_receive.getActualTCPPose()
    print(f"Current Pose: {current_pose}")
    input("Press enter to continue...")

if __name__ == "__main__":
    # # Read current robot position
    # read_current_position()

    # Example position to move to
    target_position = [-0.05210053790641115, -0.27605043364346304, 0.4070199063449296, -0.039457233465660886, 2.5466319129173374, -1.7801884436371929]
    move_to_position(target_position)