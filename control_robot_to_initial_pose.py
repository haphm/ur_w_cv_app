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
    # input("Press enter to continue...")

if __name__ == "__main__":
    # while True:
    #     # Read current robot position
    #     read_current_position()
    #     input("Press Enter to continue...")

    # Example position to move to
    target_position = [-0.052120238377700065, -0.2760688724179189, 0.40696261118541904, -0.03782026574893962, 2.5915557209775604, -1.71312026416318]
    move_to_position(target_position)