from rtde_control import RTDEControlInterface

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

if __name__ == "__main__":
    # Example position to move to
    target_position = [-0.052069181437518125, -0.27607119937006347, 0.4008986149866134, -0.03799443728613493, 2.5927110603628365, -1.7112709281720009]
    move_to_position(target_position)