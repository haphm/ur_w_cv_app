# gripper_worker.py
from multiprocessing.connection import Listener
from common import MAIN_TO_GRIPPER_ADDRESS, AUTH_KEY

def main():
    listener = Listener(MAIN_TO_GRIPPER_ADDRESS, authkey=AUTH_KEY)
    print("Gripper worker: Waiting for command...")

    while True:
        conn = listener.accept()
        msg = conn.recv()
        if msg == "open":
            print("Gripper opened.")
            # Here: insert hardware control code to open gripper
        elif msg == "close":
            print("Gripper closed.")
            # Here: insert hardware control code to close gripper
        else:
            print(f"Unknown command: {msg}")
        conn.close()

if __name__ == "__main__":
    main()
