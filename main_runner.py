import subprocess

def run_program(path):
    try:
        result = subprocess.run(["python3", path], check=True, text=True, capture_output=True)
        print(f"Output of {path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {path}: \n{e.stderr}")

if __name__=="__main__":
    run_program("take_image_for_detect.py")
    run_program("robot_matrix_transformation.py")
