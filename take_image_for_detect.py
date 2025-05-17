import os
import shutil
import cv2
import numpy as np
import zivid
from ultralytics import YOLO


def _point_cloud_to_cv_bgr(point_cloud: zivid.PointCloud) -> np.ndarray:
    bgra = point_cloud.copy_data("bgra_srgb")
    return bgra[:, :, :3]

def _visualize_and_save_image(image: np.ndarray, image_file: str, title: str) -> None:
    # display_bgr(image, title)
    cv2.imwrite(image_file, image)

def _detect_objects(image: np.ndarray) -> None:
    # Load a model
    model = YOLO("runs/detect/train4/weights/best.pt")  # load a custom model

    # Predict with the model
    results = model(image)  # predict on an image

    # Access the results
    with open("test/result.txt", "w") as f:
        for result in results:
            xywh = result.boxes.xywh.tolist()  # center-x, center-y, width, height
            for i in range(len(xywh)):
                f.write(str(xywh[i]) + "\n")
            result.save("test/result.png")  # save the image with predictions

def _detect_color(frame, x_coor, y_coor) -> str:
    color_thresholds = [10, 20, 30, 85, 130, 160, 179]
    color_name = ['red', 'brown', 'yellow', 'green', 'blue', 'violet', 'red']
    hsv_pixel = frame[x_coor, y_coor]
    h, s, v = hsv_pixel
    print(f"h: {h}, s: {s}, v: {v}")
    if v < 50:
        return 'black'
    if s < 30 and v > 200:
        return 'white'
    for i, thresh in enumerate(color_thresholds):
        if h <= thresh:
            color = color_name[i]
            return color
    return "Undefined"


def _main() -> None:
    # Specify the directory name
    test_directory = "test"

    # Create the directory
    if os.path.isdir(f"{test_directory}"):
        shutil.rmtree(f"{test_directory}")
    os.mkdir(f"{test_directory}")

    # Connect to the camera
    app = zivid.Application()
    camera = app.connect_camera(serial_number="21469A61")

    settings = zivid.Settings(
        acquisitions=[zivid.Settings.Acquisition()],
        color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
    )

    with camera.capture_2d_3d(settings) as frame:
        data_file = f"test/detect.zdf"
        frame.save(data_file)

        # Save the 2D image
        color_image = frame.frame_2d().image_bgra_srgb()
        color_image.save(f"test/detect.jpg")

    print("Image captured.")

    with (app):
        data_file = "test/detect.zdf"
        print(f"Reading ZDF frame from file: {data_file}")
        frame = zivid.Frame(data_file)
        point_cloud = frame.point_cloud()

        print("Converting to BGR image in OpenCV format")
        bgr = _point_cloud_to_cv_bgr(point_cloud)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        bgr_image_file = "test/ImageRGB.png"
        print(f"Visualizing and saving BGR image to file: {bgr_image_file}")
        _visualize_and_save_image(bgr, bgr_image_file, "BGR image")
        _detect_objects(bgr)

        print("Converting to Depth map in OpenCV format")
        z_color_map = point_cloud.copy_data('xyz')

        with open("test/result.txt", "r") as f:
            for line in f:
                xywh = line.strip().split("\n")
                char = ['[', ']', ',']
                for c in char:
                    xywh = [line.replace(c, "") for line in xywh]
                xywh = [list(map(float, line.strip().split())) for line in xywh]
                xywh = np.array(xywh)
                x_pixel = int(xywh[0][0])
                y_pixel = int(xywh[0][1] + 150)
                color_detected = _detect_color(hsv, y_pixel -150, x_pixel)
                print(f"Object at x_coord: {x_pixel}, y_coord: {y_pixel -150} is in {color_detected} color")

                if y_pixel > 1200:
                    y_pixel = 1200
                x_value = z_color_map[y_pixel, x_pixel, 0]
                y_value = z_color_map[y_pixel, x_pixel, 1]
                z_value = z_color_map[y_pixel, x_pixel, 2]
                if 'nan' not in str(x_value) and 'nan' not in str(y_value) or 'nan' not in str(z_value):
                    with open("test/xyz_coordinate.txt", "a") as file:
                        print(f"XYZ value at pixel ({x_pixel}, {y_pixel}): {x_value} {y_value} {z_value}")
                        file.write(f"{x_value} {y_value} {z_value} {1} {color_detected}\n")

                with open("test/xyz_coordinate.txt", "r") as file:
                    lines = file.readlines()

                sorted_lines = sorted(lines, key=lambda line: float(line.split()[2]))

                with open("test/xyz_coordinate.txt", "w") as file:
                    file.writelines(sorted_lines)

    print("Detected object coordinates saved.")


if __name__ == "__main__":
    _main()

