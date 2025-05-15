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

    with app:
        data_file = "test/detect.zdf"
        print(f"Reading ZDF frame from file: {data_file}")
        frame = zivid.Frame(data_file)
        point_cloud = frame.point_cloud()

        print("Converting to BGR image in OpenCV format")
        bgr = _point_cloud_to_cv_bgr(point_cloud)

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
                if y_pixel > 1200:
                    y_pixel = 1200
                x_value = z_color_map[y_pixel, x_pixel, 0]
                y_value = z_color_map[y_pixel, x_pixel, 1]
                z_value = z_color_map[y_pixel, x_pixel, 2]
                with open("test/xyz_coordinate.txt", "a") as file:
                    if x_value != 'nan' and y_value != 'nan' and z_value != 'nan':
                        print(f"XYZ value at pixel ({x_pixel}, {y_pixel}): {x_value} {y_value} {z_value}")
                        file.write(f"{x_value} {y_value} {z_value} \n")
    print("Detected object coordinates saved.")


if __name__ == "__main__":
    _main()




