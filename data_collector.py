import zivid


# Specify the directory name
root = "datasets"
zdf_directory = "zdf_files"
images_directory = "images"
labels_directory = "labels"


# Connect to the camera
app = zivid.Application()
camera = app.connect_camera(serial_number="21469A61")
# print(f"Camera Info: {camera.info}")
# print(f"Camera State: {camera.state}")

settings = zivid.Settings(
    acquisitions=[zivid.Settings.Acquisition()],
    color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
)

with open("datasets/number_of_images.txt", "r") as f:
    number_of_image = int(f.read())
if number_of_image == "":
    number_of_image = 0

while True:
    # Take a 2D-3D frame and save the 3D point cloud and 2D image
    with camera.capture_2d_3d(settings) as frame:
        # Save the 3D point cloud
        point_cloud = frame.point_cloud()
        xyz = point_cloud.copy_data("xyz")
        rgba = point_cloud.copy_data("rgba_srgb")

        data_file = f"{root}/{zdf_directory}/{number_of_image}.zdf"
        frame.save(data_file)

        # Save the 2D image
        color_image = frame.frame_2d().image_bgra_srgb()
        color_image.save(f"{root}/{images_directory}/train/{number_of_image}.jpg")
        number_of_image += 1

    with open("datasets/number_of_images.txt", "w") as file:
        file.write(str(number_of_image))

    input("Press Enter to capture the next frame...")






# # Load the saved point cloud
# data_file = "data/"
# print(f"Loading point cloud from {data_file}")
# frame = zivid.Frame(data_file)
# print(frame)



