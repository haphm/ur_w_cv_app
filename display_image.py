import cv2

img = cv2.imread("datasets/images/train/25.jpg", cv2.IMREAD_COLOR)

with open("datasets/labels/train/25.txt") as f:
    lines = f.readlines()
    for line in lines:
        # Assuming the format is: class_id x_center y_center width height
        class_id, x_center, y_center, width, height = map(float, line.split())
        # Convert to pixel coordinates
        x_center = int(x_center * img.shape[1])
        y_center = int(y_center * img.shape[0])
        width = int(width * img.shape[1])
        height = int(height * img.shape[0])
        # Calculate the top-left and bottom-right coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw the class label
        cv2.putText(img, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()