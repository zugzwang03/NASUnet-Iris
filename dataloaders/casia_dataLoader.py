import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
import numpy as np


def load_from_folder(
    start, end, leftOrRight, imgFolder, maskFolder, img_size=(64, 64), grayscale=True
):
    images = []
    masks = []
    image_name = []
    for i in range(start, end + 1):
        print(i)
        folder = f"{i:03d}"
        folder = os.path.join(imgFolder, folder, leftOrRight)
        print(folder)
        count = 0
        if os.path.exists(folder):
            for img_name in sorted(os.listdir(folder)):
                if img_name != "Thumbs.db":  # Exclude Thumbs.db
                    img_path = os.path.join(folder, img_name)
                    image_name.append(os.path.splitext(img_name)[0])
                    print(img_path)
                    img = load_img(
                        img_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    count += 1
                    img_array = img_to_array(img)  # Convert image to NumPy array
                    images.append(img_array)
            # Try to load corresponding mask (from L01 to L10)
            for mask_suffix in range(1, count + 1):  # Assuming masks can be L01 to L10
                mask_name = (
                    "OperatorA_" + f"S1{i:03d}{leftOrRight}{mask_suffix:02d}.tiff"
                )
                mask_path = os.path.join(maskFolder, mask_name)
                if os.path.exists(mask_path):
                    print(mask_path)
                    mask = load_img(
                        mask_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    mask_array = img_to_array(mask)
                    mask_array = (mask_array > 0).astype(np.float32)
                    masks.append(mask_array)
    return np.array(images), np.array(masks), image_name


baseFolder = "/content/drive/MyDrive/CASIA-Iris-Interval"
maskFolder = "/content/drive/MyDrive/casia4i"
leftOrRight = "R"

# Define output folder for predictions
if leftOrRight == "L":
    output_folder = "/content/drive/MyDrive/NASUnet-Iris/Iris-Outputs/Casia/L"
else:
    output_folder = "/content/drive/MyDrive/NASUnet-Iris/Iris-Outputs/Casia/R"  # Replace with your desired output folder path

train = load_from_folder(1, 200, leftOrRight, baseFolder, maskFolder, (64, 64), True)
test = load_from_folder(200, 250, leftOrRight, baseFolder, maskFolder, (64, 64), True)
val = load_from_folder(150, 200, leftOrRight, baseFolder, maskFolder, (64, 64), True)
