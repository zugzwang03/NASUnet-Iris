import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
import numpy as np


def load_from_folder(imgFolder, maskFolder, img_size=(64, 64), grayscale=True):
    images = []
    masks = []
    image_names = []
    mask_names = []
    count = 0
    for i in range(1, 224 + 1):
        print(i)
        folder = f"{i:03d}"
        folder = os.path.join(imgFolder, folder)
        print(folder)
        if os.path.exists(folder):
            for img_name in sorted(os.listdir(folder)):
                if img_name != "Thumbs.db":  # Exclude Thumbs.db
                    img_path = os.path.join(folder, img_name)
                    image_names.append(os.path.splitext(img_name)[0])
                    print(img_path)
                    img = load_img(
                        img_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    count += 1
                    print(count)
                    img_array = img_to_array(img)  # Convert image to NumPy array
                    images.append(img_array)
    maskCnt = 0
    for mask_name in sorted(os.listdir(maskFolder)):
        mask_path = os.path.join(maskFolder, mask_name)
        if os.path.exists(mask_path):
            print(mask_path)
            mask_names.append(os.path.splitext(mask_name)[0])
            maskCnt += 1
            print(maskCnt)
            mask = load_img(
                mask_path,
                target_size=img_size,
                color_mode="grayscale" if grayscale else "rgb",
            )
            mask_array = img_to_array(mask)
            mask_array = (mask_array > 0).astype(np.float32)
            masks.append(mask_array)
            if maskCnt == count:
                break

    images = np.array(images)
    masks = np.array(masks)

    train = images[0:2001], masks[0:2001], mask_names[0:2001]
    test = images[2001:2241], masks[2001:2241], mask_names[2001:2241]
    val = images[1000:1500], masks[1000:1500], mask_names[1000:1500]

    return train, test, val


baseFolder = "/content/drive/MyDrive/IITD_database"
maskFolder = "/content/drive/MyDrive/iitd"
output_folder = "/content/drive/MyDrive/NASUnet-Iris/Iris-Outputs/IITD"

train, test, val = load_from_folder(baseFolder, maskFolder, (64, 64), True)
