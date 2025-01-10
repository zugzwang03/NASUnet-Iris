import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tqdm import tqdm  # Import tqdm for progress tracking


def load_from_folder(imgFolder, maskFolder, img_size=(64, 64), grayscale=True):
    images = []
    masks = []
    image_names = []

    # Total iterations for tqdm: 104 classes * 2 sessions * 15 images = 3120
    total_iterations = 104 * 2 * 15
    pbar = tqdm(total=total_iterations, desc="Loading Images and Masks")

    for i in range(1, 105):
        for j in range(1, 3):
            for idx in range(1, 16):
                img_name = f"C{i}_S{j}_I{idx}.tiff"
                mask_name = "OperatorA_" + img_name
                img_path = os.path.join(imgFolder, img_name)
                mask_path = os.path.join(maskFolder, mask_name)

                if os.path.exists(img_path) and os.path.exists(mask_path):
                    image = load_img(
                        img_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    mask = load_img(
                        mask_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    img_array = img_to_array(image)
                    mask_array = img_to_array(mask)
                    mask_array = (mask_array > 0).astype(np.float32)  # Binarize mask

                    images.append(img_array)
                    masks.append(mask_array)
                    image_names.append(os.path.splitext(img_name)[0])

                pbar.update(1)  # Update tqdm progress bar

    pbar.close()  # Close the progress bar once done

    images = np.array(images)
    masks = np.array(masks)

    train = images[0:2001], masks[0:2001], image_names[0:2001]
    test = images[2001:2251], masks[2001:2251], image_names[2001:2251]
    val = images[1000:1500], masks[1000:1500], image_names[1000:1500]

    return train, test, val


imgFolder = "/content/drive/MyDrive/Ubiris"
maskFolder = "/content/drive/MyDrive/ubiris"
output_folder = "/content/drive/MyDrive/NASUnet-Iris/Iris-Outputs/Ubiris"

train, test, val = load_from_folder(imgFolder, maskFolder, (64, 64), True)
