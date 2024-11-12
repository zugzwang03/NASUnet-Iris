import nas
import os
import numpy as np
import matplotlib.pyplot as plt
import dataLoader
import tensorflow as tf
import cv2  # Import OpenCV for image processing

tuner = nas.tuner
train = dataLoader.train
val = dataLoader.val
test = dataLoader.test

best_model = tuner.get_best_models(num_models=1)[0]

# Train the U-Net model
history = best_model.fit(
    train[0], train[1], batch_size=8, epochs=5, validation_split=0.1
)

print(history.history.keys())

# Check the data for accuracy
print("Training Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])

# Plot the training and validation accuracy values
plt.figure(figsize=(8, 6))  # Set the size of the plot
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy over Epochs')
plt.grid(True)

plt.savefig('training_validation_accuracy.png')

# Evaluate the best model
loss, accuracy = best_model.evaluate(
    val[0], val[1]
)  # Assuming you have a validation set
print(f"Best Model - Loss: {loss}, Accuracy: {accuracy}")

y_out = best_model.predict(val[0])

# Optionally save the best model
best_model.save("best_unet_model.h5")

# # Predict masks
predictions = best_model.predict(test[0])

# Post-process predictions
# Assuming masks are binary (0 or 1), threshold the predictions
predictions = (predictions > 0.28).astype(np.uint8)


def save_predictions(predictions, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, pred in enumerate(predictions):
        # Remove single-dimensional entries
        pred_image = pred.squeeze()

        # Resize with Lanczos interpolation
        pred_image_resized_lanczos = cv2.resize(pred_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        # Apply Bilateral filter
        pred_image_resized_bilateral = cv2.bilateralFilter(pred_image_resized_lanczos, d=9, sigmaColor=75, sigmaSpace=75)

        # Scale to [0, 255] and convert to uint8
        pred_image_resized = (pred_image_resized_bilateral * 255).astype(np.uint8)

        # Save as grayscale image
        output_path = os.path.join(output_folder, f"pred_{test[2][i]}.png")
        plt.imsave(output_path, pred_image_resized, cmap="gray")
