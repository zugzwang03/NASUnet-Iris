import model
import dataLoader
import comparison

test = dataLoader.test
predictions = model.predictions
save_predictions = model.save_predictions
comparison.evaluate_test_results(test[1], predictions)

# Define output folder for predictions
if dataLoader.leftOrRight == 'L':
    output_folder = '/content/drive/MyDrive/Fusion/L'
else:
    output_folder = '/content/drive/MyDrive/Fusion/R'  # Replace with your desired output folder path
save_predictions(predictions, output_folder)