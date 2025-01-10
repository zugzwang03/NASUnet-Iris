import model
import dataloaders.__init__ as dataloader
import comparison

test = dataloader.test
predictions = model.predictions
save_predictions = model.save_predictions
comparison.evaluate_test_results(
    test[1], predictions, "textFiles/output.txt", "Result with casia dataset with right eye"
)

save_predictions(predictions, dataloader.output_folder)
