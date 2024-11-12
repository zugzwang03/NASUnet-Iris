from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, jaccard_score, average_precision_score
import numpy as np

def dice_score(gt_flat, pred_flat):
    # Calculate Dice score
    intersection = np.sum(gt_flat * pred_flat)
    dice = (2.0 * intersection) / (np.sum(gt_flat) + np.sum(pred_flat))
    return dice

def evaluate_segmentation(gt_image, pred_image):
    # Flatten images to 1D for comparison
    gt_flat = gt_image.flatten()
    pred_flat = pred_image.flatten()

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(gt_flat, pred_flat)

    # Calculate metrics
    precision = precision_score(gt_flat, pred_flat, average='binary')
    recall = recall_score(gt_flat, pred_flat, average='binary')
    accuracy = accuracy_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat, average='binary')
    iou = jaccard_score(gt_flat, pred_flat, average='binary')  # IoU
    dice = dice_score(gt_flat, pred_flat)  # Dice score

    # Calculate mean Average Precision (mAP)
    mAP = average_precision_score(gt_flat, pred_flat)  # average_precision_score gives the AP for binary class

    return conf_matrix, precision, recall, accuracy, f1, iou, mAP, dice

def evaluate_test_results(gt_array, pred_array):
    conf_matrix_arr = []
    precision_arr = []
    recall_arr = []
    accuracy_arr = []
    f1_arr = []
    iou_arr = []
    mAP_arr = []
    dice_arr = []

    for idx in range(len(gt_array)):
        conf_matrix, precision, recall, accuracy, f1, iou, mAP, dice = evaluate_segmentation(gt_array[idx], pred_array[idx])
        conf_matrix_arr.append(conf_matrix)
        precision_arr.append(precision)
        recall_arr.append(recall)
        accuracy_arr.append(accuracy)
        f1_arr.append(f1)
        iou_arr.append(iou)
        mAP_arr.append(mAP)
        dice_arr.append(dice)

    # Calculate the mean confusion matrix by summing and averaging
    mean_conf_matrix = np.mean(conf_matrix_arr, axis=0)

    # Display results
    print("Mean Confusion Matrix:\n", mean_conf_matrix)
    print(f"Precision: {np.mean(precision_arr):.4f}")
    print(f"Recall: {np.mean(recall_arr):.4f}")
    print(f"Accuracy: {np.mean(accuracy_arr):.4f}")
    print(f"F1 Score: {np.mean(f1_arr):.4f}")
    print(f"IoU Score: {np.mean(iou_arr):.4f}")
    print(f"Mean Average Precision (mAP): {np.mean(mAP_arr):.4f}")
    print(f"Dice Loss: {np.mean(dice_arr):.4f}")