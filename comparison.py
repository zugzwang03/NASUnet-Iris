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

from datetime import datetime
import pytz
def evaluate_test_results(gt_array, pred_array, output_file='output.txt', reason="No specific reason provided"):
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

    # Convert to NumPy arrays for easier processing
    precision_arr = np.array(precision_arr)
    recall_arr = np.array(recall_arr)
    accuracy_arr = np.array(accuracy_arr)
    f1_arr = np.array(f1_arr)
    iou_arr = np.array(iou_arr)
    mAP_arr = np.array(mAP_arr)
    dice_arr = np.array(dice_arr)

    # Calculate top 10 mean for each metric
    def top_k_mean(arr, k=10):
        top_k_values = np.sort(arr)[-k:]  # Sort and take the top k
        return np.mean(top_k_values)

    top10_precision = top_k_mean(precision_arr)
    top10_recall = top_k_mean(recall_arr)
    top10_accuracy = top_k_mean(accuracy_arr)
    top10_f1 = top_k_mean(f1_arr)
    top10_iou = top_k_mean(iou_arr)
    top10_mAP = top_k_mean(mAP_arr)
    top10_dice = top_k_mean(dice_arr)

    # Calculate the mean confusion matrix by summing and averaging
    mean_conf_matrix = np.mean(conf_matrix_arr, axis=0)

    # Get the current date and time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

    # Append results to the file
    with open(output_file, 'a') as f:
        f.write(f"\n=== Evaluation Results ===\n")
        f.write(f"Date and Time (IST): {current_time_ist}\n")
        f.write(f"Reason for Execution: {reason}\n")
        f.write("Mean Confusion Matrix:\n")
        f.write(str(mean_conf_matrix) + '\n')
        f.write(f"Precision (mean): {np.mean(precision_arr):.4f}\n")
        f.write(f"Precision (top 10 mean): {top10_precision:.4f}\n")
        f.write(f"Recall (mean): {np.mean(recall_arr):.4f}\n")
        f.write(f"Recall (top 10 mean): {top10_recall:.4f}\n")
        f.write(f"Accuracy (mean): {np.mean(accuracy_arr):.4f}\n")
        f.write(f"Accuracy (top 10 mean): {top10_accuracy:.4f}\n")
        f.write(f"F1 Score (mean): {np.mean(f1_arr):.4f}\n")
        f.write(f"F1 Score (top 10 mean): {top10_f1:.4f}\n")
        f.write(f"IoU Score (mean): {np.mean(iou_arr):.4f}\n")
        f.write(f"IoU Score (top 10 mean): {top10_iou:.4f}\n")
        f.write(f"Mean Average Precision (mAP) (mean): {np.mean(mAP_arr):.4f}\n")
        f.write(f"Mean Average Precision (mAP) (top 10 mean): {top10_mAP:.4f}\n")
        f.write(f"Dice Loss (mean): {np.mean(dice_arr):.4f}\n")
        f.write(f"Dice Loss (top 10 mean): {top10_dice:.4f}\n")
