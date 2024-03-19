import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tabulate import tabulate

def pixel_accuracy(y_true, y_pred):

  return accuracy_score(y_true.flatten(), y_pred.flatten())

def pixel_precision(y_true, y_pred):

  true_positives = np.sum(np.logical_and(y_true, y_pred))
  false_positives = np.sum(np.logical_and(np.logical_not(y_true), y_pred))

  if true_positives + false_positives == 0:
      return 0.0
  else:
      return true_positives / (true_positives + false_positives)

def pixel_recall(y_true, y_pred):

  true_positives = np.sum(np.logical_and(y_true, y_pred))
  false_negatives = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))

  if true_positives + false_negatives == 0:
      return 0.0
  else:
      return true_positives / (true_positives + false_negatives)

def intersection_over_union(y_true, y_pred):

  intersection = np.logical_and(y_true, y_pred)
  union = np.logical_or(y_true, y_pred)
  return np.sum(intersection) / np.sum(union)

def dice_coefficient(y_true, y_pred):

  intersection = np.logical_and(y_true, y_pred)

  return 2.0 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))

def get_true_labels(mask, tumor=True):
   y_true = (mask[:,:,0]/255).astype(bool)

   if tumor == False:
      y_true = np.logical_not(y_true)

   return y_true

def get_pred_labels(mask, tumor=True):
   y_pred = (mask/255).astype(bool)

   if tumor == False:
      y_pred = np.logical_not(y_pred)

   return y_pred

def show_metrics(acc, prec, recall, iou, dice):

  # TODO: refactor this with dictionaries

  # Organize the metrics into a list of lists
  metrics_table = [
    ["Pixel Accuracy", f"{acc:0.4f}"],
    ["Pixel Precision", f"{prec:0.4f}"],
    ["Pixel Recall", f"{recall:0.4f}"],
    ["IoU", f"{iou:0.4f}"],
    ["Dice Coeff.", f"{dice:0.4f}"]
  ]

  # Display the metrics table
  print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

def conf_matrix(y_true, y_pred, normalise=None):
   
   conf_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten(), normalize=normalise)
   return conf_matrix

def show_conf_matrices(conf_matrix, conf_norm_matrix):

   fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

   text = """
   - 0: non-tumor pixel
   - 1: tumor pixel
   """

   fig.text(0.5, 0.90, text, ha='center', fontsize=16, color='black')

   sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True, ax=axs[0])
   axs[0].set_title("Confusion matrix")
   axs[0].set_xlabel("Predicted label")
   axs[0].set_ylabel("True label")

   sns.heatmap(conf_norm_matrix, annot=True, fmt=".2%", cmap="Blues", linewidths=.5, square=True, ax=axs[1])
   axs[1].set_title("Confusion normalised matrix")
   axs[1].set_xlabel("Predicted label")
   axs[1].set_ylabel("True label")

   plt.show()
   
def visualize_bordered_mask(testImage, modelGuess, radius):
   # Define the disk structuring element
   radius = 2
   disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

   # Apply erosion using the disk structuring element
   eroded_mask = cv2.erode(modelGuess, disk_kernel)

   tumorBorder = modelGuess - eroded_mask

   map_rgb = cv2.cvtColor(tumorBorder, cv2.COLOR_GRAY2RGB)
   alpha = 0.5  # Adjust the transparency of the overlaid image
   overlay = cv2.addWeighted(testImage[:1283, :2040], alpha, map_rgb, 1 - alpha, 0)

   plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
   plt.axis('off')
   plt.show()