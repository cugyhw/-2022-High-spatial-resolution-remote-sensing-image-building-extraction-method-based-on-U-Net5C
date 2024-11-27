# Semantic Segmentation Evaluation

This is an evaluation metrics package for semantic segmentation, including confusion matrix calculation, pixel accuracy, mean IoU, etc. 

## Usage 
### for single image

```python
import semseg_metrics as metrics
confusion_matrix = np.zeros((n_labels, n_labels), dtype=np.int)
ignore_labels = [0]

# pred and gt should be the same size and 1ch image
confusion_matrix = metrics.calc_confusion_matrix(pred, gt, confusion_matrix)

print('pixel accuracy:', metrics.calc_pixel_accuracy(confusion_matrix, ignore_labels))
print('mean IoU:', metrics.calc_mean_IoU(confusion_matrix, ignore_labels)[0])
print('class IoU:', metrics.calc_mean_IoU(confusion_matrix, ignore_labels)[1])
print('mean precision:', metrics.calc_mean_precision(confusion_matrix, ignore_labels)[0])
print('class precision:', metrics.calc_mean_precision(confusion_matrix, ignore_labels)[1])
print('mean recall:', metrics.calc_mean_recall(confusion_matrix, ignore_labels)[0])
print('class recall:', metrics.calc_mean_recall(confusion_matrix, ignore_labels)[1])
```

### for multiple image

```python
import semseg_metrics as metrics
confusion_matrix = np.zeros((n_labels, n_labels), dtype=np.int)
ignore_labels = [0]

# pred and gt should be the same size and 1ch image
for i in range(n_dataset):
    confusion_matrix = metrics.calc_confusion_matrix(pred[i], gt[i], confusion_matrix)
```
