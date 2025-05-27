import tensorflow as tf
import numpy as np
from eda import get_classes


# for the shuffle
def get_predictions_train(model, dataset):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in dataset:
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch, verbose=0)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    y_true = tf.concat([item for item in y_true], axis=0)
    y_pred_classes = tf.concat([item for item in y_pred], axis=0)
    class_labels = get_classes("data/train")

    return y_true, y_pred_classes, class_labels