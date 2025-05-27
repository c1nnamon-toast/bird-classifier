import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from get_prediction_function import get_predictions_train  
from dataset_split import generate_sets  
from eda import get_classes  


def create_model(num_classes, input_shape=(256,256,3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', 
                               input_shape=input_shape,
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels, title):
    cm = confusion_matrix(y_true, y_pred)
    # number classes from 1 to N 
    num_classes = len(class_labels)
    numeric_labels = range(1, num_classes+1)

    plt.figure(figsize=(20, 18))  
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=numeric_labels, 
                yticklabels=numeric_labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_dir = 'data/train'
    val_dir = 'data/validation'
    test_dir = 'data/test'

    train_set, val_set, test_set = generate_sets(train_dir, val_dir, test_dir)

    class_names = get_classes("data/train")
    num_classes = len(class_names)
    print("Number of classes:", num_classes)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_set = train_set.map(lambda x,y: (data_augmentation(x), y))

    model = create_model(num_classes, input_shape=(256,256,3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=75, 
        callbacks=[early_stopping]
    )

    plot_training(history)

    test_loss, test_acc = model.evaluate(test_set)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save the model 
    model.save("my_model.h5")
    print("Model saved to my_model.h5")

    y_true_train, y_pred_train, train_class_labels = get_predictions_train(model, train_set)
    plot_confusion_matrix(y_true_train.numpy(), y_pred_train.numpy(), class_names, title='Training Confusion Matrix')

    y_true_test = []
    y_pred_test = []
    for images, labels in test_set:
        preds = model.predict(images, verbose=0)
        y_pred_test.extend(np.argmax(preds, axis=-1))
        y_true_test.extend(labels.numpy())

    plot_confusion_matrix(y_true_test, y_pred_test, class_names, title='Test Confusion Matrix')
