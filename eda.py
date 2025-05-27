import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_image_sizes(directory_path):
    widths = []
    heights = []
    count = 0
    class_names = [d for d in sorted(os.listdir(directory_path)) if os.path.isdir(os.path.join(directory_path, d))]
    
    for class_name in class_names:
        class_path = os.path.join(directory_path, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except:
                pass
            count += 1

    return widths, heights

def count_images_in_directory(directory_path):
        class_counts = {}
        class_names = sorted(os.listdir(directory_path)) 
        
        for class_name in class_names:
            class_path = os.path.join(directory_path, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                class_counts[class_name] = count
                
        return class_counts


def get_classes(path):
    class_counts=count_images_in_directory(path)
    
    return list(class_counts.keys())
    

def magic(train_dir, val_dir, test_dir):
    # Analyzing Class Distributions
    train_class_counts = count_images_in_directory(train_dir)
    val_class_counts = count_images_in_directory(val_dir)
    test_class_counts = count_images_in_directory(test_dir)

    print(train_class_counts)
    
    train_size = list(set([cnt for _, cnt in train_class_counts.items()]))[0]*len(train_class_counts)
    val_size = list(set([cnt for _, cnt in val_class_counts.items()]))[0]*len(val_class_counts)
    test_size = list(set([cnt for _, cnt in test_class_counts.items()]))[0]*len(test_class_counts)
    
    print(f"Number of classes: {len(train_class_counts)}")
    print(f"Training set size: {train_size}")
    print(f"Testing set size: {val_size}")
    print(f"Validation set size: {test_size}")
    print(f"Overall dataset size: {train_size + val_size + test_size}")
    
    # Training - image from each class
    train_class_names = sorted(os.listdir(train_dir))
    train_class_names = [cn for cn in train_class_names if os.path.isdir(os.path.join(train_dir, cn))]

    fig, axes = plt.subplots(nrows=1, ncols=min(len(train_class_names), 5), figsize=(15, 3))
    for i, class_name in enumerate(train_class_names[:5]):
        class_path = os.path.join(train_dir, class_name)
        img_files = os.listdir(class_path)
        if len(img_files) > 0:
            img_path = os.path.join(class_path, img_files[0])
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(class_name, fontsize=8)
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Analyzing image dimentions
    # Get widths and heights from training, validation, and test sets
    train_widths, train_heights = get_image_sizes(train_dir)
    val_widths, val_heights = get_image_sizes(val_dir)
    test_widths, test_heights = get_image_sizes(test_dir)

    print("Unique widths in training set:", np.unique(train_widths))
    print("Unique heights in training set:", np.unique(train_heights))

    # histograms of widths for train, val, test
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(train_widths, bins=20, color='blue', alpha=0.7)
    plt.title("Training Widths Distribution")
    plt.xlabel("Width")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.hist(val_widths, bins=20, color='green', alpha=0.7)
    plt.title("Validation Widths Distribution")
    plt.xlabel("Width")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(test_widths, bins=20, color='red', alpha=0.7)
    plt.title("Test Widths Distribution")
    plt.xlabel("Width")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    # histograms of heights for train, val, test
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(train_heights, bins=20, color='blue', alpha=0.7)
    plt.title("Training Heights Distribution")
    plt.xlabel("Height")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.hist(val_heights, bins=20, color='green', alpha=0.7)
    plt.title("Validation Heights Distribution")
    plt.xlabel("Height")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(test_heights, bins=20, color='red', alpha=0.7)
    plt.title("Test Heights Distribution")
    plt.xlabel("Height")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_dir = 'data/train'
    val_dir = 'data/validation'
    test_dir = 'data/test'

    #get_classes(train_dir)
    
    magic(train_dir, val_dir, test_dir)

    # GPU Check
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))