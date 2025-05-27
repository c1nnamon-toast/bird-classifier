import tensorflow as tf 

def generate_sets(train_dir, val_dir, test_dir):
    batch_size = 32
    img_size = (256, 256) 

    # spliting the dataset
    # - train       (shuffle=T)
    # - validation  (shuffle=F)
    # - test        (shuffle=F)

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True   
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False  
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False  
    )

    # Normalization without any augmentation
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Improving performance with cache() and prefetch()
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    
    return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
    train_dir = 'data/train'
    val_dir = 'data/validation'
    test_dir = 'data/test'

    train_set, val_set, test_set = generate_sets(train_dir, val_dir, test_dir)
    