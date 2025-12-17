import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess(data_dir):
    """Simple data loading with keras preprocessing"""
    # Load images using keras built-in
    dataset = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='binary',
        image_size=(224, 224),
        batch_size=32,
        shuffle=True
    )
    
    # Normalize
    normalization = keras.layers.Rescaling(1./255)
    normalized_dataset = dataset.map(lambda x, y: (normalization(x), y))
    
    return normalized_dataset

def create_simple_model():
    """Simpler CNN model"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🚗 Simple Car vs Truck Classifier")
    print("=" * 40)
    
    DATA_DIR = "vehicle_dataset"
    
    if not os.path.exists(DATA_DIR):
        print(f"Create folder: {DATA_DIR}")
        print("With subfolders: cars/ and trucks/")
        return
    
    # Load data
    print("Loading data...")
    dataset = load_and_preprocess(DATA_DIR)
    
    # Split dataset
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)
    
    # Create and train model
    print("Training model...")
    model = create_simple_model()
    model.summary()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        verbose=1
    )
    
    # Debug: Print available keys in history
    print("\nAvailable keys in history.history:")
    for key in history.history.keys():
        print(f"  - {key}")
    
    # Evaluate
    print("Evaluating...")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Save model (using the recommended Keras format)
    model.save('simple_car_truck_model.keras')
    print("Model saved as 'simple_car_truck_model.keras'")
    
    # Plot - Handle different key naming conventions
    plt.figure(figsize=(10, 4))
    
    # Accuracy plot - try different possible key names
    plt.subplot(1, 2, 1)
    
    # Determine the correct keys
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    
    plt.plot(history.history[acc_key], label='Train Acc')
    plt.plot(history.history[val_acc_key], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simple_training_plot.png')
    plt.show()

if __name__ == "__main__":
    main()