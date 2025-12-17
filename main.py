import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def create_augmentation_layer():
    """Create data augmentation layer"""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
    ])

def load_and_preprocess(data_dir, augment=True):
    """Load data with optional augmentation"""
    # Load images
    dataset = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='binary',
        image_size=(224, 224),
        batch_size=32,
        shuffle=True
    )
    
    # Apply augmentation during training
    if augment:
        augmentation = create_augmentation_layer()
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))
    
    # Normalize
    normalization = keras.layers.Rescaling(1./255)
    normalized_dataset = dataset.map(lambda x, y: (normalization(x), y))
    
    return normalized_dataset

def create_model_with_augmentation():
    """Create CNN model with input and augmentation built-in"""
    model = keras.Sequential([
        # 1. Define the input shape explicitly
        keras.layers.Input(shape=(224, 224, 3)),
        
        # 2. Data augmentation layers (active only during training)
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
        
        # 3. Normalization
        keras.layers.Rescaling(1./255),
        
        # 4. CNN layers
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # 5. Classification head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🚗 Car vs Truck Classifier with Data Augmentation")
    print("=" * 50)
    
    DATA_DIR = "vehicle_dataset"
    
    if not os.path.exists(DATA_DIR):
        print(f"Create folder: {DATA_DIR}")
        print("With subfolders: cars/ and trucks/")
        return
    
    # Load data
    print("Loading data...")
    
    # For small datasets, use a simpler approach
    # First, load without augmentation to get the data
    dataset = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        label_mode='binary',
        image_size=(224, 224),
        batch_size=32,
        shuffle=True
    )
    
    # Convert to numpy arrays
    images = []
    labels = []
    for batch in dataset:
        batch_images, batch_labels = batch
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
    
    X = np.concatenate(images)
    y = np.concatenate(labels)
    
    print(f"\nDataset Info:")
    print(f"Total images: {len(X)}")
    print(f"Cars: {sum(y == 0)}")
    print(f"Trucks: {sum(y == 1)}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData Split:")
    print(f"Training: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Testing: {len(X_test)} images")
    
    # Create data augmentation
    data_augmentation = create_augmentation_layer()
    
    # Create model
    print("\nCreating model...")
    model = create_model_with_augmentation()
    model.summary()
    
    # Train model
    print("\nTraining model...")
    
    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Reduce learning rate when plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,  # More epochs with early stopping
        batch_size=16,  # Smaller batch size for small dataset
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Save model
    model.save('car_truck_model_augmented.keras')
    print("Model saved as 'car_truck_model_augmented.keras'")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("Training plot saved as 'training_results.png'")
    
    # Show some predictions
    print("\nSample Test Predictions:")
    predictions = model.predict(X_test[:5])
    
    classes = ['Car', 'Truck']
    for i in range(min(5, len(X_test))):
        pred_prob = predictions[i][0]
        pred_class = 1 if pred_prob >= 0.5 else 0
        true_class = int(y_test[i])
        
        confidence = pred_prob if pred_prob >= 0.5 else 1 - pred_prob
        
        print(f"Image {i+1}:")
        print(f"  Predicted: {classes[pred_class]} ({confidence:.1%} confidence)")
        print(f"  Actual: {classes[true_class]}")
        print(f"  {'✓ Correct' if pred_class == true_class else '✗ Wrong'}")
        print()

if __name__ == "__main__":
    main()