import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import cifar10
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f'Training data shape: {X_train.shape}')
print(f'Test data shape: {X_test.shape}')

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

print(f'After preprocessing - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')

def build_model(kernel_sizes):
    model = Sequential([
        Conv2D(32, kernel_sizes=(3,3), activation='relu', input_shape=(32, 32, 3), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_sizes=(3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_sizes=(3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

optimizers = {
    "Adam": Adam(learning_rate=0.001),
    "SGD": SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": RMSprop(learning_rate=0.001)
}

results = {}
kernels_to_test = [(3,3), (5,5), (7,7)]

for kernel_sizes in [(kernels_to_test[0], kernels_to_test[1], kernels_to_test[2])]:
    print(f"\nTesting with kernel sizes: {kernel_sizes}")
    for name, opt in optimizers.items():
        print(f"\nTraining with {name} optimizer")
        model = build_model(kernel_sizes)
        model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=['accuracy'])

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(f'best_cifar10_model_{name}.keras', save_best_only=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        results[name] = (test_loss, test_accuracy)
        print(f"{name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print(f"Classification Report for {name} with kernels {kernel_sizes}:")
        print(classification_report(y_true, y_pred_classes))

        print(f"Confusion Matrix for {name} with kernels {kernel_sizes}:")
        print(confusion_matrix(y_true, y_pred_classes))

model.save('final_cifar10_cnn_model.h5')

X_sample = X_test[0:1]
y_sample_true = np.argmax(y_test[0])

y_pred = model.predict(X_sample)
y_pred_label = np.argmax(y_pred[0])

print(f"Predicted Label: {y_pred_label}, True Label: {y_sample_true}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
