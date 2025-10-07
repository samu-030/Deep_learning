
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train_flat = x_train.reshape((-1, 28*28))
x_test_flat  = x_test.reshape((-1, 28*28))

def build_mlp(input_dim=28*28, num_classes=10):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_mlp()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train_flat, y_train,
    validation_split=0.1,
    epochs=12,
    batch_size=128,
    verbose=2
)

test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc*100:.2f}%  —  Test loss: {test_loss:.4f}")

model.save("mnist_mlp_model.h5")
print("Model saved to mnist_mlp_model.h5")

preds = np.argmax(model.predict(x_test_flat[:10]), axis=1)
plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"P:{preds[i]}\nT:{y_test[i]}")
    plt.axis('off')
plt.show()

def predict_image(img_28x28, model):
  
    arr = img_28x28.astype('float32') / 255.0
    arr = arr.reshape(1, 28*28)
    probs = model.predict(arr)[0]
    pred = np.argmax(probs)
    return pred, probs

