import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image, ImageOps

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy =", acc)

def predict_image(path):
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0
    return np.argmax(model.predict(img))

sample = x_test[0].reshape(28, 28)
plt.imshow(sample, cmap='gray')
plt.title("Predicted = " + str(np.argmax(model.predict(x_test[0].reshape(1, 28, 28, 1)))))
plt.show()

plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.show()
