import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

DATASET_PATH = "dataset/"

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

num_classes = len(classes)

class_mode = (
    "binary"
    if num_classes == 2
    else "categorical"
)

if os.path.exists("image_classifier.keras"):
    model = tf.keras.models.load_model(
        "image_classifier.keras"
    )

elif os.path.exists("image_classifier.h5"):
    model = tf.keras.models.load_model(
        "image_classifier.h5"
    )

else:
    raise FileNotFoundError(
        "Файл модели не найден"
    )


def predict_image(image_path):

    if not os.path.exists(image_path):
        print("Файл не найден")
        return

    try:
        Image.open(image_path).verify()
    except:
        print("Поврежденное изображение")
        return

    img = cv2.imread(image_path)

    if img is None:
        print("Ошибка чтения изображения")
        return

    original = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (128, 128)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(
        img,
        verbose=0
    )

    if class_mode == "binary":

        probability = float(
            prediction[0][0]
        )

        predicted_index = (
            1
            if probability > 0.5
            else 0
        )

        confidence = (
            probability
            if probability > 0.5
            else 1 - probability
        )

    else:

        predicted_index = np.argmax(
            prediction[0]
        )

        confidence = prediction[0][
            predicted_index
        ]

    predicted_class = classes[
        predicted_index
    ]

    print(
        f"Класс: {predicted_class}"
    )

    print(
        f"Уверенность: {confidence*100:.2f}%"
    )

    plt.imshow(original)

    plt.title(
        f"{predicted_class} ({confidence*100:.1f}%)"
    )

    plt.axis("off")
    plt.show()


predict_image(
    "dataset/dogs/images.jpg"
)