from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Input,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
import os

DATASET_PATH = "dataset/"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Папка {DATASET_PATH} не найдена")

classes = [
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
]

num_classes = len(classes)

if num_classes < 2:
    raise ValueError("Минимум 2 класса требуется")

class_mode = "binary" if num_classes == 2 else "categorical"

print(f"Найдено классов: {num_classes}")
print("Классы:", classes)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset="training",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset="validation",
    shuffle=False
)

model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(
        1,
        activation="sigmoid"
    ) if class_mode == "binary"
    else Dense(
        num_classes,
        activation="softmax"
    )
])

loss_function = (
    "binary_crossentropy"
    if class_mode == "binary"
    else "categorical_crossentropy"
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=loss_function,
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2
    ),

    ModelCheckpoint(
        "best_model.keras",
        save_best_only=True
    )
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(val_data)

print(f"\nТочность модели: {test_accuracy:.4f}")

model.save("image_classifier.keras")

print("Модель сохранена")