import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Rescaling
from keras.utils import image_dataset_from_directory, load_img, img_to_array


class PokemonModel:
    def __init__(self, model_path="models/pokemon_cnn.h5"):
        self.model_path = model_path
        self.classes_path = os.path.join(os.path.dirname(model_path), "classes.txt")
        self.img_size = (64, 64)
        self.model = None
        self.class_names = []

        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)

            # 🔥 LOAD CLASS NAMES
            if os.path.exists(self.classes_path):
                with open(self.classes_path, "r") as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                print("Loaded classes:", self.class_names)
            else:
                raise RuntimeError("classes.txt missing. Retrain the model.")
        else:
            print("No trained model found. Train the model first.")

    # -----------------------------
    # Model Architecture
    # -----------------------------
    def _build_architecture(self, num_classes):
        model = Sequential([
            Rescaling(1.0 / 255, input_shape=(64, 64, 3)),

            Conv2D(32, 3, activation="relu"),
            MaxPool2D(2, 2),

            Conv2D(64, 3, activation="relu"),
            MaxPool2D(2, 2),

            Flatten(),
            Dense(128, activation="relu"),

            Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, train_dir, test_dir, epochs=25):
        train_ds = image_dataset_from_directory(
            train_dir,
            image_size=self.img_size,
            batch_size=32
        )

        test_ds = image_dataset_from_directory(
            test_dir,
            image_size=self.img_size,
            batch_size=32
        )

        self.class_names = train_ds.class_names
        print("Classes found:", self.class_names)

        self.model = self._build_architecture(len(self.class_names))

        self.model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs
        )

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

        # 🔥 SAVE CLASS NAMES
        with open(self.classes_path, "w") as f:
            for name in self.class_names:
                f.write(name + "\n")

        print("Model and classes saved successfully.")

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, img_path):
        if self.model is None or not self.class_names:
            raise RuntimeError("Model or class names not loaded")

        img = load_img(img_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        preds = self.model.predict(img_array)[0]

        predicted_index = int(np.argmax(preds))
        confidence = float(np.max(preds))
        prediction = self.class_names[predicted_index]

        return prediction, confidence
