import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Rescaling
from keras.utils import image_dataset_from_directory, load_img, img_to_array

class PokemonModel:
    def __init__(self, model_path="models/pokemon_cnn.h5"):
        self.model_path = model_path
        self.img_size = (64, 64)
        self.model = None

        # Default fallback (overwritten after training)
        self.class_indices = ["pikachu", "raichu"] 

        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("No model found. Please train the model first.")

    def _build_architecture(self):
        model = Sequential([
            # Input shape must match image size
            Conv2D(32, 3, activation="relu", input_shape=(64, 64, 3)),
            MaxPool2D(2, 2),
            Conv2D(32, 3, activation="relu"),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, train_dir, test_dir, epochs=25):
        """Training pipeline using Keras 3 API"""
        
        # 1. Load Data
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

        # --- FIX IS HERE ---
        # Capture class names BEFORE mapping. 
        # .map() returns a new object that doesn't have .class_names
        self.class_indices = train_ds.class_names
        print("Classes found:", self.class_indices)
        # -------------------

        # 2. Normalize images (0-255 -> 0-1)
        normalization_layer = Rescaling(1.0 / 255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

        # 3. Train
        self.model = self._build_architecture()
        self.model.fit(train_ds, validation_data=test_ds, epochs=epochs)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print("Model saved successfully.")

    def predict(self, img_path):
        """Inference logic"""
        if self.model is None:
            return "Model not loaded", 0.0

        try:
            img = load_img(img_path, target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            score = float(self.model.predict(img_array)[0][0])
            
            # Since we are using binary_crossentropy:
            # The model usually learns classes alphabetically.
            # If class_indices is ['pikachu', 'raichu']:
            # 0 = pikachu, 1 = raichu.
            
            predicted_index = 1 if score > 0.5 else 0
            
            # Safety check if class_indices is list or dict
            if isinstance(self.class_indices, list):
                 prediction = self.class_indices[predicted_index]
            else:
                 # Fallback for old dictionary style
                 prediction = list(self.class_indices.keys())[list(self.class_indices.values()).index(predicted_index)]

            # Calculate confidence
            confidence = score if predicted_index == 1 else 1 - score
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error", 0.0