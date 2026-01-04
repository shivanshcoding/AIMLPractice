import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

class PokemonModel:
    def __init__(self, model_path='models/pokemon_cnn.h5'):
        self.model_path = model_path
        self.img_size = (64, 64)
        self.model = None
        self.class_indices = {'pikachu': 0, 'raichu': 1} # Default map based on directory sort

        # Load existing model if available
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("No model found. Please train the model first.")

    def _build_architecture(self):
        """Recreates the architecture from your notebook"""
        cnn = Sequential()
        
        # Step 1 - Convolution & Pooling
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        
        # Second Layer
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        
        # Step 3 - Flattening
        cnn.add(Flatten())
        
        # Step 4 - Full Connection
        cnn.add(Dense(units=128, activation='relu'))
        
        # Step 5 - Output Layer (Binary)
        cnn.add(Dense(units=1, activation='sigmoid'))
        
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return cnn

    def train(self, train_dir, test_dir, epochs=25):
        """Training pipeline"""
        # Preprocessing (Augmentation)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='binary'
        )

        test_set = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='binary'
        )

        # Store class indices to ensure prediction mapping is correct
        self.class_indices = training_set.class_indices
        print(f"Class mapping: {self.class_indices}")

        # Build and Train
        self.model = self._build_architecture()
        self.model.fit(x=training_set, validation_data=test_set, epochs=epochs)
        
        # Save
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print("Model saved successfully.")

    def predict(self, img_path):
        """Inference logic"""
        if self.model is None:
            return "Model not loaded", 0.0

        # Load and preprocess image
        test_image = image.load_img(img_path, target_size=self.img_size)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Normalize (Crucial step often missed when moving from notebook to production)
        test_image = test_image / 255.0

        result = self.model.predict(test_image)
        score = float(result[0][0])

        # Determine class based on binary output (0 or 1)
        # Inverting the dictionary map {name: index} -> {index: name}
        prediction = "Unknown"
        
        # Standard logic: < 0.5 is class 0, > 0.5 is class 1
        predicted_index = 1 if score > 0.5 else 0
        
        # Find name for index
        for name, index in self.class_indices.items():
            if index == predicted_index:
                prediction = name
                
        return prediction, score