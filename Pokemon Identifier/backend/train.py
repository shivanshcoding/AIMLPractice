import os
from model import PokemonModel

# 1. Initialize the model wrapper
# This looks for 'models/pokemon_cnn.h5'. If not found, it prepares to train a new one.
print("Initializing model architecture...")
poke_model = PokemonModel()

TRAIN_DIR = 'C:\\Users\\Shivansh Rana\\Desktop\\DTU Academic\\Codes\\Python\\AIMLPractice\\Pokemon Identifier\\Dataset\\Training' 
TEST_DIR = 'C:\\Users\\Shivansh Rana\\Desktop\\DTU Academic\\Codes\\Python\\AIMLPractice\\Pokemon Identifier\\Dataset\\Testing'

# Check if folders exist to avoid errors
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    print(f"Error: Could not find dataset at {TRAIN_DIR} or {TEST_DIR}")
    print("Please check your paths in train_manual.py")
else:
    # 3. Start Training
    print(f"Starting training for 25 epochs...")
    print(f"Reading images from: {TRAIN_DIR}")
    
    # This will train the model and save it to 'backend/models/pokemon_cnn.h5'
    poke_model.train(train_dir=TRAIN_DIR, test_dir=TEST_DIR, epochs=25)
    
    print("Training Complete! Model saved to 'backend/models/pokemon_cnn.h5'")