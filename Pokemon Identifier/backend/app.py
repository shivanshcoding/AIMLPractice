import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model import PokemonModel

app = Flask(__name__)
CORS(app) # Allow frontend to communicate with backend

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model
pokemon_model = PokemonModel()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            label, score = pokemon_model.predict(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'class': label,
                'confidence': score,
                'message': f"I think this is a {label.capitalize()}!"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to trigger training manually if needed"""
    data = request.json
    train_dir = data.get('train_dir') # Send absolute path from frontend or Postman
    test_dir = data.get('test_dir')
    
    if not train_dir or not test_dir:
        return jsonify({'error': 'Paths required'}), 400
        
    try:
        pokemon_model.train(train_dir, test_dir)
        return jsonify({'message': 'Training completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)