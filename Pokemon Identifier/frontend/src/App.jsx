import React, { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import { uploadImage } from './services/api';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (file) => {
    setLoading(true);
    setResult(null);
    try {
      const data = await uploadImage(file);
      setResult(data);
    } catch (error) {
      alert("Failed to predict");
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Pokédex Identifier</h1>
      </header>
      <main>
        <div className="card">
          <ImageUpload onResult={handleUpload} />
          
          {loading && <p className="loading">Analyzing...</p>}
          
          {result && (
            <div className="result">
              <h2>It's {result.class}!</h2>
              {/* Optional: Show raw score */}
              {/* <small>Confidence Score: {result.confidence.toFixed(4)}</small> */}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;