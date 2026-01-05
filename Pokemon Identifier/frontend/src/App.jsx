import React, { useRef, useState } from "react";
import "./App.css";
import ImageUpload from "./components/ImageUpload";
import { uploadImage } from "./services/api";

function App() {
  const predictRef = useRef(null);
  const aboutRef = useRef(null);
  const contactRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const scrollTo = (ref) =>
    ref.current?.scrollIntoView({ behavior: "smooth" });

  // called when image is selected
  const handleUpload = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError("");
  };

  // called when Predict button is clicked
  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const data = await uploadImage(selectedFile);

      // cinematic delay for animation
      setTimeout(() => {
        setResult(data);
        setLoading(false);
      }, 2400);
    } catch {
      setError("Prediction failed. Try a clearer Pokémon image.");
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* NAVBAR */}
      <nav className="navbar">
        <h2 className="logo">Pokédex AI</h2>
        <div className="nav-links">
          <button onClick={() => scrollTo(predictRef)}>Predict</button>
          <button onClick={() => scrollTo(aboutRef)}>About</button>
          <button onClick={() => scrollTo(contactRef)}>Contact</button>
        </div>
      </nav>

      {/* PREDICT */}
      <section ref={predictRef} className="section hero">
        <h1>AI Pokémon Identifier</h1>
        <p>Upload an image → Predict → Watch AI think</p>

        <div className="card glass">
          <ImageUpload onResult={handleUpload} />

          {selectedFile && !loading && !result && (
            <button className="predict-btn" onClick={handlePredict}>
              Predict Pokémon
            </button>
          )}

          {loading && (
            <div className="predicting">
              <div className="scanner" />
              <p>Analyzing Pokémon...</p>
            </div>
          )}

          {error && <p className="error">{error}</p>}

          {result && (
            <div className="result reveal">
              <h2>
                It’s <span>{result.class}</span>!
              </h2>

              {result.confidence && (
                <div className="radial">
                  <svg viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="50" />
                    <circle
                      cx="60"
                      cy="60"
                      r="50"
                      style={{
                        strokeDashoffset:
                          314 - 314 * result.confidence,
                      }}
                    />
                  </svg>
                  <div className="radial-text">
                    {(result.confidence * 100).toFixed(2)}%
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* ABOUT */}
      <section ref={aboutRef} className="section about">
        <h2>How I Built This</h2>

        <div className="about-grid">
          <div className="about-card">
            <h3>🧠 Model</h3>
            <p>
              Trained a deep learning image classifier using transfer learning
              and fine-tuning.
            </p>
          </div>

          <div className="about-card">
            <h3>⚙️ Backend</h3>
            <p>
              FastAPI handles preprocessing, inference, and confidence scoring.
            </p>
          </div>

          <div className="about-card">
            <h3>🎨 Frontend</h3>
            <p>
              React with a clean prediction flow and animated AI feedback.
            </p>
          </div>

          <div className="about-card">
            <h3>🚀 Focus</h3>
            <p>
              UX, explainability, and making AI feel intuitive.
            </p>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer ref={contactRef} className="footer">
        <h3>Let’s Connect</h3>
        <p>Built by Shivansh Rana</p>
        <p>AI • ML • Generative & Agentic AI</p>
      </footer>
    </div>
  );
}

export default App;
