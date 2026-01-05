import React, { useState } from "react";
import "./ImageUpload.css";

const ImageUpload = ({ onResult }) => {
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    onResult(file); // only pass file, no API call here
  };

  return (
    <div className="upload-container">
      <input
        type="file"
        id="file-upload"
        accept="image/*"
        onChange={handleFileChange}
        hidden
      />

      <label htmlFor="file-upload" className="upload-label">
        {preview ? (
          <img src={preview} alt="Preview" className="preview-img" />
        ) : (
          <div className="placeholder">
            <span>+ Upload Pokémon Image</span>
          </div>
        )}
      </label>
    </div>
  );
};

export default ImageUpload;
