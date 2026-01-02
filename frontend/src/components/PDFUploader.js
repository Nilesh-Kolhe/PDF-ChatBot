import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { FiUpload, FiFile } from 'react-icons/fi';
import './PDFUploader.css';

const PDFUploader = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];

    if (!file) return;

    if (file.type !== 'application/pdf') {
      setError('Please upload a PDF file');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        'http://localhost:5000/api/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      onUploadSuccess(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to upload PDF');
    } finally {
      setUploading(false);
    }
  }, [onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: false,
  });

  return (
    <div className="pdf-uploader">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />

        <div className="dropzone-content">
          {uploading ? (
            <>
              <div className="spinner"></div>
              <p>Processing PDF...</p>
            </>
          ) : (
            <>
              <FiUpload className="upload-icon" />
              <h3>Drop your PDF here</h3>
              <p>or click to browse</p>
              <div className="file-info">
                <FiFile /> Supports PDF files up to 16MB
              </div>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
};

export default PDFUploader;
