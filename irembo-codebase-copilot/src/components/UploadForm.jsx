/* eslint-disable react/prop-types */
import { useState } from 'react';
import axios from 'axios';

const UploadForm = ({ onUploadSuccess }) => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return;
        setLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            alert(response.data.success);
            onUploadSuccess();
        } catch (error) {
            alert('An error occurred. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <label htmlFor="file">Upload your codebase (only .zip files are allowed):</label><br />
                <input type="file" id="file" accept=".zip" onChange={handleFileChange} required />
                <div className='loader-submit'>
                    <button type='submit' disabled={!file || loading}>Upload <span>{loading && <div className="loader"></div>}</span></button>
                </div>
            </form>
        </div>
    );
};

export default UploadForm;
