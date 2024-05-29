/* eslint-disable react/prop-types */
import { useState } from 'react';
import axios from 'axios';
// import Prism from 'prismjs';
// import 'prismjs/themes/prism.css';  

const PromptForm = ({ onQueryResponse }) => {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            const response = await axios.post('/ask', { query });
            onQueryResponse(response.data.response);
            // Prism.highlightAll();
            console.log(response, "my response");
        } catch (error) {
            alert('An error occurred. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <label htmlFor="query">Enter your question</label><br />
                <textarea id="query" value={query} onChange={(e) => setQuery(e.target.value)} required></textarea>
                <div className='loader-submit'>
                    <button type='submit' disabled={!query || loading}>Submit <span>{loading && <div className="loader"></div>}</span></button>
                </div>
            </form>
        </div>
    );
};

export default PromptForm;
