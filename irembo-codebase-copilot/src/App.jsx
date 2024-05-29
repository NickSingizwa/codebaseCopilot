import { useState } from 'react';
import UploadForm from './components/UploadForm';
import PromptForm from './components/PromptForm';
import Response from './components/Response';
import './App.css';

const App = () => {
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [response, setResponse] = useState('');

    return (
        <div className="App">
            <h1>Javascript Codebase Copilot</h1>
            <UploadForm onUploadSuccess={() => setUploadSuccess(true)} />
            {uploadSuccess && <PromptForm onQueryResponse={(response) => setResponse(response)} />}
            {response && <Response response={response} />}
        </div>
    );
};

export default App;
