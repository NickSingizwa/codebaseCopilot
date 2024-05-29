/* eslint-disable react/prop-types */
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ResponseDisplay = ({ response }) => {
    // const codeString = '(num) => num + 1';
    return (
        <div>
            <h4>Response:</h4>
            <div id="response">
                <SyntaxHighlighter language="javascript" style={dark}>
                    {response}
                </SyntaxHighlighter>
            </div>
        </div>
    );
};

export default ResponseDisplay;
