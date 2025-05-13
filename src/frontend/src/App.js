import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [lang, setLang] = useState('en');
  const [answer, setAnswer] = useState('');
  const [retrieved, setRetrieved] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, lang }),
      });
      const data = await response.json();
      setAnswer(data.answer);
      setRetrieved(data.retrieved_essays);
    } catch (error) {
      console.error('Error:', error);
      setAnswer('Error processing query');
      setRetrieved([]);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Startup Assistant Bot</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Language:
          <select value={lang} onChange={(e) => setLang(e.target.value)}>
            <option value="en">English</option>
            <option value="hin">Hindi</option>
            <option value="tam">Tamil</option>
            <option value="tel">Telugu</option>
          </select>
        </label>
        <br />
        <label>
          Query:
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What does Paul Graham say about hiring?"
          />
        </label>
        <br />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Ask'}
        </button>
      </form>
      {answer && (
        <div>
          <h2>Answer:</h2>
          <p>{answer}</p>
          <h2>Retrieved Essays:</h2>
          <ul>
            {retrieved.map((essay, idx) => (
              <li key={idx}>{essay}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;