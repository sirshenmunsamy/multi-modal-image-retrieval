"use client";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5); // ✅ Allow user to select `top_k`
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post("http://127.0.0.1:8000/search", {
        text: query,
        top_k: topK, // ✅ Now dynamically using selected `top_k`
      });
      setResults(response.data.results);
    } catch (error) {
      setError("Failed to fetch results. Is the backend running?");
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mb-4">Multi-Modal Image Search</h1>

      <div className="flex space-x-2 w-full max-w-lg">
        <input
          type="text"
          className="px-4 py-2 w-full text-black rounded-lg"
          placeholder="Enter a search term..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select
          className="px-2 py-2 text-black rounded-lg"
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value))}
        >
          {[3, 5, 10, 15, 20].map((k) => (
            <option key={k} value={k}>
              Top {k}
            </option>
          ))}
        </select>
        <button
          className="bg-blue-500 px-4 py-2 rounded-lg"
          onClick={handleSearch}
          disabled={loading}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      {results.length > 0 && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          {results.map((result, index) => (
            <div key={index} className="bg-gray-800 p-4 rounded-lg">
              <img
  src={result.image}
  alt="Retrieved result"
  className="w-full h-40 object-contain rounded-lg"
/>
              <p className="mt-2 text-sm text-gray-400">Distance: {result.distance.toFixed(2)}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}