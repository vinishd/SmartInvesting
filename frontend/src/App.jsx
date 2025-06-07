import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import { useEffect } from 'react'

import './App.css'

function App() {
  const [count, setCount] = useState(0)
  const [data, setData] = useState(null)
  const [price, setPrice] = useState(0)
  const [symbol, setSymbol] = useState('');
  const [inputValue, setInputValue] = useState('');


  useEffect(() => {
  fetch("http://127.0.0.1:5000/api/data")  // Changed from localhost to 127.0.0.1
    .then(res => res.json())
    .then(data => setData(data.message))
    .catch(error => console.error('Error:', error))
}, [])

  useEffect(() => {
  fetch("http://127.0.0.1:5000/api/price?") 
    .then(res => res.json())
    .then(price => setPrice(price.price))
    .catch(error => console.error('Error:', error))
}, [])

  const handleSubmit = (event) => {
    event.preventDefault();

    // Trim whitespace and validate
    const cleanSymbol = symbol.trim().toUpperCase();

    if (!/^[A-Z]{1,5}$/.test(cleanSymbol)) {
      alert("Please enter a valid stock symbol (1–5 uppercase letters)");
      return;
    }

    fetch(`/api/price?symbol=${cleanSymbol}`)
      .then(res => res.json())
      .then(data => {
        setPrice(data.price);
      })
      .catch(err => {
        console.error('Error fetching price:', err);
      });
  };

 const handleChange = (event) => {
       setSymbol(event.target.value);
     };



  return (
    <>
      <h1>Stock Analysis</h1>
      <p>Current Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>

      <div className="p-4">
      <h1 className="text-xl font-bold">Smart Investing Dashboard</h1>
      {data ? <div>{JSON.stringify(data)}</div> : <div>Loading...</div>}
    </div>

      <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={symbol}
        onChange={handleChange}
        placeholder="Enter stock symbol"
      />
      <button type="submit">Get Price</button>
    </form>

    {price !== null && (
      <p>Price for {symbol.toUpperCase()}: ${price}</p>
    )}

      
    </>
  )
}

export default App
