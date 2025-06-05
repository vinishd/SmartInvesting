import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import { useEffect } from 'react'

import './App.css'

function App() {
  const [count, setCount] = useState(0)
  const [data, setData] = useState(null)

  useEffect(() => {
      // Call your backend here
      fetch('/api/data')  // Example endpoint
        .then(res => res.json())
        .then(data => setData(data))
    }, [])


    return (
    <>
      <h1>Stock Analysis</h1>
      <p>Current Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>

      <div className="p-4">
      <h1 className="text-xl font-bold">Smart Investing Dashboard</h1>
      {data ? <div>{JSON.stringify(data)}</div> : <div>Loading...</div>}
    </div>
      
    </>
  )
}

export default App
