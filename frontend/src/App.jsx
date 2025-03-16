import { useState } from 'react'
import LoginPage from './pages/login'
import SensorDataPage from './pages/SensorData'
import WeatherDataPage from './pages/WeatherData'
import './index.css'
import VideoFeedPage from './pages/VideoFeed'
import { Route, Routes } from 'react-router-dom'
import Dashboard from './pages/dashboard'

function App() {
  const [count, setCount] = useState(0)

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/sensor" element={<SensorDataPage />} />
      <Route path="/video" element={<VideoFeedPage />} />
      <Route path="/weather" element={<WeatherDataPage />} />
    </Routes>
  )
}

export default App
