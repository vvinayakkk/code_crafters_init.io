import { useState } from 'react'
import LoginPage from './pages/login'
import SensorDataPage from './pages/SensorData'
import WeatherDataPage from './pages/WeatherData'
import './index.css'
import VideoFeedPage from './pages/VideoFeed'
import { Route, Routes } from 'react-router-dom'
import Dashboard from './pages/dashboard'
import GTranslate from './components/Gtranslate'
import Home from './pages/Landing'
import PoliceStationMap from './pages/Map'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <GTranslate />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/sensor" element={<SensorDataPage />} />
        <Route path="/video" element={<VideoFeedPage />} />
        <Route path="/weather" element={<WeatherDataPage />} />
        <Route path="/map" element={<PoliceStationMap />} />
      </Routes>
    </>
  )
}

export default App
