import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BsExclamationTriangleFill, BsFillPlayFill, BsPauseFill, BsGearFill, BsFillCameraVideoFill, BsCloudFill } from 'react-icons/bs';
import { RiDashboard3Fill, RiVideoFill, RiSensorFill, RiAlarmWarningFill } from 'react-icons/ri';
import { FaLocationArrow, FaUserCircle } from 'react-icons/fa';
import { MdOutlineNotifications, MdNotificationsActive } from 'react-icons/md';
import { GiCctvCamera } from 'react-icons/gi';
import Header from '../components/Header'; // Import the Header component
import Sidebar from '../components/sidebar'; // Import the Sidebar component
import { useNavigate } from 'react-router-dom';
import videoSource from '../assets/video.mp4'  // Update this path to match your video location

// Sample data for visualization
const sampleAnomalyData = [
  { name: 'Abuse', confidence: 0.12 },
  { name: 'Arson', confidence: 0.05 },
  { name: 'Assault', confidence: 0.18 },
  { name: 'Burglary', confidence: 0.78 },
  { name: 'Fighting', confidence: 0.32 },
  { name: 'Normal', confidence: 0.22 },
  { name: 'RoadAccidents', confidence: 0.08 },
  { name: 'Robbery', confidence: 0.92 },
  { name: 'Shooting', confidence: 0.15 },
];

const sampleNotifications = [
  { id: 1, type: 'alert', message: 'Robbery detected at Sector 5', time: '10:15 AM', read: false },
  { id: 2, type: 'warning', message: 'Unusual activity detected in Sector 3', time: '09:45 AM', read: false },
  { id: 3, type: 'info', message: 'System update completed successfully', time: 'Yesterday', read: true },
];

function Dashboard({ onDataTypeSelect }) {
  const [anomalyData, setAnomalyData] = useState([]);
  const [loadingData, setLoadingData] = useState(true);
  const [playingVideo, setPlayingVideo] = useState(false); // Changed initial state to false
  const [sidebarOpen, setSidebarOpen] = useState(false); // State for sidebar visibility
  const [notificationsOpen, setNotificationsOpen] = useState(false); // State for notifications
  const navigate = useNavigate();
  const videoRef = useRef(null); // Add this line at the top with other state declarations
  const [isLoading, setIsLoading] = useState(false);
  const [isToggling, setIsToggling] = useState(false);

  useEffect(() => {
    // Simulate loading data
    setLoadingData(true);
    setTimeout(() => {
      setAnomalyData(sampleAnomalyData);
      setLoadingData(false);
    }, 1500);
  }, []);

  const handleDataTypeSelect = (type) => {
    onDataTypeSelect(type);
    onPageChange(type); // Trigger page change from parent
  };

  // Replace the toggleVideo function
  const toggleVideo = async () => {
    if (isToggling || !videoRef.current) return;
    
    setIsToggling(true);
    try {
      if (videoRef.current.paused) {
        setIsLoading(true);
        await videoRef.current.play();
        setPlayingVideo(true);
      } else {
        videoRef.current.pause();
        setPlayingVideo(false);
      }
    } catch (error) {
      console.error("Video playback error:", error);
      setPlayingVideo(false);
    } finally {
      setIsLoading(false);
      setIsToggling(false);
    }
  };

  // Add event listeners for video
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleEnded = () => setPlayingVideo(false);
    const handleError = () => {
      setPlayingVideo(false);
      setIsLoading(false);
    };

    video.addEventListener('ended', handleEnded);
    video.addEventListener('error', handleError);

    return () => {
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('error', handleError);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 overflow-hidden">
      {/* Header Component */}
      <Header
        userName="Admin"
        initialNotificationCount={2}
        title="SecureVision AI"
      />

      {/* Sidebar Component */}
      <Sidebar 
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />

      {/* Notifications Panel */}
      <AnimatePresence>
        {notificationsOpen && (
          <motion.div
            initial={{ x: 300 }}
            animate={{ x: 0 }}
            exit={{ x: 300 }}
            transition={{ duration: 0.3 }}
            className="fixed top-16 right-0 w-80 h-full bg-white z-50 shadow-lg"
          >
            <div className="p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">Notifications</h2>
                <button
                  onClick={() => setNotificationsOpen(false)}
                  className="p-1 rounded-md hover:bg-gray-200"
                >
                  ✕
                </button>
              </div>
              <div className="space-y-3">
                {sampleNotifications.map((notification) => (
                  <div 
                    key={notification.id}
                    className={`p-3 rounded-md border-l-4 ${
                      notification.type === 'alert'
                        ? 'border-red-500 bg-red-50'
                        : notification.type === 'warning'
                        ? 'border-yellow-500 bg-yellow-50'
                        : 'border-blue-500 bg-blue-50'
                    } ${notification.read ? 'opacity-70' : ''}`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex items-start space-x-2">
                        {notification.type === 'alert' ? (
                          <RiAlarmWarningFill size={20} className="text-red-500 mt-1" />
                        ) : notification.type === 'warning' ? (
                          <BsExclamationTriangleFill size={18} className="text-yellow-500 mt-1" />
                        ) : (
                          <MdNotificationsActive size={20} className="text-blue-500 mt-1" />
                        )}
                        <div>
                          <p className="font-medium">{notification.message}</p>
                          <p className="text-sm text-gray-500">{notification.time}</p>
                        </div>
                      </div>
                      {!notification.read && (
                        <span className="bg-blue-500 h-2 w-2 rounded-full"></span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Dashboard Content */}
      <main className="pt-4 px-4 pb-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-2xl font-bold mb-6">Security Analytics Dashboard</h1>
            
            {/* Video Area */}
            <div className="bg-white rounded-lg shadow-md p-4 mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">Historical Footage: Hiroshima & Nagasaki 1945</h2>
                <div className="flex space-x-2">
                  <button
                    onClick={toggleVideo}
                    className="p-2 rounded-full hover:bg-gray-100"
                  >
                    {playingVideo ? <BsPauseFill size={20} /> : <BsFillPlayFill size={20} />}
                  </button>
                  <button className="p-2 rounded-full hover:bg-gray-100">
                    <BsGearFill size={18} />
                  </button>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    preload="auto"
                    playsInline
                    autoPlay
                    loop
                    onClick={toggleVideo}
                    >
                    <source src='https://egdbvwtvwqhqorfknmfj.supabase.co/storage/v1/object/public/uploads//tmp0tt93co6%20(1)%20(1).mp4' type="video/mp4" />  // Fix MIME type
                    Your browser does not support the video tag.
                </video>
                  
                  <div 
                    className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 
                    bg-black bg-opacity-50 rounded-full p-4 cursor-pointer"
                    onClick={toggleVideo}
                  >
                    {isLoading ? (
                      <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent" />
                    ) : playingVideo ? (
                      <BsPauseFill size={30} className="text-white" />
                    ) : (
                      <BsFillPlayFill size={30} className="text-white" />
                    )}
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2 text-sm">
                    Historical footage from August 1945
                  </div>
                </div>
                
                <div className="bg-gray-100 rounded-lg p-4">
                  <h3 className="font-medium mb-2">Historical Context</h3>
                  <div className="space-y-2 text-sm">
                    <p>The atomic bombings of Hiroshima (August 6) and Nagasaki (August 9) marked the end of World War II.</p>
                    <p>This footage serves as a powerful reminder of the devastating impact of nuclear weapons.</p>
                    <div className="mt-4">
                      <h4 className="font-medium">Key Statistics:</h4>
                      <ul className="list-disc pl-4 mt-2">
                        <li>Hiroshima: August 6, 1945</li>
                        <li>Nagasaki: August 9, 1945</li>
                        <li>Combined casualties: Over 200,000</li>
                      </ul>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-100 rounded-lg p-4">
                  <h3 className="font-medium mb-2">Incident Statistics</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Today's Alerts:</span>
                      <span className="font-medium">7</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Critical Incidents:</span>
                      <span className="font-medium text-red-500">2</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Authorities Notified:</span>
                      <span className="font-medium">1</span>
                    </div>
                    <div className="flex justify-between">
                      <span>System Uptime:</span>
                      <span className="font-medium">99.8%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Data Options */}
            <h2 className="text-xl font-semibold mb-4">Available Data Sources</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <motion.div
                whileHover={{ scale: 1.03 }}
                transition={{ type: 'spring', stiffness: 300 }}
                className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer"
                onClick={() => handleDataTypeSelect('video-feed')}
              >
                <div className="bg-blue-500 p-4 text-white">
                  <BsFillCameraVideoFill size={24} />
                </div>
                <div className="p-4">
                  <h3 className="font-bold mb-2">Video Feeds</h3>
                  <p className="text-gray-600 text-sm">
                    Process and analyze video footage using our advanced AI algorithms to detect anomalies.
                  </p>
                  <button onClick={() => navigate('/video')} className="mt-4 flex justify-end">
                    <span className="text-blue-500 font-medium text-sm flex items-center">
                      View Data <FaLocationArrow size={12} className="ml-1" />
                    </span>
                  </button>
                </div>
              </motion.div>
              
              <motion.div
                whileHover={{ scale: 1.03 }}
                transition={{ type: 'spring', stiffness: 300 }}
                className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer"
                onClick={() => handleDataTypeSelect('sensor-data')}
              >
                <div className="bg-green-500 p-4 text-white">
                  <RiSensorFill size={24} />
                </div>
                <div className="p-4">
                  <h3 className="font-bold mb-2">Sensor Data</h3>
                  <p className="text-gray-600 text-sm">
                    Monitor sensor readings from connected devices such as motion sensors, door contacts, and temperature sensors.
                  </p>
                  <button onClick={() => navigate('/sensor')} className="mt-4 flex justify-end">
                    <span className="text-green-500 font-medium text-sm flex items-center">
                      View Data <FaLocationArrow size={12} className="ml-1" />
                    </span>
                  </button>
                </div>
              </motion.div>
              
              <motion.div
                whileHover={{ scale: 1.03 }}
                transition={{ type: 'spring', stiffness: 300 }}
                className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer"
                onClick={() => handleDataTypeSelect('weather-data')}
              >
                <div className="bg-purple-500 p-4 text-white">
                  <BsCloudFill size={24} />
                </div>
                <div className="p-4">
                  <h3 className="font-bold mb-2">Weather Data</h3>
                  <p className="text-gray-600 text-sm">
                    Access real-time weather conditions to correlate environmental factors with security events.
                  </p>
                  <button onClick={() => navigate('/weather')} className="mt-4 flex justify-end">
                    <span className="text-purple-500 font-medium text-sm flex items-center">
                      View Data <FaLocationArrow size={12} className="ml-1" />
                    </span>
                  </button>
                </div>
              </motion.div>
            </div>
            
            {/* Notifications Summary */}
            <div className="bg-white rounded-lg shadow-md p-4 mb-6">
              <h2 className="text-xl font-semibold mb-4">Recent Notifications</h2>
              <div className="space-y-3">
                {sampleNotifications.slice(0, 3).map((notification) => (
                  <div 
                    key={notification.id}
                    className={`p-3 rounded-md border-l-4 ${
                      notification.type === 'alert'
                        ? 'border-red-500 bg-red-50'
                        : notification.type === 'warning'
                        ? 'border-yellow-500 bg-yellow-50'
                        : 'border-blue-500 bg-blue-50'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex items-start space-x-2">
                        {notification.type === 'alert' ? (
                          <RiAlarmWarningFill size={20} className="text-red-500 mt-1" />
                        ) : notification.type === 'warning' ? (
                          <BsExclamationTriangleFill size={18} className="text-yellow-500 mt-1" />
                        ) : (
                          <MdNotificationsActive size={20} className="text-blue-500 mt-1" />
                        )}
                        <div>
                          <p className="font-medium">{notification.message}</p>
                          <p className="text-sm text-gray-500">{notification.time}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

export default Dashboard;