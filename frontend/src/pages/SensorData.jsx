import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { BsThermometerHalf, BsGearFill, BsThreeDotsVertical } from 'react-icons/bs';
import { FaLocationArrow } from 'react-icons/fa';
import { MdSecurity } from 'react-icons/md';
import { TbBrandOpenai } from 'react-icons/tb';
import axios from 'axios';
import Sidebar from '../components/sidebar';
import data from '../data.json';
import AISection from '../components/aiSections';

const sampleSensorData = {
  temperature: 24,
  humidity: 45,
  motion: false,
  door: false,
  window: false,
  time: "12:00 PM"
};

function SensorDataPage() {
  const [selectedTab, setSelectedTab] = useState('view');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [modelView, setModelView] = useState(false);
  const [pop, setPop] = useState(true);
  const [isAutomatic, setIsAutomatic] = useState(false);
  const [formData, setFormData] = useState({
    temperature: '',
    humidity: '',
    lidar_distance: '',
    fiber_vibration: '',
    ugs_motion: '',
    rainfall: '',
    noise: '',
    description: ''
  });
  const [res, setRes] = useState();

  useEffect(() => {
    let interval;
    if (isAutomatic) {
      interval = setInterval(() => {
        const randomIndex = Math.floor(Math.random() * data.length);
        const sensorData = data[randomIndex];
        setFormData(sensorData);
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isAutomatic]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('https://meerkat-welcome-remotely.ngrok-free.app/predict', formData);
      if (response.data) {
        setRes(response.data);
      }
      setPop(false);
      const dataWithNotifications = {
        ...response.data,
        notification_settings: {
          whatsapp_number: "+917977409706",
          email: "ntpjc2vinayak@gmail.com",
          threshold: "Low"
        }
      };
      
      // Fire and forget - send to process_sensor_alert endpoint
        axios.post('https://free-horribly-perch.ngrok-free.app/process_sensor_alert', dataWithNotifications)
        .catch(error => {
          console.error('Error sending alert notification:', error);
        });
    } catch (error) {
      console.error('Error submitting data:', error);
    }
  };

  return (
    <>
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex justify-between items-center mb-6 bg-blue-900 text-white p-4">
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 hover:bg-blue-800 transition-colors mr-2"
              >
                <BsThreeDotsVertical size={24} />
              </button>
              <BsThermometerHalf size={24} className="text-blue-400" />
              <span>Sensor Analytics</span>
            </h1>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedTab('view')}
                className={`px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  selectedTab === 'view' 
                    ? 'bg-blue-700 text-white' 
                    : 'bg-blue-800 text-gray-200 hover:bg-blue-700'
                }`}
              >
                View
              </button>
              <button
                onClick={() => setSelectedTab('history')}
                className={`px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  selectedTab === 'history' 
                    ? 'bg-blue-700 text-white' 
                    : 'bg-blue-800 text-gray-200 hover:bg-blue-700'
                }`}
              >
                History
              </button>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold">Sensor Status</h2>
              <div className="flex space-x-3">
                <button
                  onClick={() => setModelView(!modelView)}
                  className={`px-4 py-1 rounded-md text-sm ${
                    modelView ? 'bg-green-500 text-white' : 'bg-gray-200'
                  }`}
                >
                  AI Model View
                </button>
                <button className="p-2 rounded-full hover:bg-gray-100">
                  <BsGearFill size={18} />
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gray-100 rounded-lg p-4">
                <h3 className="font-medium mb-4 flex items-center">
                  <BsThermometerHalf size={20} className="mr-2 text-red-500" />
                  Environmental
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span>Temperature</span>
                    <span className="font-medium">{sampleSensorData.temperature}°C</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-red-500 h-2.5 rounded-full"
                      style={{ width: `${(sampleSensorData.temperature / 30) * 100}%` }}
                    ></div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span>Humidity</span>
                    <span className="font-medium">{sampleSensorData.humidity}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-blue-500 h-2.5 rounded-full"
                      style={{ width: `${sampleSensorData.humidity}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="bg-gray-100 rounded-lg p-4">
                <h3 className="font-medium mb-4 flex items-center">
                  <FaLocationArrow size={18} className="mr-2 text-yellow-500" />
                  Motion & Activity
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span>Motion Detected</span>
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      sampleSensorData.motion ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {sampleSensorData.motion ? 'Active' : 'None'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span>Last Detection</span>
                    <span className="text-sm">{sampleSensorData.time}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span>Motion Intensity</span>
                    <div className="flex items-center">
                      <div className="w-24 bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-yellow-500 h-2.5 rounded-full"
                          style={{ width: sampleSensorData.motion ? '85%' : '0%' }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-gray-100 rounded-lg p-4">
                <h3 className="font-medium mb-4 flex items-center">
                  <MdSecurity size={20} className="mr-2 text-blue-500" />
                  Security Status
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span>Door Contact</span>
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      sampleSensorData.door ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {sampleSensorData.door ? 'Open' : 'Closed'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span>Window Contact</span>
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      sampleSensorData.window ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {sampleSensorData.window ? 'Open' : 'Closed'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span>System Status</span>
                    <span className="px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                      Armed
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            {modelView && !res && (
              <div className="mt-6 bg-gray-100 rounded-lg p-4">
                <h3 className="font-medium mb-4 flex items-center">
                  <TbBrandOpenai size={20} className="mr-2 text-purple-500" />
                  AI Analysis
                </h3>
                <p>Please submit sensor data to view AI analysis.</p>
              </div>
            )}

            {modelView && res && (
              <div className="mt-6 bg-gray-100 rounded-lg p-4">
                <h3 className="font-medium mb-4 flex items-center">
                  <TbBrandOpenai size={20} className="mr-2 text-purple-500" />
                  AI Analysis
                </h3>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    {/* Normal Activity Percentage */}
                    <div className="flex justify-between items-center">
                      <span>Normal</span>
                      <div className="flex items-center">
                        <span className="text-sm mr-2">
                          {res.major_class_name === 'normal' ? '90' : 
                            res.major_class_name === 'medium' ? '30' : '10'}%
                        </span>
                        <div className="w-24 bg-gray-200 rounded-full h-2.5">
                          <div
                            className="bg-green-500 h-2.5 rounded-full"
                            style={{ 
                              width: `${res.major_class_name === 'normal' ? 90 : 
                                res.major_class_name === 'medium' ? 30 : 10}%` 
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Medium Activity Percentage */}
                    <div className="flex justify-between items-center">
                      <span>Medium</span>
                      <div className="flex items-center">
                        <span className="text-sm mr-2">
                          {res.major_class_name === 'medium' ? '50' :
                            res.major_class_name === 'extreme' ? '30' : '10'}%
                        </span>
                        <div className="w-24 bg-gray-200 rounded-full h-2.5">
                          <div
                            className="bg-yellow-500 h-2.5 rounded-full"
                            style={{ 
                              width: `${res.major_class_name === 'medium' ? 50 :
                                res.major_class_name === 'extreme' ? 30 : 10}%` 
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Extreme Activity Percentage */}
                    <div className="flex justify-between items-center">
                      <span>{res.detailed_class_name || 'Extreme'}</span>
                      <div className="flex items-center">
                        <span className="text-sm mr-2">
                          {res.major_class_name === 'extreme' ? '90' :
                            res.major_class_name === 'medium' ? '20' : '0'}%
                        </span>
                        <div className="w-24 bg-gray-200 rounded-full h-2.5">
                          <div
                            className="bg-red-500 h-2.5 rounded-full"
                            style={{ 
                              width: `${res.major_class_name === 'extreme' ? 90 :
                                res.major_class_name === 'medium' ? 20 : 0}%` 
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-4 text-sm text-gray-600">
                    <p className={res.major_class_name === 'extreme' ? 'text-red-600 font-medium' : ''}>
                      {res.insights?.summary || 
                        `${res.detailed_class_name} activity detected with ${
                          res.major_class_name === 'extreme' ? 'high' : 
                          res.major_class_name === 'medium' ? 'moderate' : 'low'
                        } confidence.`}
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            <div className="mt-6">
              <h3 className="font-medium mb-4">Sensor Map</h3>
              <div className="aspect-video bg-white border rounded-lg relative">
                <svg width="100%" height="100%" viewBox="0 0 800 400">
                  <rect x="50" y="50" width="700" height="300" fill="none" stroke="#333" strokeWidth="4" />
                  <line x1="400" y1="50" x2="400" y2="350" stroke="#333" strokeWidth="4" />
                  <line x1="50" y1="200" x2="400" y2="200" stroke="#333" strokeWidth="4" />
                  
                  <text x="100" y="30" fontSize="14" fill="#333">Living Room</text>
                  <text x="450" y="30" fontSize="14" fill="#333">Bedroom</text>
                  <text x="100" y="230" fontSize="14" fill="#333">Kitchen</text>
                  
                  <circle cx="150" cy="100" r="10" fill={sampleSensorData.motion ? "#ff4444" : "#88ccff"} />
                  <text x="170" y="105" fontSize="12" fill="#333">Motion</text>
                  
                  <circle cx="350" cy="350" r="10" fill={sampleSensorData.door ? "#ff4444" : "#88ccff"} />
                  <text x="370" y="355" fontSize="12" fill="#333">Door</text>
                  
                  <circle cx="700" cy="100" r="10" fill={sampleSensorData.window ? "#ff4444" : "#88ccff"} />
                  <text x="720" y="105" fontSize="12" fill="#333">Window</text>
                  
                  <circle cx="150" cy="250" r="10" fill="#ff4444" />
                  <text x="170" y="255" fontSize="12" fill="#333">Temp</text>
                  
                  <circle cx="250" cy="250" r="10" fill="#88ccff" />
                  <text x="270" y="255" fontSize="12" fill="#333">Humidity</text>
                </svg>
              </div>
            </div>
          </div>

          {selectedTab === 'history' && (
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-semibold mb-4">Sensor Data History</h2>
              
              <div className="mb-4">
                <div className="flex space-x-3 mb-4">
                  <input
                    type="text"
                    placeholder="Search sensor data..."
                    className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                  <button className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-md">
                    Search
                  </button>
                </div>
                
                <div className="flex space-x-2 mb-4 overflow-x-auto pb-2">
                  <button className="bg-green-500 text-white px-3 py-1 rounded-full text-sm">
                    All Data
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Temperature
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Humidity
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Motion
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Door
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Window
                  </button>
                </div>
              </div>
              
              <div className="space-y-4">
                {/* Note: This should probably map over actual history data instead of sampleSensorData */}
                {[sampleSensorData].map((data, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-medium">Sensor Readings - {data.time}</h3>
                      <span className="px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                        Normal
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 mb-2">
                      <p>Temperature: {data.temperature}°C</p>
                      <p>Humidity: {data.humidity}%</p>
                      <p>Motion: {data.motion ? "Detected" : "None"}</p>
                      <p>Door: {data.door ? "Open" : "Closed"}</p>
                      <p>Window: {data.window ? "Open" : "Closed"}</p>
                    </div>
                    <div className="flex space-x-2">
                      <button className="text-green-500 hover:text-green-700 text-sm font-medium">
                        View Details
                      </button>
                      <button className="text-gray-500 hover:text-gray-700 text-sm">
                        Export
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-4 flex justify-center">
                <button className="bg-gray-200 hover:bg-gray-300 px-4 py-2 rounded-md">
                  Load More
                </button>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />

        {modelView && (
        <AISection data={res} />
        )}

      {/* Popup Form */}
      {pop && !isAutomatic && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg w-96 max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">Enter Sensor Data</h2>
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                <div>
                  <label className="block mb-1">LIDAR Distance (m)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="lidar_distance"
                    value={formData.lidar_distance}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                  />
                </div>
                <div>
                  <label className="block mb-1">Fiber Vibration</label>
                  <input
                    type="number"
                    step="0.01"
                    name="fiber_vibration"
                    value={formData.fiber_vibration}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                  />
                </div>
                <div>
                  <label className="block mb-1">UGS Motion</label>
                  <input
                    type="number"
                    step="1"
                    name="ugs_motion"
                    value={formData.ugs_motion}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                  />
                </div>
                <div>
                  <label className="block mb-1">Temperature (°C)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="temperature"
                    value={formData.temperature}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                  />
                </div>
                <div>
                  <label className="block mb-1">Humidity (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="humidity"
                    value={formData.humidity}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                  />
                </div>
                <div>
                  <label className="block mb-1">Rainfall (mm)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="rainfall"
                    value={formData.rainfall}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                  />
                </div>
                <div>
                  <label className="block mb-1">Noise Level</label>
                  <input
                    type="number"
                    step="0.1"
                    name="noise"
                    value={formData.noise}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                  />
                </div>
                <div>
                  <label className="block mb-1">Description</label>
                  <textarea
                    name="description"
                    value={formData.description}
                    onChange={handleInputChange}
                    className="w-full border rounded p-2"
                    rows="3"
                  />
                </div>
              </div>
              <div className="flex justify-between items-center mt-4">
                <button
                  type="button"
                  onClick={() => {
                    const randomIndex = Math.floor(Math.random() * data.length);
                    const randomData = data[randomIndex];
                    setFormData(randomData);
                  }}
                  className="px-4 py-2 bg-green-500 text-white rounded"
                >
                  Automate
                </button>
                <div className="flex space-x-2">
                  <button
                    type="button"
                    onClick={() => setPop(false)}
                    className="px-4 py-2 bg-gray-200 rounded"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-blue-500 text-white rounded"
                  >
                    Submit
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}

export default SensorDataPage;