import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect, useRef } from 'react';
import { BsThermometerHalf, BsCloudFill, BsGearFill, BsThreeDotsVertical, BsArrowUpCircleFill } from 'react-icons/bs';
import { FaLocationArrow, FaWind, FaCloud, FaTemperatureHigh, FaWater } from 'react-icons/fa';
import { TbBrandOpenai } from 'react-icons/tb';
import { WiHumidity, WiBarometer, WiRaindrop, WiSunrise } from 'react-icons/wi';
import { MdAir } from 'react-icons/md';
import Sidebar from '../components/sidebar';
import axios from 'axios';
import weatherData from '../data2.json';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell } from 'recharts';
import * as d3 from 'd3';

// Replace requestBodies with data from data2.json
const requestBodies = weatherData.data;

// Generate initial mock historical data
const generateHistoricalData = (currentValue, count = 24) => {
  return Array(count).fill().map((_, i) => {
    const hour = i % 24;
    const value = currentValue * (0.85 + 0.3 * Math.sin(i / 4)) + (Math.random() * 5 - 2.5);
    return {
      hour: `${hour}:00`,
      value: parseFloat(value.toFixed(2))
    };
  });
};

// Gauge component
const Gauge = ({ value, min = 0, max = 100, label, unit = '', color = '#3B82F6' }) => {
  const percentage = ((value - min) / (max - min)) * 100;
  const clampedPercentage = Math.min(Math.max(percentage, 0), 100);
  
  return (
    <div className="relative h-32">
      <div className="absolute inset-0 flex items-center justify-center flex-col">
        <div className="font-bold text-xl text-gray-800">{value.toFixed(1)}{unit}</div>
        <div className="text-gray-500 text-sm">{label}</div>
      </div>
      <svg className="w-full h-full" viewBox="0 0 120 120">
        <circle
          cx="60"
          cy="60"
          r="50"
          fill="none"
          stroke="#e0e0e0"
          strokeWidth="12"
          strokeDasharray="314"
          strokeDashoffset="0"
          strokeLinecap="round"
          transform="rotate(-90 60 60)"
        />
        <motion.circle
          initial={{ strokeDashoffset: 314 }}
          animate={{ strokeDashoffset: 314 - (clampedPercentage / 100) * 314 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          cx="60"
          cy="60"
          r="50"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeDasharray="314"
          strokeLinecap="round"
          transform="rotate(-90 60 60)"
        />
      </svg>
    </div>
  );
};

// Weather icons with animations
const WeatherIcon = ({ type, size = 40 }) => {
  const icons = {
    temperature: <motion.div animate={{ rotateZ: [0, 10, -10, 0] }} transition={{ repeat: Infinity, duration: 4 }}><FaTemperatureHigh size={size} className="text-red-500" /></motion.div>,
    humidity: <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ repeat: Infinity, duration: 3 }}><WiHumidity size={size} className="text-blue-400" /></motion.div>,
    wind: <motion.div animate={{ rotateZ: 360 }} transition={{ repeat: Infinity, duration: 8, ease: "linear" }}><FaWind size={size} className="text-cyan-500" /></motion.div>,
    cloud: <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 3 }}><FaCloud size={size} className="text-gray-500" /></motion.div>,
    pressure: <motion.div animate={{ scale: [1, 1.05, 1] }} transition={{ repeat: Infinity, duration: 2 }}><WiBarometer size={size} className="text-purple-500" /></motion.div>,
    rain: <motion.div animate={{ y: [0, 5, 0] }} transition={{ repeat: Infinity, duration: 1.5 }}><WiRaindrop size={size} className="text-blue-600" /></motion.div>,
    sun: <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 20, ease: "linear" }}><WiSunrise size={size} className="text-yellow-500" /></motion.div>,
    air: <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ repeat: Infinity, duration: 3.5 }}><MdAir size={size} className="text-teal-500" /></motion.div>,
  };
  
  return icons[type] || null;
};

function WeatherDataPage() {
  const [selectedTab, setSelectedTab] = useState('view');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [modelView, setModelView] = useState(false);
  const [weatherData, setWeatherData] = useState(null);
  const [targetWeatherData, setTargetWeatherData] = useState(null);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [expanded, setExpanded] = useState(null);
  const [historicalData, setHistoricalData] = useState({});
  const [targetHistoricalData, setTargetHistoricalData] = useState({}); // Target historical data for slow transitions
  const [chartType, setChartType] = useState('area');
  const [randomBody, setRandomBody] = useState(null);
  const mainRef = useRef(null);
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

  useEffect(() => {
    const handleScroll = () => {
      if (mainRef.current && mainRef.current.scrollTop > 300) {
        setShowScrollTop(true);
      } else {
        setShowScrollTop(false);
      }
    };
    
    const currentRef = mainRef.current;
    if (currentRef) {
      currentRef.addEventListener('scroll', handleScroll);
    }
    
    return () => {
      if (currentRef) {
        currentRef.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  useEffect(() => {
    const getRandomBody = () => {
      const randomIndex = Math.floor(Math.random() * requestBodies.length);
      return requestBodies[randomIndex];
    };

    const fetchWeatherData = async () => {
      try {
        const randomBody = getRandomBody();
        setRandomBody(randomBody);
        console.log('Using weather data:', randomBody);
        
        const response = await axios.post(
          'https://mentally-liberal-amoeba.ngrok-free.app/predict',
          randomBody
        );
        const data = response.data;
        setTargetWeatherData(data);
        
        if (!weatherData) {
          setWeatherData(data);
          const initialHistData = {};
          Object.entries(data.insights.raw_sensors).forEach(([key, sensor]) => {
            initialHistData[key] = generateHistoricalData(sensor.value);
          });
          Object.entries(data.insights.risk_scores).forEach(([key, risk]) => {
            if (typeof risk === 'object') {
              initialHistData[key] = generateHistoricalData(risk.value, 12);
            }
          });
          setHistoricalData(initialHistData);
          setTargetHistoricalData(initialHistData);
        }
      } catch (error) {
        console.error('Error fetching weather data:', error);
        const fallbackData = {
          "detailed_class": 1,
          "detailed_class_name": "Normal: Cloudy",
          "insights": {
            "physical_metrics": {
              "atmospheric_stability": 8.0,
              "convective_energy": 13200.0,
              "moisture_flux": 6.0,
              "precip_energy": 9810.0,
              "pressure_wave": 3.045,
              "solar_flux": 60.00000000000001,
              "storm_potential": 1.8,
              "thermal_gradient": 21.674876847290644,
              "visibility_index": 13.88888888888889,
              "wind_energy": 16.5375
            },
            "raw_sensors": {
              "cloud_cover": { "color": "white", "severity": "N/A", "value": 70.0 },
              "dew_point": { "color": "white", "severity": "N/A", "value": 14.0 },
              "humidity": { "color": "yellow", "severity": "Moderate", "value": 60.0 },
              "precipitation": { "color": "green", "severity": "Low", "value": 1.0 },
              "pressure": { "color": "red", "severity": "High", "value": 1015.0 },
              "solar_radiation": { "color": "white", "severity": "N/A", "value": 200.0 },
              "temperature": { "color": "yellow", "severity": "Moderate", "value": 22.0 },
              "wind_speed": { "color": "green", "severity": "Low", "value": 3.0 }
            },
            "risk_scores": {
              "cold_risk": { "color": "green", "severity": "Low", "value": -1.6373172048274185 },
              "extreme_risk": 8.84,
              "fog_risk": { "color": "yellow", "severity": "Moderate", "value": 51.66666666666667 },
              "heat_risk": { "color": "green", "severity": "Low", "value": 8.8 },
              "storm_risk": { "color": "green", "severity": "Low", "value": 0.3 }
            },
            "statistical_measures": { "mean_sensors": 173.125, "std_sensors": 323.9669880944662 },
            "time_based_metrics": { "event_duration": 267.15295268622975, "rate_of_change_temp": 0.0823498291102136 },
            "timestamp": "2025-03-16T02:02:10.644831"
          },
          "major_class": 0,
          "major_class_name": "Normal"
        };
        setTargetWeatherData(fallbackData);
        if (!weatherData) {
          setWeatherData(fallbackData);
          const initialHistData = {};
          Object.entries(fallbackData.insights.raw_sensors).forEach(([key, sensor]) => {
            initialHistData[key] = generateHistoricalData(sensor.value);
          });
          Object.entries(fallbackData.insights.risk_scores).forEach(([key, risk]) => {
            if (typeof risk === 'object') {
              initialHistData[key] = generateHistoricalData(risk.value, 12);
            }
          });
          setHistoricalData(initialHistData);
          setTargetHistoricalData(initialHistData);
        }
      }
    };

    fetchWeatherData();
    const interval = setInterval(fetchWeatherData, 10000);
    return () => clearInterval(interval);
  }, []);

  // Smoothly transition weatherData toward targetWeatherData
  useEffect(() => {
    if (!targetWeatherData || !weatherData) return;

    const transitionInterval = setInterval(() => {
      setWeatherData((prevData) => {
        const newData = JSON.parse(JSON.stringify(prevData));

        Object.keys(newData.insights.raw_sensors).forEach((key) => {
          const currentValue = newData.insights.raw_sensors[key].value;
          const targetValue = targetWeatherData.insights.raw_sensors[key].value;
          const step = (targetValue - currentValue) * 0.1;
          newData.insights.raw_sensors[key].value = currentValue + step;
          newData.insights.raw_sensors[key].color = targetWeatherData.insights.raw_sensors[key].color;
          newData.insights.raw_sensors[key].severity = targetWeatherData.insights.raw_sensors[key].severity;
        });

        Object.keys(newData.insights.risk_scores).forEach((key) => {
          if (typeof newData.insights.risk_scores[key] === 'object') {
            const currentValue = newData.insights.risk_scores[key].value;
            const targetValue = targetWeatherData.insights.risk_scores[key].value;
            const step = (targetValue - currentValue) * 0.1;
            newData.insights.risk_scores[key].value = currentValue + step;
            newData.insights.risk_scores[key].color = targetWeatherData.insights.risk_scores[key].color;
            newData.insights.risk_scores[key].severity = targetWeatherData.insights.risk_scores[key].severity;
          } else {
            const currentValue = newData.insights.risk_scores[key];
            const targetValue = targetWeatherData.insights.risk_scores[key];
            const step = (targetValue - currentValue) * 0.1;
            newData.insights.risk_scores[key] = currentValue + step;
          }
        });

        Object.keys(newData.insights.physical_metrics).forEach((key) => {
          const currentValue = newData.insights.physical_metrics[key];
          const targetValue = targetWeatherData.insights.physical_metrics[key];
          const step = (targetValue - currentValue) * 0.1;
          newData.insights.physical_metrics[key] = currentValue + step;
        });

        newData.insights.timestamp = targetWeatherData.insights.timestamp;

        return newData;
      });
    }, 100);

    return () => clearInterval(transitionInterval);
  }, [targetWeatherData]);

  // Generate and slowly transition historical data
  useEffect(() => {
    if (!targetWeatherData || !weatherData) return;

    // Generate new target historical data based on targetWeatherData
    const newTargetHistData = {};
    Object.entries(targetWeatherData.insights.raw_sensors).forEach(([key, sensor]) => {
      newTargetHistData[key] = generateHistoricalData(sensor.value);
    });
    Object.entries(targetWeatherData.insights.risk_scores).forEach(([key, risk]) => {
      if (typeof risk === 'object') {
        newTargetHistData[key] = generateHistoricalData(risk.value, 12);
      }
    });
    setTargetHistoricalData(newTargetHistData);

    // Slowly transition historicalData toward targetHistoricalData
    const transitionInterval = setInterval(() => {
      setHistoricalData((prevHistData) => {
        const newHistData = JSON.parse(JSON.stringify(prevHistData));

        Object.keys(newTargetHistData).forEach((key) => {
          if (!newHistData[key]) {
            newHistData[key] = newTargetHistData[key]; // Initialize if not present
          } else {
            newHistData[key].forEach((dataPoint, index) => {
              const currentValue = dataPoint.value;
              const targetValue = newTargetHistData[key][index].value;
              const step = (targetValue - currentValue) * 0.02; // Very small step for slow transition
              dataPoint.value = currentValue + step;
            });
          }
        });

        return newHistData;
      });
    }, 200); // Slower interval (200ms) for very slow transitions

    return () => clearInterval(transitionInterval);
  }, [targetWeatherData]);

  const getColorClass = (color) => {
    switch (color.toLowerCase()) {
      case 'red': return 'text-red-600';
      case 'yellow': return 'text-yellow-600';
      case 'green': return 'text-green-600';
      case 'white': return 'text-gray-600';
      default: return 'text-gray-600';
    }
  };
  
  const getBgColorClass = (color) => {
    switch (color.toLowerCase()) {
      case 'red': return 'bg-red-500';
      case 'yellow': return 'bg-yellow-500';
      case 'green': return 'bg-green-500';
      case 'white': return 'bg-gray-400';
      default: return 'bg-gray-400';
    }
  };
  
  const getWeatherIcon = (key) => {
    switch (key) {
      case 'temperature': return 'temperature';
      case 'humidity': return 'humidity';
      case 'wind_speed': return 'wind';
      case 'cloud_cover': return 'cloud';
      case 'pressure': return 'pressure';
      case 'precipitation': return 'rain';
      case 'solar_radiation': return 'sun';
      case 'dew_point': return 'air';
      default: return 'temperature';
    }
  };
  
  const scrollToTop = () => {
    if (mainRef.current) {
      mainRef.current.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }
  };
  
  const toggleExpand = (section) => {
    if (expanded === section) {
      setExpanded(null);
    } else {
      setExpanded(section);
    }
  };
  
  const formatRiskScoreData = () => {
    if (!weatherData) return [];
    
    return Object.entries(weatherData.insights.risk_scores)
      .filter(([key, risk]) => typeof risk === 'object')
      .map(([key, risk]) => ({
        name: key.replace('_', ' '),
        value: Math.max(0, risk.value),
        color: risk.color.toLowerCase() === 'green' ? '#22c55e' : 
               risk.color.toLowerCase() === 'yellow' ? '#eab308' : 
               risk.color.toLowerCase() === 'red' ? '#ef4444' : '#9ca3af'
      }));
  };
  
  const formatPhysicalMetricsData = () => {
    if (!weatherData) return [];
    
    const metrics = weatherData.insights.physical_metrics;
    const maxValues = {
      atmospheric_stability: 10,
      convective_energy: 15000,
      moisture_flux: 10,
      precip_energy: 10000,
      pressure_wave: 5,
      solar_flux: 100,
      storm_potential: 10,
      thermal_gradient: 30,
      visibility_index: 20,
      wind_energy: 30
    };
    
    return Object.entries(metrics).map(([key, value]) => ({
      subject: key.replace(/_/g, ' '),
      A: (value / maxValues[key]) * 100,
      fullMark: 100
    }));
  };

  if (!weatherData) return null;

  return (
    <div className="max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="flex justify-between items-center mb-6 bg-blue-900 text-white p-4">
          <h1 className="text-xl font-bold flex items-center space-x-2">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 hover:bg-blue-800 transition-colors mr-2"
            >
              <BsThreeDotsVertical size={24} />
            </button>
            <BsThermometerHalf size={24} className="text-blue-400" />
            <span>Weather Analytics</span>
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

        {/* Display the randomly fetched body */}
        {randomBody && (
          <div className="p-6 bg-gray-50 border-b">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Randomly Fetched Weather Data</h2>
            <pre className="bg-white p-4 rounded-lg shadow-md overflow-x-auto">
              <code>{JSON.stringify(randomBody, null, 2)}</code>
            </pre>
          </div>
        )}

        {/* Main dashboard content */}
        <div className="p-6">
          {/* Weather Overview */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-800">Current Weather</h2>
              <div className="text-sm text-gray-500">
                Last updated: {new Date(weatherData.insights.timestamp).toLocaleString()}
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(weatherData.insights.raw_sensors).map(([key, sensor]) => (
                <motion.div
                  key={key}
                  whileHover={{ scale: 1.03 }}
                  className="bg-white p-4 rounded-lg shadow-md"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center">
                      <WeatherIcon type={getWeatherIcon(key)} size={28} />
                      <h3 className="ml-2 font-medium text-gray-700 capitalize">
                        {key.replace(/_/g, ' ')}
                      </h3>
                    </div>
                    <div className={`px-2 py-1 text-xs rounded-full ${getBgColorClass(sensor.color)} bg-opacity-20 ${getColorClass(sensor.color)}`}>
                      {sensor.severity}
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-gray-800">
                    {sensor.value.toFixed(1)}
                    {key === 'temperature' ? '°C' : 
                      key === 'humidity' || key === 'cloud_cover' ? '%' : 
                      key === 'wind_speed' ? ' km/h' : 
                      key === 'pressure' ? ' hPa' : 
                      key === 'precipitation' ? ' mm' : 
                      key === 'solar_radiation' ? ' W/m²' : 
                      key === 'dew_point' ? '°C' : ''}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Risk Scores */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-800">Risk Assessment</h2>
              <button 
                onClick={() => toggleExpand('risks')}
                className="text-blue-500 hover:text-blue-700"
              >
                {expanded === 'risks' ? 'Collapse' : 'Expand'}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-medium text-gray-800 mb-4">Weather Risk Scores</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={formatRiskScoreData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value">
                        {formatRiskScoreData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-medium text-gray-800 mb-4">Extreme Weather Risk</h3>
                <div className="flex items-center justify-center">
                  <Gauge 
                    value={weatherData.insights.risk_scores.extreme_risk} 
                    min={0} 
                    max={20} 
                    label="Risk Level" 
                    color={
                      weatherData.insights.risk_scores.extreme_risk < 5 ? '#22c55e' : 
                      weatherData.insights.risk_scores.extreme_risk < 10 ? '#eab308' : 
                      '#ef4444'
                    }
                  />
                </div>
              </div>
            </div>

            {expanded === 'risks' && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 grid grid-cols-1 gap-4"
              >
                {Object.entries(weatherData.insights.risk_scores)
                  .filter(([key, risk]) => typeof risk === 'object')
                  .map(([key, risk]) => (
                  <div key={key} className="bg-white p-4 rounded-lg shadow-md">
                    <h4 className="font-medium text-gray-800 capitalize mb-2">{key.replace(/_/g, ' ')} Trend</h4>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={historicalData[key] || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="hour" />
                          <YAxis />
                          <Tooltip />
                          <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke={
                              risk.color.toLowerCase() === 'green' ? '#22c55e' : 
                              risk.color.toLowerCase() === 'yellow' ? '#eab308' : 
                              risk.color.toLowerCase() === 'red' ? '#ef4444' : '#9ca3af'
                            } 
                            strokeWidth={2} 
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ))}
              </motion.div>
            )}
          </div>

          {/* Physical Metrics */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-800">Atmospheric Metrics</h2>
              <button 
                onClick={() => toggleExpand('metrics')}
                className="text-blue-500 hover:text-blue-700"
              >
                {expanded === 'metrics' ? 'Collapse' : 'Expand'}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-medium text-gray-800 mb-4">Physical Measurements</h3>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart outerRadius={90} data={formatPhysicalMetricsData()}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="subject" />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} />
                      <Radar name="Values" dataKey="A" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {expanded === 'metrics' ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-white p-6 rounded-lg shadow-md"
                >
                  <h3 className="text-lg font-medium text-gray-800 mb-4">Key Values</h3>
                  <div className="space-y-3">
                    {Object.entries(weatherData.insights.physical_metrics).map(([key, value]) => (
                      <div key={key} className="flex justify-between items-center">
                        <span className="text-gray-700 capitalize">{key.replace(/_/g, ' ')}</span>
                        <span className="font-medium">{value.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              ) : (
                <div className="bg-white p-6 rounded-lg shadow-md">
                  <h3 className="text-lg font-medium text-gray-800 mb-4">Statistical Summary</h3>
                  <div className="space-y-4">
                    <div>
                      <p className="text-gray-600 mb-1">Mean Sensor Value</p>
                      <p className="text-xl font-semibold">{weatherData.insights.statistical_measures.mean_sensors.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-gray-600 mb-1">Standard Deviation</p>
                      <p className="text-xl font-semibold">{weatherData.insights.statistical_measures.std_sensors.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-gray-600 mb-1">Event Duration</p>
                      <p className="text-xl font-semibold">{weatherData.insights.time_based_metrics.event_duration.toFixed(2)} min</p>
                    </div>
                    <div>
                      <p className="text-gray-600 mb-1">Temperature Change Rate</p>
                      <p className="text-xl font-semibold">{weatherData.insights.time_based_metrics.rate_of_change_temp.toFixed(4)} °C/min</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Historical Charts */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-800">Historical Trends</h2>
              <div className="flex items-center space-x-2">
                <select 
                  className="px-3 py-2 border rounded-md text-sm"
                  value={chartType}
                  onChange={(e) => setChartType(e.target.value)}
                >
                  <option value="area">Area</option>
                  <option value="line">Line</option>
                  <option value="bar">Bar</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(weatherData.insights.raw_sensors).map(([key, sensor]) => (
                <div key={key} className="bg-white p-4 rounded-lg shadow-md">
                  <h3 className="font-medium text-gray-800 capitalize mb-2">{key.replace(/_/g, ' ')} History</h3>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      {chartType === 'area' ? (
                        <AreaChart data={historicalData[key] || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="hour" />
                          <YAxis />
                          <Tooltip />
                          <Area type="monotone" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
                        </AreaChart>
                      ) : chartType === 'bar' ? (
                        <BarChart data={historicalData[key] || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="hour" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="value" fill="#8884d8" />
                        </BarChart>
                      ) : (
                        <LineChart data={historicalData[key] || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="hour" />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
                        </LineChart>
                      )}
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Integration Section */}
          {modelView && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8 bg-white p-6 rounded-lg shadow-md"
            >
              <div className="flex items-center mb-4">
                <TbBrandOpenai size={24} className="text-blue-600 mr-2" />
                <h2 className="text-xl font-bold text-gray-800">AI Weather Analysis</h2>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-medium text-gray-800 mb-2">Model Insights</h3>
                <div className="p-4 bg-blue-50 rounded-lg">
                  <p className="text-gray-700 mb-3">Based on current atmospheric conditions, the AI model has classified the weather as:</p>
                  <div className="flex items-center mb-4">
                    <div className="px-4 py-2 bg-blue-100 text-blue-800 rounded-lg text-lg font-medium">
                      {weatherData.major_class_name}: {weatherData.detailed_class_name}
                    </div>
                  </div>
                  <p className="text-gray-700">This classification is determined by analyzing multiple atmospheric metrics and their relationships.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-800 mb-2">Contributing Factors</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-700">Temperature</span>
                      <span className={`px-2 py-1 rounded-full text-sm ${getColorClass(weatherData.insights.raw_sensors.temperature.color)}`}>
                        {weatherData.insights.raw_sensors.temperature.value.toFixed(1)}°C - {weatherData.insights.raw_sensors.temperature.severity}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-700">Humidity</span>
                      <span className={`px-2 py-1 rounded-full text-sm ${getColorClass(weatherData.insights.raw_sensors.humidity.color)}`}>
                        {weatherData.insights.raw_sensors.humidity.value.toFixed(1)}% - {weatherData.insights.raw_sensors.humidity.severity}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-700">Pressure</span>
                      <span className={`px-2 py-1 rounded-full text-sm ${getColorClass(weatherData.insights.raw_sensors.pressure.color)}`}>
                        {weatherData.insights.raw_sensors.pressure.value.toFixed(1)} hPa - {weatherData.insights.raw_sensors.pressure.severity}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-700">Cloud Cover</span>
                      <span className={`px-2 py-1 rounded-full text-sm ${getColorClass(weatherData.insights.raw_sensors.cloud_cover.color)}`}>
                        {weatherData.insights.raw_sensors.cloud_cover.value.toFixed(1)}% - {weatherData.insights.raw_sensors.cloud_cover.severity}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium text-gray-800 mb-2">Prediction Confidence</h3>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={[
                            { name: 'Major Class', value: 85 },
                            { name: 'Other Classes', value: 15 }
                          ]}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          outerRadius={60}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {[0, 1].map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-800 mb-2">AI Recommendations</h3>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-gray-700">
                    {weatherData.major_class_name === "Normal" ? 
                      `Weather conditions are currently within normal parameters. No specific precautions needed beyond standard weather awareness.` :
                      `Current weather conditions show elevated metrics. Consider monitoring for potential changes and prepare for possible weather intensification.`
                    }
                  </p>
                  <div className="mt-4 p-3 border-l-4 border-blue-500 bg-blue-50">
                    <p className="text-blue-800 font-medium">Key observation:</p>
                    <p className="text-gray-700">
                      {weatherData.insights.risk_scores.extreme_risk > 10 ?
                        `High extreme weather risk detected (${weatherData.insights.risk_scores.extreme_risk.toFixed(1)}). Stay alert for rapidly changing conditions.` :
                        `Low extreme weather risk (${weatherData.insights.risk_scores.extreme_risk.toFixed(1)}). Conditions are stable.`
                      }
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Footer */}
          <div className="mt-8 p-4 bg-white rounded-lg shadow-md">
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-500">
                Weather Analysis Dashboard v1.0
              </div>
              <div className="flex items-center space-x-2">
                <div className="text-sm text-gray-500">
                  Data updated: {new Date().toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Scroll to top button */}
        <AnimatePresence>
          {showScrollTop && (
            <motion.button
              initial={{ opacity: 0, right: 20, bottom: 20 }}
              animate={{ opacity: 1, right: 20, bottom: 20 }}
              exit={{ opacity: 0, right: 20, bottom: 20 }}
              onClick={scrollToTop}
              className="fixed p-3 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-colors"
            >
              <BsArrowUpCircleFill size={24} />
            </motion.button>
          )}
        </AnimatePresence>
      </motion.div>

      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />
    </div>
  );
}

export default WeatherDataPage;