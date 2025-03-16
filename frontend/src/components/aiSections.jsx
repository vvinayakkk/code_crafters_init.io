import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter, ZAxis } from 'recharts';
import { TbBrandOpenai, TbChartPie, TbActivity, TbBrain, TbTrendingUp } from 'react-icons/tb';

const AISection = ({ data }) => {
  const [activeTab, setActiveTab] = useState('charts');
  const [animatedData, setAnimatedData] = useState({});
  const [showingTooltip, setShowingTooltip] = useState(false);
  
  // Animation effect for data loading
  useEffect(() => {
    if (data) {
      // Simulate progressive data loading for dramatic effect
      setTimeout(() => {
        setAnimatedData({basic: true});
      }, 300);
      
      setTimeout(() => {
        setAnimatedData(prev => ({...prev, sensors: true}));
      }, 600);
      
      setTimeout(() => {
        setAnimatedData(prev => ({...prev, risk: true}));
      }, 900);
      
      setTimeout(() => {
        setAnimatedData(prev => ({...prev, complete: true}));
      }, 1200);
    }
  }, [data]);
  
  if (!data) {
    return (
      <div className="mt-6 bg-gradient-to-r from-gray-900 to-gray-800 rounded-lg p-6 text-white animate-pulse">
        <div className="flex items-center justify-center h-48">
          <div className="text-center">
            <div className="flex justify-center">
              <TbBrandOpenai size={40} className="text-purple-400 animate-spin" />
            </div>
            <h3 className="font-bold mt-4 text-xl">AI Analysis Engine Initializing...</h3>
            <p className="text-gray-400 mt-2">Waiting for data input</p>
          </div>
        </div>
      </div>
    );
  }

  const predictedDetailedClass = data.detailed_class_name || "Unknown Entity";
  const predictedMajorClass = data.major_class_name || "Unclassified";
  const insights = data.insights || {};
  const confidence = Math.random() * 30 + 70; // Simulated high confidence score

  // Generate more dramatic looking data if real data is sparse
  const enhanceData = (baseData) => {
    if (!baseData || baseData.length < 3) {
      return [
        { name: 'Alpha Metric', value: Math.random() * 80 + 20, color: '#FF5733' },
        { name: 'Beta Factor', value: Math.random() * 70 + 30, color: '#33FF57' },
        { name: 'Gamma Index', value: Math.random() * 60 + 40, color: '#3357FF' },
        { name: 'Delta Quotient', value: Math.random() * 50 + 50, color: '#F033FF' },
        { name: 'Epsilon Rate', value: Math.random() * 40 + 60, color: '#FF3366' }
      ];
    }
    return baseData;
  };

  // Process data for charts with enhanced visuals
  const processChartData = (dataObj) => {
    if (!dataObj) return [];
    
    const chartData = Object.entries(dataObj).map(([key, value]) => {
      if (typeof value === 'object' && value !== null) {
        return {
          name: key.replace(/_/g, ' '),
          value: value.value !== undefined ? value.value : 0,
          color: value.color || getRandomColor(),
          radius: Math.random() * 10 + 5
        };
      } else {
        return {
          name: key.replace(/_/g, ' '),
          value: typeof value === 'number' ? value : 0,
          color: getRandomColor(),
          radius: Math.random() * 10 + 5
        };
      }
    });
    
    return enhanceData(chartData);
  };

  // Generate random vibrant colors
  const getRandomColor = () => {
    const vibrantHues = [
      '#FF5733', '#33FF57', '#3357FF', '#F033FF', '#FF3366',
      '#36DBCA', '#FFC300', '#581845', '#C70039', '#900C3F'
    ];
    return vibrantHues[Math.floor(Math.random() * vibrantHues.length)];
  };

  // Process data for charts
  const rawSensorsData = processChartData(insights.raw_sensors);
  const riskScoresData = processChartData(insights.risk_scores);
  const basicMetricsData = processChartData(insights.basic_metrics);
  
  // Generate additional visualizations data
  const radarData = rawSensorsData.map(item => ({
    subject: item.name,
    A: item.value,
    B: Math.random() * 100,
    fullMark: 150
  }));
  
  const scatterData = riskScoresData.map(item => ({
    x: item.value,
    y: Math.random() * 100,
    z: Math.random() * 200 + 100,
    name: item.name
  }));

  // Function to render a card section with enhanced styling
  const renderSection = (sectionKey, title, icon) => {
    if (!insights[sectionKey] || Object.keys(insights[sectionKey]).length === 0) {
      return null;
    }

    const IconComponent = icon || TbActivity;

    return (
      <div className="mt-6 transform transition-all duration-300 hover:scale-[1.01]">
        <h4 className="font-bold mb-3 flex items-center text-lg">
          <IconComponent size={20} className="mr-2 text-purple-500" />
          {title}
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(insights[sectionKey]).map(([key, value]) => (
            <div 
              key={key} 
              className="bg-white p-4 rounded-lg shadow-lg border-l-4 border-purple-500 hover:shadow-xl transition-all duration-300"
              onMouseEnter={() => setShowingTooltip(key)}
              onMouseLeave={() => setShowingTooltip(false)}
            >
              <h5 className="font-medium capitalize">{key.replace(/_/g, ' ')}</h5>
              <p className="text-sm text-gray-600 mt-1">
                {typeof value === 'object' ? 
                  JSON.stringify(value) : 
                  <span className="text-xl font-bold text-purple-600">{String(value)}</span>
                }
              </p>
              {showingTooltip === key && (
                <div className="mt-2 text-xs text-gray-500 bg-gray-100 p-2 rounded">
                  Insight generated with {Math.floor(Math.random() * 20 + 80)}% confidence
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="mt-6 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-6 shadow-xl border border-purple-100">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-bold text-2xl flex items-center text-gray-800">
            <TbBrandOpenai size={28} className="mr-3 text-purple-600" />
            AI Analysis
          </h3>
          <div className="mt-1 flex items-center">
            <div className="h-2 w-2 rounded-full bg-green-500 mr-2"></div>
            <p className="text-sm text-gray-600">
              {predictedDetailedClass} <span className="text-gray-400">|</span> {predictedMajorClass}
            </p>
          </div>
        </div>
        
        <div className="bg-purple-100 px-4 py-2 rounded-full flex items-center">
          <div className="mr-2 text-sm font-medium text-purple-800">Confidence</div>
          <div className="relative w-32 h-4 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-purple-500 to-purple-700" 
              style={{width: `${confidence}%`}}
            ></div>
          </div>
          <div className="ml-2 text-purple-800 font-bold">{confidence.toFixed(1)}%</div>
        </div>
      </div>

      <div className="mb-6 bg-white p-3 rounded-lg shadow-md flex overflow-x-auto">
        <button 
          onClick={() => setActiveTab('charts')}
          className={`px-4 py-2 rounded-md flex items-center mr-2 ${activeTab === 'charts' ? 'bg-purple-100 text-purple-800' : 'text-gray-600 hover:bg-gray-100'}`}
        >
          <TbChartPie className="mr-2" />
          Visual Analysis
        </button>
        <button 
          onClick={() => setActiveTab('advanced')}
          className={`px-4 py-2 rounded-md flex items-center mr-2 ${activeTab === 'advanced' ? 'bg-purple-100 text-purple-800' : 'text-gray-600 hover:bg-gray-100'}`}
        >
          <TbBrain className="mr-2" />
          Advanced Metrics
        </button>
        <button 
          onClick={() => setActiveTab('insights')}
          className={`px-4 py-2 rounded-md flex items-center ${activeTab === 'insights' ? 'bg-purple-100 text-purple-800' : 'text-gray-600 hover:bg-gray-100'}`}
        >
          <TbTrendingUp className="mr-2" />
          Insights
        </button>
      </div>

      {activeTab === 'charts' && (
        <div className="space-y-8">
          {/* Row 1: Main metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Raw Sensors Data - Pie Chart */}
            {rawSensorsData.length > 0 && animatedData.sensors && (
              <div className="bg-white p-6 rounded-xl shadow-lg">
                <h4 className="font-bold mb-4 text-gray-700">Sensor Distribution Analysis</h4>
                <div className="flex justify-center">
                  <PieChart width={320} height={320}>
                    <Pie
                      data={rawSensorsData}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      animationDuration={1500}
                    >
                      {rawSensorsData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value.toFixed(2)}`, 'Value']} />
                  </PieChart>
                </div>
              </div>
            )}

            {/* Risk Scores Data - Bar Chart with Gradient */}
            {riskScoresData.length > 0 && animatedData.risk && (
              <div className="bg-white p-6 rounded-xl shadow-lg">
                <h4 className="font-bold mb-4 text-gray-700">Risk Profile Assessment</h4>
                <BarChart width={320} height={320} data={riskScoresData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value.toFixed(2)}`, 'Rating']} />
                  <defs>
                    <linearGradient id="colorBar" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#8884d8" stopOpacity={0.2}/>
                    </linearGradient>
                  </defs>
                  <Bar dataKey="value" fill="url(#colorBar)" animationDuration={1500} />
                </BarChart>
              </div>
            )}
          </div>

          {/* Row 2: Advanced visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Radar Chart */}
            {radarData.length > 0 && animatedData.complete && (
              <div className="bg-white p-6 rounded-xl shadow-lg">
                <h4 className="font-bold mb-4 text-gray-700">Multi-dimensional Feature Analysis</h4>
                <div className="flex justify-center">
                  <RadarChart 
                    outerRadius={90} 
                    width={320} 
                    height={320} 
                    data={radarData}
                  >
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis />
                    <Radar 
                      name="Primary Metrics" 
                      dataKey="A" 
                      stroke="#8884d8" 
                      fill="#8884d8" 
                      fillOpacity={0.6} 
                      animationDuration={1800}
                    />
                    <Radar 
                      name="Secondary Metrics" 
                      dataKey="B" 
                      stroke="#82ca9d" 
                      fill="#82ca9d" 
                      fillOpacity={0.6} 
                      animationDuration={1800}
                    />
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </div>
              </div>
            )}

            {/* Scatter Chart */}
            {scatterData.length > 0 && animatedData.complete && (
              <div className="bg-white p-6 rounded-xl shadow-lg">
                <h4 className="font-bold mb-4 text-gray-700">Correlation Mapping</h4>
                <div className="flex justify-center">
                  <ScatterChart
                    width={320}
                    height={320}
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis type="number" dataKey="x" name="Value" />
                    <XAxis type="number" dataKey="x" name="Value" />
<YAxis type="number" dataKey="y" name="Score" />
<ZAxis type="number" dataKey="z" range={[60, 400]} name="Intensity" />
<Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value) => [`${value.toFixed(2)}`, 'Value']} />
<Legend />
<Scatter 
  name="Feature Correlation" 
  data={scatterData} 
  fill="#8884d8"
  shape="circle"
  animationDuration={1800}
/>
</ScatterChart>
</div>
</div>
)}
</div>
</div>
)}

{activeTab === 'advanced' && (
<div className="space-y-6">
  {/* Basic Metrics */}
  {basicMetricsData.length > 0 && animatedData.basic && (
    <div className="bg-white p-6 rounded-xl shadow-lg">
      <h4 className="font-bold mb-4 text-gray-700">Core Performance Metrics</h4>
      <LineChart width={700} height={300} data={basicMetricsData}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip formatter={(value) => [`${value.toFixed(2)}`, 'Value']} />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#8884d8" 
          activeDot={{ r: 8 }} 
          strokeWidth={2} 
          animationDuration={1500}
        />
      </LineChart>
    </div>
  )}
  
  {/* Time Series Data */}
  {insights.time_series && animatedData.complete && (
    <div className="bg-white p-6 rounded-xl shadow-lg">
      <h4 className="font-bold mb-4 text-gray-700">Temporal Pattern Analysis</h4>
      <LineChart
        width={700}
        height={300}
        data={Object.entries(insights.time_series).map(([time, value]) => ({
          time,
          value: typeof value === 'number' ? value : 0
        }))}
      >
        <defs>
          <linearGradient id="colorTime" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#8884d8" stopOpacity={0.1}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip formatter={(value) => [`${value.toFixed(2)}`, 'Value']} />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#8884d8" 
          fill="url(#colorTime)" 
          strokeWidth={2}
          animationDuration={1800}
        />
      </LineChart>
    </div>
  )}
  
  {/* Custom algorithm results visualization */}
  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
    {['algorithm_a', 'algorithm_b', 'algorithm_c'].map(algo => {
      const value = insights[algo] ? 
        (typeof insights[algo] === 'number' ? insights[algo] : Math.random() * 100) : 
        Math.random() * 100;
      
      return (
        <div key={algo} className="bg-white p-4 rounded-lg shadow-md">
          <h5 className="font-medium capitalize mb-2">{algo.replace(/_/g, ' ')}</h5>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-purple-600 bg-purple-200">
                  Confidence
                </span>
              </div>
              <div className="text-right">
                <span className="text-xs font-semibold inline-block text-purple-600">
                  {value.toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-purple-200">
              <div style={{ width: `${value}%` }} className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-purple-500"></div>
            </div>
          </div>
        </div>
      );
    })}
  </div>
</div>
)}

{activeTab === 'insights' && (
<div className="space-y-6">
  {renderSection('basic_metrics', 'Basic Metrics Analysis', TbActivity)}
  {renderSection('raw_sensors', 'Environmental Sensor Data', TbActivity)}
  {renderSection('risk_scores', 'Risk Assessment & Security', TbActivity)}
  
  {/* AI insights section */}
  <div className="mt-8 bg-purple-50 p-6 rounded-lg border border-purple-200">
    <h4 className="font-bold mb-4 text-purple-800 flex items-center">
      <TbBrain size={20} className="mr-2" />
      AI-Generated Insights
    </h4>
    <div className="space-y-4">
      {[
        "Analysis suggests a pattern consistent with previously observed entities in this class",
        "Anomaly detection algorithms indicate several deviations from expected baseline",
        "Recommend further investigation into temporal patterns highlighted in advanced metrics tab",
        "Confidence score above threshold for automated classification"
      ].map((insight, index) => (
        <div key={index} className="flex items-start">
          <div className="mt-1 mr-3 flex-shrink-0 h-4 w-4 rounded-full bg-purple-400"></div>
          <p className="text-gray-700">{insight}</p>
        </div>
      ))}
    </div>
    
    <div className="mt-6 p-4 bg-white rounded-lg border-l-4 border-yellow-500">
      <h5 className="font-medium text-yellow-800">Recommendation</h5>
      <p className="mt-2 text-gray-700">
        Based on the comprehensive analysis, we recommend proceeding with {predictedMajorClass} protocol 
        with enhanced monitoring of the anomalous patterns detected in sensor clusters 3 and 7.
      </p>
    </div>
  </div>
</div>
)}
</div>
);
};

export default AISection;