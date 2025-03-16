import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { TbBrandOpenai } from 'react-icons/tb';
import React from 'react';

const AISection = ({ data }) => {
  if (!data) {
    return (
      <div className="mt-6 bg-gray-100 rounded-lg p-4">
        <h3 className="font-medium mb-4 flex items-center">
          <TbBrandOpenai size={20} className="mr-2 text-purple-500" />
          No AI Analysis Data Available
        </h3>
      </div>
    );
  }

  // Extract classification data
  const predictedDetailedClass = data.detailed_class_name || "Unknown";
  const predictedMajorClass = data.major_class_name || "Unknown";
  
  // Access insights with fallback
  const insights = data.insights || {};

  // Format raw sensors data for pie chart
  const rawSensorsData = insights.raw_sensors ? 
    Object.entries(insights.raw_sensors).map(([key, value]) => ({
      name: key,
      value: value.value || 0,
      color: value.color || '#8884d8',
      severity: value.severity || 'N/A'
    })) : [];

  // Format risk scores data for bar chart
  const riskScoresData = insights.risk_scores ?
    Object.entries(insights.risk_scores).map(([key, value]) => {
      // Handle both object format and simple number format
      if (typeof value === 'object' && value !== null) {
        return {
          name: key.replace(/_/g, ' '),
          value: value.value || 0,
          color: value.color || '#8884d8',
          severity: value.severity || 'N/A'
        };
      } else {
        return {
          name: key.replace(/_/g, ' '),
          value: typeof value === 'number' ? value : 0,
          color: '#8884d8'
        };
      }
    }) : [];

  // Format basic metrics data for line chart
  const basicMetricsData = insights.basic_metrics ?
    Object.entries(insights.basic_metrics).map(([key, value]) => ({
      name: key.replace(/_/g, ' '),
      value: typeof value === 'number' ? value : 0
    })) : [];

  // Helper function to render text-based sections
  const renderTextSection = (sectionKey, title) => {
    if (!insights[sectionKey] || Object.keys(insights[sectionKey]).length === 0) {
      return null;
    }

    return (
      <div className="mt-6">
        <h4 className="font-medium mb-2">{title}</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(insights[sectionKey]).map(([key, value]) => (
            <div key={key} className="bg-white p-4 rounded-lg shadow-md">
              <h5 className="font-medium">{key.replace(/_/g, ' ')}</h5>
              <p className="text-sm text-gray-600">
                {typeof value === 'object' ? 
                  (value.value !== undefined ? 
                    `${value.value}${value.severity ? ` (${value.severity})` : ''}` : 
                    JSON.stringify(value)) : 
                  String(value)}
              </p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="mt-6 bg-gray-100 rounded-lg p-4">
      <h3 className="font-medium mb-4 flex items-center">
        <TbBrandOpenai size={20} className="mr-2 text-purple-500" />
        AI Analysis - {predictedDetailedClass} ({predictedMajorClass})
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Raw Sensors Data */}
        {rawSensorsData.length > 0 && (
          <div>
            <h4 className="font-medium mb-2">Raw Sensors</h4>
            <PieChart width={400} height={300}>
              <Pie
                data={rawSensorsData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(2)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {rawSensorsData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color || '#8884d8'} />
                ))}
              </Pie>
              <Tooltip formatter={(value, name, props) => [`${value} (${props.payload.severity})`, name]} />
            </PieChart>
          </div>
        )}

        {/* Risk Scores Data */}
        {riskScoresData.length > 0 && (
          <div>
            <h4 className="font-medium mb-2">Risk Scores</h4>
            <BarChart width={400} height={300} data={riskScoresData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip formatter={(value, name, props) => {
                return props.payload.severity ? 
                  [`${value} (${props.payload.severity})`, name] : 
                  [value, name];
              }} />
              <Legend />
              <Bar dataKey="value" fill="#8884d8">
                {riskScoresData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color || '#8884d8'} />
                ))}
              </Bar>
            </BarChart>
          </div>
        )}
      </div>

      {/* Basic Metrics Data */}
      {basicMetricsData.length > 0 && (
        <div className="mt-6">
          <h4 className="font-medium mb-2">Basic Metrics</h4>
          <LineChart width={800} height={300} data={basicMetricsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" activeDot={{ r: 8 }} />
          </LineChart>
        </div>
      )}

      {/* Additional sections based on what's available in the response */}
      {renderTextSection('additional_insights', 'Additional Insights')}
      {renderTextSection('composite_indices', 'Composite Indices')}
      {renderTextSection('differential_metrics', 'Differential Metrics')}
      {renderTextSection('interaction_terms', 'Interaction Terms')}
      {renderTextSection('physical_metrics', 'Physical Metrics')}
      {renderTextSection('probabilistic_metrics', 'Probabilistic Metrics')}
      {renderTextSection('statistical_measures', 'Statistical Measures')}
      {renderTextSection('time_based_metrics', 'Time Based Metrics')}
      {renderTextSection('transformations', 'Transformations')}
    </div>
  );
};

export default AISection;