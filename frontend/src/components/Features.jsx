import React, { useState } from 'react';

import { FeaturesCards } from './FeaturesCards';

const Features = () => {
  const [showAlternatives, setShowAlternatives] = useState(false);

  const featuresData = [
    {
      title: "Voice-to-Text Conversion",
      description: "Convert spoken language into well-structured, accurate text effortlessly.",
      steps: [
        "Speak naturally",
        "Get clear, structured text",
      ],
      iconColor: "text-blue-500",
      hoverColor: "#4c9aff"
    },
    {
      title: "Intelligent Follow-Up Suggestions",
      description: "AI-powered suggestions to enhance conversation flow and clarity.",
      steps: [
        "Receive smart recommendations",
        "Improve response efficiency",
      ],
      iconColor: "text-green-500",
      hoverColor: "#06d6a0"
    },
    {
      title: "Real-Time Multilingual Translation",
      description: "Automatic language detection and seamless real-time translation.",
      steps: [
        "Detect language instantly",
        "Translate effortlessly",
      ],
      iconColor: "text-purple-500",
      hoverColor: "#9b5de5"
    },
    {
      title: "Context-Aware Messaging",
      description: "Understand message context for more meaningful and coherent conversations.",
      steps: [
        "Analyze context",
        "Generate precise responses",
      ],
      iconColor: "text-yellow-500",
      hoverColor: "#ffb703"
    },
    {
      title: "AI-Powered Workflow Assistance",
      description: "Automate communication tasks, meeting summaries, and workflow suggestions.",
      steps: [
        "Summarize discussions",
        "Automate responses",
      ],
      iconColor: "text-red-500",
      hoverColor: "#ff4d6d"
    },
    {
      title: "Offline Mode for Seamless Communication",
      description: "Use essential features even without an active internet connection.",
      steps: [
        "Work offline",
        "Sync automatically",
      ],
      iconColor: "text-gray-500",
      hoverColor: "#adb5bd"
    }
  ];
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 text-black flex flex-col items-center py-16">
      <h2 className="text-5xl font-bold text-black mb-4">Our Features</h2>
      <h4 className="text-md font-serif text-center text-gray-500 mb-12">
        Elevate communication with AI-driven solutions: <br/>
        From voice-to-text to workflow automation, experience seamless efficiency!
      </h4>

      {/* Toggle Switch for Alternative View */}
     

      {/* Feature Cards */}
      <div className="relative flex flex-wrap justify-center gap-8 p-8">
        {/* Overlay Image */}
        {/* Feature Cards */}
        <div className="flex flex-wrap justify-center gap-16 p-8">
          {featuresData.map((feature, index) => (
            <FeaturesCards
              key={index}
              title={feature.title}
              description={feature.description}
              steps={feature.steps}
              iconColor={feature.iconColor}
              hoverColor={feature.hoverColor}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Features;