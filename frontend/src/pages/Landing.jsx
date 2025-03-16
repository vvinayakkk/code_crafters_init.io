import React, { useEffect, useState } from "react";
import Footer from '../components/Footer';

import { BsChatDots } from 'react-icons/bs'; // Using react-icons for a chat icon
import { motion } from 'framer-motion';
import { useNavigate } from "react-router-dom";
import { TextGenerateEffect } from "../components/ui/text-generate-effect";
import { HoverBorderGradient } from "../components/ui/hover-border-gradient";
import Features from "../components/Features";
//import { FeaturesCards } from "../components/FeaturesCards";


const Home = () => {
  const [isChatBotVisible, setIsChatBotVisible] = useState(false);
  const navigate = useNavigate();
  const toggleChatBot = () => {
    setIsChatBotVisible(!isChatBotVisible);
  };

  const words1 = `Advanced Surveillance System`;
  const words2 = `with AI Fusion`;

  useEffect(() => {
    // Inject Naker.io script dynamically
    const script = document.createElement("script");
    script.setAttribute("data-who", "💎 Made with naker.io 💎");
    script.src =
      "https://d23jutsnau9x47.cloudfront.net/back/v1.0.9/viewer.js";
    script.setAttribute(
      "data-option",
      JSON.stringify({
        environment: {
          gradient: "radial",
          sensitivity: 0.8,
          colorStart: [59,130,246,1],
          colorEnd: [68,188,112,1],
        },
        particle: {
          life: 5,
          power: 0.045,
          texture:
            "https://res.cloudinary.com/naker-io/image/upload/v1566560053/circle_05.png",
          number: 101,
          colorStart: [116, 129, 92, 0.13],
          colorEnd: [198,188,107,0.94],
          sizeStart: 1.57,
          sizeEnd: 3.14,
          direction1: { x: 100, y: 100, z: 100 },
          direction2: { x: -100, y: -100, z: -100 },
        },
      })
    );
    // Append to the hero container
    const heroContainer = document.querySelector("#hero-section");
    if (heroContainer) {
      heroContainer.appendChild(script);
    }

    return () => {
      // Cleanup script on component unmount
      if (heroContainer) {
        heroContainer.removeChild(script);
      }
    };
  }, []);


  return (
    <div className=" relative min-h-screen bg-gradient-to-br z-11 from-green-50 to-blue-50 ">
     
     
      <div id="hero-section" className="relative pt-5 px-7 h-screen w-full flex items-center flex-col justify-center px-4">
         {/* Overlay content */}
      <div className="relative z-20 text-center mt-2  max-w-5xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className=""
        >
          <h1 className="mt-10 mb-6 font-bold bg-clip-text text-transparent bg-white">
            <TextGenerateEffect words={words1} />
            <br /> 
            <span className="text-6xl text-green-400">{words2}</span> 
          </h1>
        </motion.div>

        {/* <motion.p 
          className="text-xl md:text-2xl mb-8 text-gray-800 max-w-3xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Join RECit and convert your company to green and sustainable by buying and selling renewable energy certificates (RECs) and green certificates (GCs).
        </motion.p> */}

        <motion.div
          className="flex flex-col sm:flex-row gap-4 justify-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >

          <HoverBorderGradient
            as="button"
            containerClassName="p-1 rounded-full"
            className="bg-violet-800 text-white font-semibold hover:bg-violet-700 transform hover:scale-105 transition-all"
            onClick={() => navigate("/login")}
          >
            Get Started Now
          </HoverBorderGradient>

          <HoverBorderGradient
            as="button"
            containerClassName="p-1 rounded-full"
            className="border-2 border-violet-800 text-violet-900 font-semibold hover:bg-violet-500/60 transform hover:scale-105 transition-all"
            onClick={() => navigate("/login")}
          >
            Learn More
          </HoverBorderGradient>

        </motion.div>

        {/* Stats */}
        <motion.div 
          className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <div className="bg-white/80 backdrop-blur-sm p-6 rounded-xl transform hover:scale-105 transition-all">
            <h3 className="text-4xl font-bold text-blue-600 mb-2">99.9%</h3>
            <p className="text-gray-600">Threat Detection Accuracy</p>
          </div>
          <div className="bg-white/80 backdrop-blur-sm p-6 rounded-xl transform hover:scale-105 transition-all">
            <h3 className="text-4xl font-bold text-green-500 mb-2">0.5s</h3>
            <p className="text-gray-600">Response Time</p>
          </div>
          <div className="bg-white/80 backdrop-blur-sm p-6 rounded-xl transform hover:scale-105 transition-all">
            <h3 className="text-4xl font-bold text-blue-600 mb-2">24/7</h3>
            <p className="text-gray-600">Real-time Monitoring</p>
          </div>
        </motion.div>
      </div>
      </div>
        <Features />
      <main className="container mx-auto my-7  px-4 py-8">
      </main>
      <Footer />
      </div>
      
   
  );
};

export default Home;