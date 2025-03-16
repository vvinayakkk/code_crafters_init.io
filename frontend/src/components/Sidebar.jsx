import { AnimatePresence, motion } from 'framer-motion';
import { useState } from 'react';
import { RiDashboard3Fill, RiVideoFill, RiSensorFill } from 'react-icons/ri';
import { BsCloudFill, BsBoxArrowRight } from 'react-icons/bs';
import { useNavigate } from 'react-router-dom';


const Sidebar = ({ sidebarOpen, setSidebarOpen }) => {
  const navigate = useNavigate();


  const handlePageChange = (page) => {
    setSidebarOpen(false);
    navigate(`/${page}`)
  };

  return (
    <AnimatePresence>
      {sidebarOpen && (
        <motion.div
          initial={{ x: -300 }}
          animate={{ x: 0 }}
          exit={{ x: -300 }}
          transition={{ duration: 0.3 }}
          className="fixed top-16 left-0 w-64 h-full bg-gray-800 text-white z-50 shadow-lg"
        >
          <div className="p-4">
            <h2 className="text-xl font-bold mb-4 text-blue-400">Menu</h2>
            <ul className="space-y-2">
              <li>
                <button
                  onClick={() => handlePageChange('dashboard')}
                  className="flex items-center space-x-3 w-full p-2 rounded-md hover:bg-gray-700"
                >
                  <RiDashboard3Fill size={20} />
                  <span>Dashboard</span>
                </button>
              </li>
              <li>
                <button
                  onClick={() => handlePageChange('video')}
                  className="flex items-center space-x-3 w-full p-2 rounded-md hover:bg-gray-700"
                >
                  <RiVideoFill size={20} />
                  <span>Video Feed</span>
                </button>
              </li>
              <li>
                <button
                  onClick={() => handlePageChange('sensor')}
                  className="flex items-center space-x-3 w-full p-2 rounded-md hover:bg-gray-700"
                >
                  <RiSensorFill size={20} />
                  <span>Sensor Data</span>
                </button>
              </li>
              <li>
                <button
                  onClick={() => handlePageChange('weather')}
                  className="flex items-center space-x-3 w-full p-2 rounded-md hover:bg-gray-700"
                >
                  <BsCloudFill size={20} />
                  <span>Weather Data</span>
                </button>
              </li>
              <li>
                <button
                  onClick={() => handlePageChange('login')}
                  className="flex items-center space-x-3 w-full p-2 rounded-md hover:bg-gray-700"
                >
                  <BsBoxArrowRight size={20} />
                  <span>Logout</span>
                </button>
              </li>
            </ul>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Sidebar;