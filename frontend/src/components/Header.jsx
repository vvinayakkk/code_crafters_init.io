import React, { useState } from 'react';
import { BsThreeDotsVertical, BsBoxArrowRight } from 'react-icons/bs';
import { MdOutlineNotifications } from 'react-icons/md';
import { FaUserCircle } from 'react-icons/fa';
import { GiCctvCamera } from 'react-icons/gi';
import Sidebar from './sidebar';


const Header = ({
  userName = 'Admin',
  initialNotificationCount = 2,
  title = 'SecureVision AI',
}) => {
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [notificationCount, setNotificationCount] = useState(initialNotificationCount);
  const [sidebarOpen, setSidebarOpen] = useState(false); // Added to sync with Sidebar
  
  // Function to toggle notifications
  const handleNotificationsToggle = () => {
    setNotificationsOpen((prev) => !prev);
  };

  // Function to handle logout
  const handleLogoutClick = () => {
    if (onLogout) {
      onLogout();
    }
  };

  // Function to toggle sidebar
  const handleSidebarToggle = () => {
    setSidebarOpen((prev) => !prev);
  };

  // Example function to add notifications (for demo purposes)
  const addNotification = () => {
    setNotificationCount((prev) => prev + 1);
  };

  return (
    <>
      <nav 
        className="bg-blue-900 text-white p-4 flex justify-between items-center sticky top-0 z-40" 
        style={{ backgroundColor: '#1E3A8A' }}
      >
        {/* Left Section - Menu Button and Title */}
        <div className="flex items-center space-x-3">
          <button
            onClick={handleSidebarToggle}
            className="p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 hover:bg-blue-800 transition-colors"
            aria-label="Toggle sidebar"
          >
            <BsThreeDotsVertical size={24} />
          </button>
          <div className="flex items-center space-x-2">
            <GiCctvCamera size={28} className="text-blue-400" />
            <span className="text-xl font-bold">{title}</span>
          </div>
        </div>

        {/* Right Section - Notifications and User */}
        <div className="flex items-center space-x-4">
          <button
            onClick={handleNotificationsToggle}
            className="relative p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 hover:bg-blue-800 transition-colors"
            aria-label="Toggle notifications"
          >
            <MdOutlineNotifications size={24} />
            {notificationCount > 0 && (
              <span className="absolute top-0 right-0 w-4 h-4 bg-red-500 rounded-full text-xs flex items-center justify-center">
                {notificationCount}
              </span>
            )}
          </button>
          <div className="flex items-center space-x-2">
            <FaUserCircle size={24} />
            <span className="font-medium hidden sm:inline">{userName}</span>
          </div>
            <button
              onClick={handleLogoutClick}
              className="p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 hover:bg-blue-800 transition-colors"
              aria-label="Logout"
            >
              <BsBoxArrowRight size={24} />
            </button>
        </div>
      </nav>

      {/* Sidebar Integration */}
      <Sidebar
        sidebarOpen={sidebarOpen} 
        setSidebarOpen={setSidebarOpen}
      />
    </>
  );
};

export default Header