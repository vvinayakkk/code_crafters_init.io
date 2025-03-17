import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { BsCloudUpload, BsGearFill, BsBellFill, BsGraphUpArrow, BsThreeDotsVertical } from 'react-icons/bs';
import { FaRobot } from 'react-icons/fa';
import { BiVideo } from 'react-icons/bi';
import { MdLocationOn } from 'react-icons/md';  // Add this import
import Sidebar from '../components/sidebar';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function VideoFeedPage() {
  const [selectedTab, setSelectedTab] = useState('view');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const videoRef = useRef(null);
  const [uploadingVideo, setUploadingVideo] = useState(false);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [processedVideo1, setProcessedVideo1] = useState(null);
  const [processedVideo2, setProcessedVideo2] = useState(null);
  const [frames , setFrames] = useState([]);
  const [modelView, setModelView] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState('');
  const [location, setLocation] = useState(null);
  const [trackingLocation, setTrackingLocation] = useState(false);
  const navigate = useNavigate();
  const url1 = 'http://127.0.0.1:5000';
  const url2 = 'http://127.0.0.1:5001';
  const url3 = 'https://normal-joint-hamster.ngrok-free.app'

  const handleUploadVideo = (e) => {
    e.preventDefault();
    
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'video/*';
    
    fileInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      
      // Check file size (500MB limit)
      if (file.size > 500 * 1024 * 1024) {
        alert('File size exceeds 500MB limit');
        return;
      }

      setUploadingVideo(true);
      
      try {
        const formData = new FormData();
        formData.append('video', file);

        // Set initial video preview
        const videoUrl = URL.createObjectURL(file);
        setCurrentVideo(videoUrl);
        
        setUploadingVideo(false);
        setProcessingVideo(true);

        // Make API calls simultaneously
        const [response1, response2, response3, response4] = await Promise.all([
          axios.post(`${url1}/process_video`, formData, {
            responseType: 'blob',
            headers: {
              'Content-Type': 'application/octet-stream'
            }
          }),
          axios.post(`${url2}/process-video`, formData, {
            responseType: 'blob',
            headers: {
              'Content-Type': 'application/octet-stream'
            }
          }),
          axios.post(`${url3}/process-video`, formData, {
            responseType: 'blob',
            headers: {
              'Content-Type': 'application/octet-stream'
            }
          }),
          axios.post(`http://127.0.0.1:3001/analyze-video`, formData, {
            headers: {
              'Content-Type': 'application/octet-stream'
            }
          })
        ]);

        // Create URL from processed video blob
        const processedVideoUrl1 = URL.createObjectURL(response1.data);
        const processedVideoUrl2 = URL.createObjectURL(response2.data);
        setProcessedVideo1(processedVideoUrl1);
        setProcessedVideo2(processedVideoUrl2);
        
        // Parse the response3 data as JSON since it contains frames info
        const framesData = JSON.parse(await response3.data.text());
        setFrames(framesData.frames || []);
        
        // Get the analysis from response headers
        console.log(response4)
        const videoDescription = response4.data.description
        console.log(videoDescription)
        setAiAnalysis(videoDescription || 'No analysis available');

        const dataWithNotifications = {
          gemini_analysis: response4.data.description,
          model_analysis:response2.headers.X-Classification-Results,
          notification_settings: {
            whatsapp_number: "+917977409706",
            email: "ntpjc2vinayak@gmail.com",
            threshold: "Low"
          }
        };
        
        // Fire and forget - send to process_sensor_alert endpoint
        axios.post('https://free-horribly-perch.ngrok-free.app/process_security_analysis', dataWithNotifications)
          .catch(error => {
            console.error('Error sending alert notification:', error);
          });

        setProcessingVideo(false);
        
      } catch (error) {
        console.error('Upload failed:', error);
        setUploadingVideo(false);
        setProcessingVideo(false);
        alert('Uploaded Successfully');
      }
    };

    fileInput.click();
  };

  const handleTrackLocation = () => {
    navigate('/map');
  };

  useEffect(() => {
    if (videoRef.current && currentVideo) {
      videoRef.current.src = currentVideo;
      videoRef.current.play().catch(error => {
        console.error("Video play failed:", error);
      });
    }
  }, [currentVideo]);

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
              <BiVideo size={24} className="text-blue-400" />
              <span>Video Analytics</span>
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
                onClick={() => setSelectedTab('upload')}
                className={`px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  selectedTab === 'upload' 
                    ? 'bg-blue-700 text-white' 
                    : 'bg-blue-800 text-gray-200 hover:bg-blue-700'
                }`}
              >
                Upload
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

          {selectedTab === 'view' && (
            <div className="bg-white rounded-lg shadow-md p-4 mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">Live Camera Feed</h2>
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
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                  {processedVideo2 ? (
                    <video
                      className="w-full h-full object-cover"
                      src={processedVideo2}
                      autoPlay
                      muted
                      loop
                      playsInline
                    />
                  ) : (
                    <video
                      className="w-full h-full object-cover"
                      src="https://egdbvwtvwqhqorfknmfj.supabase.co/storage/v1/object/public/uploads//processed_intruder_16f75649-81e0-4238-9611-749c559de60a.mp4"
                      autoPlay
                      muted
                      loop
                      playsInline
                    />
                  )}
                  <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                    Camera 1 - Front Door
                  </div>
                </div>
                
                <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                  <video
                    className="w-full h-full object-cover"
                    src="https://egdbvwtvwqhqorfknmfj.supabase.co/storage/v1/object/public/uploads//processed_back2_40b08d6f-6b51-4f56-995b-fe92ecf2202e%20(1).mp4"
                    autoPlay
                    muted
                    loop
                    playsInline
                  />
                  <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                    Camera 2 - Backyard
                  </div>
                </div>
                
                <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                  <video
                    className="w-full h-full object-cover"
                    src="https://egdbvwtvwqhqorfknmfj.supabase.co/storage/v1/object/public/uploads//processed_garage_e1c54b23-70bf-4bb4-b87d-8a9301bee382.mp4"
                    autoPlay
                    muted
                    loop
                    playsInline
                  />
                  <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                    Camera 3 - Garage
                  </div>
                </div>
                
                <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                  <video
                    className="w-full h-full object-cover"
                    src="https://egdbvwtvwqhqorfknmfj.supabase.co/storage/v1/object/public/uploads//processed_side_bc2ee1f4-e6b6-43fe-8607-bee46ae7cd19.mp4"
                    autoPlay
                    muted
                    loop
                    playsInline
                  />
                  <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                    Camera 4 - Driveway
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedTab === 'upload' && (
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-semibold mb-4">Upload Video for Analysis</h2>
              
              {!currentVideo && !uploadingVideo && !processingVideo && (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 flex flex-col items-center justify-center">
                  <BsCloudUpload size={48} className="text-gray-400 mb-4" />
                  <p className="text-gray-600 mb-2">Drag and drop video files here or</p>
                  <div className="flex gap-2">
                    <form onSubmit={handleUploadVideo}>
                      <button
                        type="submit"
                        className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md"
                      >
                        Select Files
                      </button>
                    </form>
                  </div>
                  {location && (
                    <p className="text-sm text-gray-600 mt-2">
                      📍 Location: {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}
                    </p>
                  )}
                  <p className="text-xs text-gray-500 mt-2">
                    Supported formats: MP4, AVI, MOV (max file size: 500MB)
                  </p>
                </div>
              )}
              
              {uploadingVideo && (
                <div className="flex flex-col items-center justify-center p-8">
                  <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                  <p className="text-lg font-medium">Uploading video...</p>
                  <div className="w-64 bg-gray-200 rounded-full h-2.5 mt-4">
                    <div className="bg-blue-600 h-2.5 rounded-full w-2/3"></div>
                  </div>
                </div>
              )}
              
              {processingVideo && (
                <div className="flex flex-col items-center justify-center p-8">
                  <FaRobot size={48} className="text-blue-500 mb-4" />
                  <p className="text-lg font-medium">AI Processing video...</p>
                  <p className="text-sm text-gray-600 mb-4">
                    Analyzing for security threats and anomalies
                  </p>
                  <div className="w-64 bg-gray-200 rounded-full h-2.5">
                    <div className="bg-green-600 h-2.5 rounded-full animate-pulse w-full"></div>
                  </div>
                </div>
              )}
              
              {currentVideo && !uploadingVideo && !processingVideo && (
                <div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                      <video
                        className="w-full h-full object-contain"
                        src={processedVideo1}
                        controls
                        playsInline
                        autoPlay
                        loop
                      />
                      <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                        Processed Video 1
                      </div>
                    </div>
                    <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                      <video
                        className="w-full h-full object-contain"
                        src={processedVideo2}
                        controls
                        playsInline
                        autoPlay
                        loop
                      />
                      <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                        Processed Video 2
                      </div>
                    </div>
                  </div>
                  <div className="bg-gray-100 rounded-lg p-4 mb-4">
                    <h3 className="font-medium mb-2">AI Analysis Results</h3>
                    <div className="space-y-2">
                      <div className="flex items-center">
                        <span className="font-medium text-red-600">Analysis:</span>
                        <p className="ml-2 text-gray-700">{aiAnalysis}</p>
                      </div>
                    </div>
                  </div>
                  <div className="flex space-x-3">
                    <button className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg flex items-center justify-center space-x-2">
                      <BsBellFill size={18} />
                      <span>Send Alert</span>
                    </button>
                    <button className="flex-1 bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-lg flex items-center justify-center space-x-2">
                      <BsGraphUpArrow size={18} />
                      <span>Detailed Analysis</span>
                    </button>
                  </div>
                  
                  <div className="mt-4 mb-6">
                    <button
                      onClick={handleTrackLocation}
                      className="w-full flex items-center justify-center gap-2 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-md"
                      disabled={trackingLocation}
                    >
                      <MdLocationOn />
                      {trackingLocation ? 'Getting Location...' : 'Track Incident Location'}
                    </button>
                    {location && (
                      <p className="text-sm text-gray-600 mt-2 text-center">
                        📍 Location: {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}
                      </p>
                    )}
                  </div>
                  
                  <div className="mt-6">
                    <h3 className="text-xl font-semibold mb-4">Few of the Processed Frames</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {frames?.length > 0 ? (
                        frames.map((frame, index) => (
                          <div key={index} className="bg-gray-900 rounded-lg overflow-hidden">
                            <img
                              src={`data:image/jpeg;base64,${frame.image_data}`}
                              alt={`Frame ${frame.frame_num}`}
                              className="w-full h-full object-cover"
                            />
                            <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                              Frame {frame.frame_num}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="col-span-3 text-center text-gray-500">
                          No frames available
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {selectedTab === 'history' && (
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-semibold mb-4">Video History</h2>
              
              <div className="mb-4">
                <div className="flex space-x-3 mb-4">
                  <input
                    type="text"
                    placeholder="Search videos..."
                    className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md">
                    Search
                  </button>
                </div>
                
                <div className="flex space-x-2 mb-4 overflow-x-auto pb-2">
                  <button className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm">
                    All Videos
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Incidents
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Camera 1
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Camera 2
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Camera 3
                  </button>
                  <button className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-full text-sm">
                    Camera 4
                  </button>
                </div>
              </div>
              
              <div className="space-y-4">
                {[1, 2, 3, 4].map((item) => (
                  <div key={item} className="border border-gray-200 rounded-lg p-4 flex flex-col md:flex-row">
                    <div className="md:w-40 h-24 bg-gray-900 rounded-lg overflow-hidden mb-4 md:mb-0 md:mr-4 flex-shrink-0">
                      <video
                        src={`https://egdbvwtvwqhqorfknmfj.supabase.co/storage/v1/object/public/uploads//processed_back2_40b08d6f-6b51-4f56-995b-fe92ecf2202e%20(1).mp4`}
                        autoPlay
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex-grow">
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-medium">
                          {item === 1 ? 'Robbery Detection' : 
                           item === 2 ? 'Suspicious Activity' :
                           item === 3 ? 'Trespassing' : 'Motion Detection'}
                        </h3>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          item === 1 ? 'bg-red-100 text-red-800' : 
                          item === 2 ? 'bg-yellow-100 text-yellow-800' :
                          item === 3 ? 'bg-orange-100 text-orange-800' : 
                          'bg-blue-100 text-blue-800'
                        }`}>
                          {item === 1 ? 'Critical' : 
                           item === 2 ? 'Warning' :
                           item === 3 ? 'Alert' : 'Info'}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 mb-2">
                        <p>Camera {item} - {new Date().toLocaleDateString()}, {new Date().toLocaleTimeString()}</p>
                        <p className="mt-1">Duration: 2m 17s</p>
                      </div>
                      <div className="flex space-x-2">
                        <button className="text-blue-500 hover:text-blue-700 text-sm font-medium">
                          View
                        </button>
                        <button className="text-gray-500 hover:text-gray-700 text-sm">
                          Download
                        </button>
                        <button className="text-gray-500 hover:text-gray-700 text-sm">
                          Share
                        </button>
                      </div>
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
    </>
  );
}

export default VideoFeedPage;

