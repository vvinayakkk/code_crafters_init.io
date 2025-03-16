
import React, { useEffect, useState, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const PoliceStationMap = () => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [userLocation, setUserLocation] = useState(null);
  const [policeStations, setPoliceStations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapboxApiKey, setMapboxApiKey] = useState(null);
  const [selectedStation, setSelectedStation] = useState(null);

  // Custom CSS for the police station marker and other UI elements
  const customCSS = `
    .police-marker {
      background-image: url('https://img.icons8.com/color/48/000000/police-badge.png');
      background-size: cover;
      width: 40px;
      height: 40px;
      border-radius: 50%;
      cursor: pointer;
      background-color: rgba(25, 55, 109, 0.8);
      border: 3px solid #ffffff;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
      transition: transform 0.2s ease;
    }
    
    .police-marker:hover {
      transform: scale(1.1);
    }
    
    .user-marker {
      background-color: #ffffff;
      border: 4px solid #0c2c64;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.5);
      position: relative;
    }
    
    .user-marker::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 10px;
      height: 10px;
      background-color: #4285f4;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0% { transform: translate(-50%, -50%) scale(0.5); opacity: 1; }
      100% { transform: translate(-50%, -50%) scale(2); opacity: 0; }
    }
    
    .mapboxgl-popup-content {
      background-color: #0c2c64;
      color: white;
      border-radius: 8px;
      padding: 12px 20px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .mapboxgl-popup-tip {
      border-top-color: #0c2c64 !important;
      border-bottom-color: #0c2c64 !important;
    }
    
    .mapboxgl-popup h3 {
      margin: 0 0 8px;
      color: #ffffff;
      font-size: 16px;
      font-weight: 600;
    }
    
    .mapboxgl-popup p {
      margin: 5px 0;
      color: #e0e6f1;
      font-size: 14px;
    }
    
    .mapboxgl-popup-close-button {
      color: white;
      font-size: 16px;
      padding: 5px;
    }
    
    .mapboxgl-ctrl-zoom-in, .mapboxgl-ctrl-zoom-out {
      background-color: #0c2c64 !important;
      color: white !important;
    }
    
    .station-card {
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .station-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    .emergency-button {
      animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-red {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
  `;

  // First effect - add CSS and get API key from env
  useEffect(() => {
    // Add custom CSS to document
    const style = document.createElement('style');
    style.textContent = customCSS;
    document.head.appendChild(style);

    // Get API key from environment variables
    const apiKey = import.meta.env.MAPBOX_API_KEY ||"pk.eyJ1IjoiYmhhbnVoYXJzIiwiYSI6ImNtOGI0ZHR2dTFxdmcya3NmMXR1ZnhrYnYifQ.tudlNhBcrIlyw6Ez8onolQ";
    
    if (apiKey) {
      setMapboxApiKey(apiKey);
    } else {
      console.error("No Mapbox API key found in environment variables");
      setError("Mapbox API key not found in environment variables. Check your .env file.");
      setLoading(false);
    }

    // Cleanup function to remove style when component unmounts
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // Second effect - initialize Mapbox and get user location once we have the API key
  useEffect(() => {
    if (!mapboxApiKey) return; // Don't proceed without API key
    
    // Set the Mapbox access token
    mapboxgl.accessToken = mapboxApiKey;
    
    // Get user location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setUserLocation({ lat: latitude, lng: longitude });
        },
        (err) => {
          console.log('Geolocation error:', err);
          setError('Unable to retrieve your location. Using default location.');
          // Default location as fallback (Mumbai area based on your screenshot)
          setUserLocation({ lat: 19.1287, lng: 72.9323 }); // Kanjurmarg area
        }
      );
    } else {
      setError('Geolocation is not supported by your browser. Using default location.');
      setUserLocation({ lat: 19.1287, lng: 72.9323 });
    }
  }, [mapboxApiKey]);

  // Third effect - initialize map once we have both API key and user location
  useEffect(() => {
    if (!userLocation || !mapboxApiKey) return;
    
    if (map.current) return; // Initialize map only once
    
    try {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/dark-v10', // Dark theme for better contrast with blue elements
        center: [userLocation.lng, userLocation.lat],
        zoom: 14
      });

      // Add user location marker
      const userMarkerElement = document.createElement('div');
      userMarkerElement.className = 'user-marker';
      
      const userMarker = new mapboxgl.Marker({ element: userMarkerElement })
        .setLngLat([userLocation.lng, userLocation.lat])
        .setPopup(new mapboxgl.Popup({ offset: 25 }).setHTML('<h3>Your Location</h3>'))
        .addTo(map.current);

      // Add navigation controls
      map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

      // Wait for map to load before fetching police stations
      map.current.on('load', () => {
        fetchPoliceStationsMultiMethod();
      });
    } catch (err) {
      console.error('Map initialization error:', err);
      setError(`Error initializing map: ${err.message}`);
      setLoading(false);
    }
  }, [userLocation, mapboxApiKey]);

  // Add markers for police stations
  useEffect(() => {
    if (!map.current || policeStations.length === 0) return;
    
    // Clear any existing markers first
    const existingMarkers = document.querySelectorAll('.police-marker');
    existingMarkers.forEach(marker => marker.remove());

    policeStations.forEach(station => {
      // Create a custom DOM element for the marker
      const el = document.createElement('div');
      el.className = 'police-marker';

      // Add the marker to the map
      const marker = new mapboxgl.Marker(el)
        .setLngLat([station.longitude, station.latitude])
        .setPopup(
          new mapboxgl.Popup({ offset: 25, closeOnClick: false })
            .setHTML(`
              <h3>${station.name}</h3>
              <p>${station.address || 'Address not available'}</p>
              <p><strong>Distance:</strong> ${station.distance.toFixed(2)} km</p>
              <p><strong>Est. Time:</strong> ${Math.round(station.distance * 2)} mins by car</p>
            `)
        )
        .addTo(map.current);
        
      // Add click event to sync with station list
      el.addEventListener('click', () => {
        setSelectedStation(station.id);
        
        // Scroll the station into view in the list
        setTimeout(() => {
          const element = document.getElementById(`station-${station.id}`);
          if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }, 100);
      });
    });

    // Adjust map bounds to fit all markers if we have more than one station
    if (policeStations.length > 1) {
      const bounds = new mapboxgl.LngLatBounds();
      
      // Add user location to bounds
      bounds.extend([userLocation.lng, userLocation.lat]);
      
      // Add all station markers to bounds
      policeStations.forEach(station => {
        bounds.extend([station.longitude, station.latitude]);
      });
      
      // Fit map to these bounds with padding
      map.current.fitBounds(bounds, {
        padding: 50,
        maxZoom: 15
      });
    }

    setLoading(false);
  }, [policeStations]);

  // Try multiple methods to get police stations
  const fetchPoliceStationsMultiMethod = async () => {
    try {
      // For Mumbai area (based on your screenshot), include these known police stations
      const mumbaiPoliceStations = [
        {
          id: "ps-1",
          name: "Police Station Road",
          address: "Police Station Road, 400042, Kanjurmarg East, Mumbai, Maharashtra, India",
          latitude: 19.1287,
          longitude: 72.9323,
          distance: 0.1,
          phone: "022-2579 4321"
        },
        {
          id: "ps-2",
          name: "Kanjurmarg Police Station",
          address: "Kanjurmarg, Mumbai, Maharashtra, India",
          latitude: 19.1293,
          longitude: 72.9327,
          distance: 0.2,
          phone: "022-2579 2100"
        },
        {
          id: "ps-3",
          name: "Powai Police Station",
          address: "Powai, Mumbai, Maharashtra, India",
          latitude: 19.1191,
          longitude: 72.9043,
          distance: 3.1,
          phone: "022-2570 4252"
        },
        {
          id: "ps-4",
          name: "Bhandup Police Station",
          address: "Bhandup West, Mumbai, Maharashtra, India",
          latitude: 19.1462,
          longitude: 72.9371,
          distance: 2.0,
          phone: "022-2566 7171"
        },
        {
          id: "ps-5",
          name: "Parksite Police Station",
          address: "Vikhroli West, Mumbai, Maharashtra, India",
          latitude: 19.1105,
          longitude: 72.9276,
          distance: 2.3,
          phone: "022-2522 3678"
        }
      ];

      // First method: Try Mapbox Geocoding API
      const response = await fetch(
        `https://api.mapbox.com/geocoding/v5/mapbox.places/police%20station.json?proximity=${userLocation.lng},${userLocation.lat}&limit=5&access_token=${mapboxApiKey}`
      );
      
      const data = await response.json();
      
      // Check if we got valid results
      let apiStations = [];
      if (data.features && data.features.length > 0) {
        // Transform the response to our format
        apiStations = data.features.map(feature => {
          // Calculate rough distance in km
          const distance = calculateDistance(
            userLocation.lat, 
            userLocation.lng, 
            feature.center[1], 
            feature.center[0]
          );
          
          return {
            id: feature.id,
            name: feature.text || 'Police Station',
            address: feature.place_name,
            latitude: feature.center[1],
            longitude: feature.center[0],
            distance: distance,
            phone: "100" // Emergency police number in India
          };
        });
      }
      
      // Second method: Also try POI search for more results
      const poiResponse = await fetch(
        `https://api.mapbox.com/geocoding/v5/mapbox.places/police.json?proximity=${userLocation.lng},${userLocation.lat}&types=poi&limit=5&access_token=${mapboxApiKey}`
      );
      
      const poiData = await poiResponse.json();
      
      // Check if we got valid POI results
      let poiStations = [];
      if (poiData.features && poiData.features.length > 0) {
        // Transform the response to our format
        poiStations = poiData.features.map(feature => {
          // Calculate rough distance in km
          const distance = calculateDistance(
            userLocation.lat, 
            userLocation.lng, 
            feature.center[1], 
            feature.center[0]
          );
          
          return {
            id: `poi-${feature.id}`,
            name: feature.text || 'Police POI',
            address: feature.place_name,
            latitude: feature.center[1],
            longitude: feature.center[0],
            distance: distance,
            phone: "100" // Emergency police number in India
          };
        });
      }
      
      // Combine all sources of police stations
      let allStations = [...mumbaiPoliceStations, ...apiStations, ...poiStations];
      
      // Filter out duplicates (based on proximity)
      const uniqueStations = [];
      const seen = new Set();
      
      allStations.forEach(station => {
        // Create a key based on rounded coordinates (to find stations that are very close to each other)
        const key = `${station.latitude.toFixed(3)}-${station.longitude.toFixed(3)}`;
        if (!seen.has(key)) {
          seen.add(key);
          uniqueStations.push(station);
        }
      });
      
      // Sort by distance
      // Sort by distance
      uniqueStations.sort((a, b) => a.distance - b.distance);
      
      // Take the closest stations (up to 5)
      const nearbyStations = uniqueStations.slice(0, 5);
      
      if (nearbyStations.length > 0) {
        setPoliceStations(nearbyStations);
      } else {
        // Fallback to Mumbai stations only if we found nothing
        setPoliceStations(mumbaiPoliceStations);
      }
    } catch (err) {
      console.error('Error fetching police stations:', err);
      // Mumbai police stations as fallback
      const mumbaiPoliceStations = [
        {
          id: "ps-1",
          name: "Police Station Road",
          address: "Police Station Road, 400042, Kanjurmarg East, Mumbai, Maharashtra, India",
          latitude: 19.1287,
          longitude: 72.9323,
          distance: 0.1,
          phone: "022-2579 4321"
        },
        {
          id: "ps-2",
          name: "Kanjurmarg Police Station",
          address: "Kanjurmarg, Mumbai, Maharashtra, India",
          latitude: 19.1293,
          longitude: 72.9327,
          distance: 0.2,
          phone: "022-2579 2100"
        }
      ];
      setPoliceStations(mumbaiPoliceStations);
      setError('Warning: Using fallback police station data');
      setLoading(false);
    }
  };

  // Haversine formula to calculate distance between two points in km
  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371; // Radius of the Earth in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // Function to get directions to a police station
  const getDirections = (station) => {
    if (!userLocation) return;
    
    // Open Google Maps directions in a new tab
    const url = `https://www.google.com/maps/dir/?api=1&origin=${userLocation.lat},${userLocation.lng}&destination=${station.latitude},${station.longitude}&travelmode=driving`;
    window.open(url, '_blank');
  };

  // Function to simulate calling a police station
  const callStation = (phone) => {
    window.location.href = `tel:${phone}`;
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-900 to-blue-800 text-white shadow-lg">
        <div className="container mx-auto px-4 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-white rounded-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-blue-900" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 2a4 4 0 00-4 4v1H5a1 1 0 00-.994.89l-1 9A1 1 0 004 18h12a1 1 0 00.994-1.11l-1-9A1 1 0 0015 7h-1V6a4 4 0 00-4-4zm2 5V6a2 2 0 10-4 0v1h4zm-6 3a1 1 0 112 0 1 1 0 01-2 0zm7-1a1 1 0 100 2 1 1 0 000-2z" clipRule="evenodd" />
                </svg>
              </div>
              <h1 className="text-2xl font-bold">Police Station Finder</h1>
            </div>
            
            <button 
              className="emergency-button bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-6 rounded-full shadow-md flex items-center space-x-2 transition-all"
              onClick={() => callStation("100")}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z" />
              </svg>
              <span>Emergency Call</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-6">
        {/* Error and Loading Messages */}
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded shadow-md" role="alert">
            <div className="flex items-center">
              <svg className="h-6 w-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <p>{error}</p>
            </div>
          </div>
        )}
        
        {loading && (
          <div className="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-6 rounded shadow-md flex items-center">
            <svg className="animate-spin h-5 w-5 mr-3 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            {!mapboxApiKey ? "Loading API key..." : "Loading map and nearby police stations..."}
          </div>
        )}

        {/* Map and Stations Container */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Map Container */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
              <div ref={mapContainer} className="w-full h-96 lg:h-[600px]" />
            </div>
          </div>

          {/* Station List */}
          <div>
            <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
              <div className="bg-blue-900 text-white p-4">
                <h2 className="text-xl font-bold flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Nearby Police Stations
                </h2>
              </div>
              
              {!loading && policeStations.length > 0 ? (
                <div className="divide-y divide-gray-200 max-h-96 lg:max-h-[500px] overflow-y-auto">
                  {policeStations.map(station => (
                    <div 
                      key={station.id} 
                      id={`station-${station.id}`}
                      className={`p-4 station-card ${selectedStation === station.id ? 'bg-blue-50' : ''}`}
                      onClick={() => setSelectedStation(station.id)}
                    >
                      <div className="flex items-start">
                        <div className="bg-blue-900 rounded-full p-3 mr-3">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <h3 className="font-bold text-lg text-blue-900">{station.name}</h3>
                          <p className="text-gray-600 text-sm mb-2">{station.address}</p>
                          <div className="flex items-center mb-2">
                            <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                              {station.distance.toFixed(2)} km
                            </span>
                            <span className="mx-2 text-gray-400">•</span>
                            <span className="text-gray-600 text-sm">
                              ~{Math.round(station.distance * 2)} min drive
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex space-x-2 mt-3">
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            callStation(station.phone);
                          }}
                          className="flex-1 bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-lg text-sm font-medium flex items-center justify-center"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z" />
                          </svg>
                          Call
                        </button>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            getDirections(station);
                          }}
                          className="flex-1 bg-blue-800 hover:bg-blue-900 text-white py-2 px-4 rounded-lg text-sm font-medium flex items-center justify-center"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10.293 15.707a1 1 0 010-1.414L14.586 10l-4.293-4.293a1 1 0 111.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z" clipRule="evenodd" />
                            <path fillRule="evenodd" d="M4.293 15.707a1 1 0 010-1.414L8.586 10 4.293 5.707a1 1 0 011.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z" clipRule="evenodd" />
                          </svg>
                          Directions
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : !loading && (
                <div className="p-8 text-center text-gray-500">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto mb-3 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <p>No police stations found nearby.</p>
                  <p className="text-sm mt-1">Please try again or check your location settings.</p>
                </div>
              )}
            </div>
            
            {/* Emergency Info Card */}
            <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200 mt-6">
              <div className="bg-red-600 text-white p-4">
                <h2 className="text-xl font-bold flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Emergency Contacts
                </h2>
              </div>
              <div className="p-4">
                <div className="flex items-center mb-4">
                  <div className="bg-red-100 text-red-600 p-2 rounded-full mr-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium">Police Emergency</p>
                    <a href="tel:100" className="text-lg font-bold text-blue-800">100</a>
                  </div>
                </div>
                <div className="flex items-center mb-4">
                  <div className="bg-red-100 text-red-600 p-2 rounded-full mr-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium">Medical Emergency</p>
                    <a href="tel:108" className="text-lg font-bold text-blue-800">108</a>
                  </div>
                </div>
                <div className="flex items-center">
                  <div className="bg-red-100 text-red-600 p-2 rounded-full mr-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium">Fire Emergency</p>
                    <a href="tel:101" className="text-lg font-bold text-blue-800">101</a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-4 mt-6">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-gray-400">
              &copy; {new Date().getFullYear()} Police Station Finder. All rights reserved.
            </p>
            <div className="flex space-x-4 mt-2 md:mt-0">
              <a href="#privacy" className="text-sm text-gray-400 hover:text-white">Privacy Policy</a>
              <a href="#terms" className="text-sm text-gray-400 hover:text-white">Terms of Service</a>
              <a href="#about" className="text-sm text-gray-400 hover:text-white">About</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PoliceStationMap;
