import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const ThreeDSensorMap = ({ sensorData }) => {
  const containerRef = useRef(null);
  const requestRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const nodeRefs = useRef({});
  const textRefs = useRef({});
  const nodeScales = useRef({});

  // Default sensor data if none is provided
  const defaultSensorData = {
    temperature: 24,
    humidity: 45,
    motion: false,
    door: false,
    window: false
  };

  // Use provided sensor data or fallback to default
  const data = sensorData || defaultSensorData;

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // Initialize camera
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(20, 20, 20); // Increased camera distance
    cameraRef.current = camera;

    // Initialize renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Add orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Add light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    scene.add(directionalLight);

    // Create a house
    createHouse(scene);

    // Create sensor nodes
    createSensorNodes(scene, data);

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current) return;
      
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    // Animation loop
    const animate = () => {
      requestRef.current = requestAnimationFrame(animate);
      
      // Rotate house slightly
      if (scene.getObjectByName('house')) {
        scene.getObjectByName('house').rotation.y += 0.002;
      }

      // Pulse sensor nodes
      pulseSensorNodes();
      
      // Update controls
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      
      // Render
      renderer.render(scene, camera);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(requestRef.current);
      window.removeEventListener('resize', handleResize);
      
      if (containerRef.current && rendererRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
      
      // Clean up objects
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(material => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
    };
  }, []);

  // Update sensor nodes when data changes
  useEffect(() => {
    if (!sceneRef.current) return;
    updateSensorNodes(data);
  }, [data]);

  const createHouse = (scene) => {
    const houseGroup = new THREE.Group();
    houseGroup.name = 'house';
    
    // Floor - increased size
    const floorGeometry = new THREE.BoxGeometry(18, 0.3, 12);
    const floorMaterial = new THREE.MeshStandardMaterial({ color: 0xa9a9a9 });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.position.y = -0.15;
    floor.receiveShadow = true;
    houseGroup.add(floor);

    // Walls - adjust all wall dimensions by multiplying existing dimensions by 2
    const wallMaterial = new THREE.MeshStandardMaterial({ color: 0xe8e8e8 });
    
    // Back wall
    const backWallGeometry = new THREE.BoxGeometry(18, 6, 0.3);
    const backWall = new THREE.Mesh(backWallGeometry, wallMaterial);
    backWall.position.set(0, 3, -6);
    backWall.castShadow = true;
    backWall.receiveShadow = true;
    houseGroup.add(backWall);
    
    // Front wall (with gap for door)
    const frontWallLeft = new THREE.Mesh(
      new THREE.BoxGeometry(6.9, 6, 0.3),
      wallMaterial
    );
    frontWallLeft.position.set(-5.55, 3, 6);
    frontWallLeft.castShadow = true;
    frontWallLeft.receiveShadow = true;
    houseGroup.add(frontWallLeft);
    
    const frontWallRight = new THREE.Mesh(
      new THREE.BoxGeometry(6.9, 6, 0.3),
      wallMaterial
    );
    frontWallRight.position.set(5.55, 3, 6);
    frontWallRight.castShadow = true;
    frontWallRight.receiveShadow = true;
    houseGroup.add(frontWallRight);
    
    const topFrontWall = new THREE.Mesh(
      new THREE.BoxGeometry(4.2, 1.5, 0.3),
      wallMaterial
    );
    topFrontWall.position.set(0, 5.25, 6);
    topFrontWall.castShadow = true;
    topFrontWall.receiveShadow = true;
    houseGroup.add(topFrontWall);
    
    // Left wall (with window)
    const leftWallLeft = new THREE.Mesh(
      new THREE.BoxGeometry(0.3, 6, 3.9),
      wallMaterial
    );
    leftWallLeft.position.set(-9, 3, -4.05);
    leftWallLeft.castShadow = true;
    leftWallLeft.receiveShadow = true;
    houseGroup.add(leftWallLeft);
    
    const leftWallRight = new THREE.Mesh(
      new THREE.BoxGeometry(0.3, 6, 3.9),
      wallMaterial
    );
    leftWallRight.position.set(-9, 3, 4.05);
    leftWallRight.castShadow = true;
    leftWallRight.receiveShadow = true;
    houseGroup.add(leftWallRight);
    
    const leftWallTop = new THREE.Mesh(
      new THREE.BoxGeometry(0.3, 1.5, 4.2),
      wallMaterial
    );
    leftWallTop.position.set(-9, 5.25, 0);
    leftWallTop.castShadow = true;
    leftWallTop.receiveShadow = true;
    houseGroup.add(leftWallTop);
    
    const leftWallBottom = new THREE.Mesh(
      new THREE.BoxGeometry(0.3, 1.5, 4.2),
      wallMaterial
    );
    leftWallBottom.position.set(-9, 0.75, 0);
    leftWallBottom.castShadow = true;
    leftWallBottom.receiveShadow = true;
    houseGroup.add(leftWallBottom);
    
    // Right wall - depth increased from 4 to 6
    const rightWall = new THREE.Mesh(
      new THREE.BoxGeometry(0.3, 6, 12),
      wallMaterial
    );
    rightWall.position.set(9, 3, 0);
    rightWall.castShadow = true;
    rightWall.receiveShadow = true;
    houseGroup.add(rightWall);
    
    // Roof - increased from 4.5x2 to 6.75x3
    const roofGeometry = new THREE.ConeGeometry(13.5, 6, 4);
    const roofMaterial = new THREE.MeshStandardMaterial({ color: 0x8b4513 });
    const roof = new THREE.Mesh(roofGeometry, roofMaterial);
    roof.rotation.y = Math.PI / 4;
    roof.position.y = 7.5;
    roof.castShadow = true;
    houseGroup.add(roof);
    
    // Door - increased from 1.4x1.8 to 2.1x2.7
    const doorGeometry = new THREE.BoxGeometry(4.2, 5.4, 0.15);
    const doorMaterial = new THREE.MeshStandardMaterial({ color: 0x8b4513 });
    const door = new THREE.Mesh(doorGeometry, doorMaterial);
    door.position.set(0, 2.7, 6.15);
    door.castShadow = true;
    door.name = 'door';
    houseGroup.add(door);
    
    // Window on left wall - increased from 0.1x1x1.4 to 0.15x1.5x2.1
    const windowGeometry = new THREE.BoxGeometry(0.3, 3, 4.2);
    const windowMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x87ceeb,
      transparent: true,
      opacity: 0.7
    });
    const window = new THREE.Mesh(windowGeometry, windowMaterial);
    window.position.set(-9.15, 3, 0);
    window.name = 'window';
    houseGroup.add(window);
    
    // Window frame - adjusted to match larger window
    const windowFrameGeometry = new THREE.BoxGeometry(0.36, 3.3, 4.5);
    const windowFrameMaterial = new THREE.MeshStandardMaterial({ color: 0x8b4513 });
    const windowFrame = new THREE.Mesh(windowFrameGeometry, windowFrameMaterial);
    windowFrame.position.set(-9.18, 3, 0);
    houseGroup.add(windowFrame);
    
    // Interior wall - depth increased from 4 to 6
    const interiorWall = new THREE.Mesh(
      new THREE.BoxGeometry(0.3, 6, 12),
      wallMaterial
    );
    interiorWall.position.set(0, 3, 0);
    interiorWall.castShadow = true;
    interiorWall.receiveShadow = true;
    houseGroup.add(interiorWall);
    
    // Interior door - increased from 0.1x1.8x1 to 0.15x2.7x1.5
    const interiorDoorGeometry = new THREE.BoxGeometry(0.3, 5.4, 3);
    const interiorDoorMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x8b4513,
      transparent: true,
      opacity: 0.7
    });
    const interiorDoor = new THREE.Mesh(interiorDoorGeometry, interiorDoorMaterial);
    interiorDoor.position.set(0, 2.7, 0);
    houseGroup.add(interiorDoor);
    
    scene.add(houseGroup);
  };

  const createSensorNodes = (scene, data) => {
    // Create sensor node groups
    const createSensorNode = (name, position, color, value) => {
      const nodeGroup = new THREE.Group();
      nodeGroup.name = name;
      nodeGroup.position.copy(position);
      
      // Increased sphere size
      const sphereGeometry = new THREE.SphereGeometry(0.4, 32, 32);
      const sphereMaterial = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.7,
        transparent: true,
        opacity: 0.9
      });
      const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      sphere.castShadow = true;
      nodeGroup.add(sphere);
      
      // Increased text size and positioning
      const valueText = createTextLabel(value, 0.5);
      valueText.position.y = 0.8;
      nodeGroup.add(valueText);
      
      const nameText = createTextLabel(name, 0.3);
      nameText.position.y = 1.4;
      nodeGroup.add(nameText);
      
      // Store references to update later
      nodeRefs.current[name] = sphere;
      textRefs.current[name] = valueText;
      nodeScales.current[name] = { 
        scale: 1, 
        growing: true,
        baseSize: 0.4,
        minScale: 0.9,
        maxScale: 1.3,
        speed: 0.015
      };
      
      scene.add(nodeGroup);
      return nodeGroup;
    };
    
    // Create the sensor nodes with adjusted positions for larger house
    createSensorNode('temperature', new THREE.Vector3(-6, 3, -3), 0xff4444, data.temperature + '°C');
    createSensorNode('humidity', new THREE.Vector3(-6, 3, 3), 0x88ccff, data.humidity + '%');
    createSensorNode('motion', new THREE.Vector3(6, 3, -3), data.motion ? 0xff4444 : 0x88ccff, data.motion ? 'Active' : 'None');
    createSensorNode('door', new THREE.Vector3(0, 0.9, 6.6), data.door ? 0xff4444 : 0x88ccff, data.door ? 'Open' : 'Closed');
    createSensorNode('window', new THREE.Vector3(-9.6, 3, 0), data.window ? 0xff4444 : 0x88ccff, data.window ? 'Open' : 'Closed');
  };

  const updateSensorNodes = (data) => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;
    
    // Update temperature node
    const tempNode = scene.getObjectByName('temperature');
    if (tempNode && textRefs.current.temperature) {
      textRefs.current.temperature.element.textContent = data.temperature + '°C';
    }
    
    // Update humidity node
    const humidityNode = scene.getObjectByName('humidity');
    if (humidityNode && textRefs.current.humidity) {
      textRefs.current.humidity.element.textContent = data.humidity + '%';
    }
    
    // Update motion node
    const motionNode = scene.getObjectByName('motion');
    if (motionNode && textRefs.current.motion && nodeRefs.current.motion) {
      textRefs.current.motion.element.textContent = data.motion ? 'Active' : 'None';
      nodeRefs.current.motion.material.color.set(data.motion ? 0xff4444 : 0x88ccff);
      nodeRefs.current.motion.material.emissive.set(data.motion ? 0xff4444 : 0x88ccff);
    }
    
    // Update door node
    const doorNode = scene.getObjectByName('door');
    if (doorNode && textRefs.current.door && nodeRefs.current.door) {
      textRefs.current.door.element.textContent = data.door ? 'Open' : 'Closed';
      nodeRefs.current.door.material.color.set(data.door ? 0xff4444 : 0x88ccff);
      nodeRefs.current.door.material.emissive.set(data.door ? 0xff4444 : 0x88ccff);
      
      // Animate door
      const door = scene.getObjectByName('house').getObjectByName('door');
      if (door) {
        door.rotation.y = data.door ? Math.PI / 4 : 0;
      }
    }
    
    // Update window node
    const windowNode = scene.getObjectByName('window');
    if (windowNode && textRefs.current.window && nodeRefs.current.window) {
      textRefs.current.window.element.textContent = data.window ? 'Open' : 'Closed';
      nodeRefs.current.window.material.color.set(data.window ? 0xff4444 : 0x88ccff);
      nodeRefs.current.window.material.emissive.set(data.window ? 0xff4444 : 0x88ccff);
      
      // Animate window
      const window = scene.getObjectByName('house').getObjectByName('window');
      if (window) {
        window.material.opacity = data.window ? 0.2 : 0.7;
      }
    }
  };

  // Function to create 2D text labels in 3D space
  const createTextLabel = (text, size) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 1024; // Doubled canvas size
    canvas.height = 512;
    
    // Background
    context.fillStyle = 'rgba(50, 50, 50, 0.8)'; // Increased opacity
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    // Text
    context.fillStyle = 'white';
    context.font = 'bold 96px Arial'; // Increased font size and made bold
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.strokeStyle = 'black';
    context.lineWidth = 4;
    context.strokeText(text, canvas.width / 2, canvas.height / 2);
    context.fillText(text, canvas.width / 2, canvas.height / 2);
    
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ 
      map: texture,
      transparent: true,
      opacity: 0.95
    });
    
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(size * 10, size * 5, 1); // Increased scale values
    
    sprite.element = context;
    sprite.canvas = canvas;
    
    return sprite;
  };

  // Function to animate sensor nodes pulsing
  const pulseSensorNodes = () => {
    Object.keys(nodeRefs.current).forEach(key => {
      const node = nodeRefs.current[key];
      const scaleInfo = nodeScales.current[key];
      
      if (!node || !scaleInfo) return;
      
      // Update scale based on direction
      if (scaleInfo.growing) {
        scaleInfo.scale += scaleInfo.speed;
        if (scaleInfo.scale >= scaleInfo.maxScale) {
          scaleInfo.growing = false;
        }
      } else {
        scaleInfo.scale -= scaleInfo.speed;
        if (scaleInfo.scale <= scaleInfo.minScale) {
          scaleInfo.growing = true;
        }
      }
      
      // Apply scale
      node.scale.set(scaleInfo.scale, scaleInfo.scale, scaleInfo.scale);
    });
  };

  return (
    <div 
      ref={containerRef} 
      style={{ 
        width: '100%', 
        height: '600px', // Increased container height
        borderRadius: '12px',
        overflow: 'hidden',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
      }}
    />
  );
};

export default ThreeDSensorMap;