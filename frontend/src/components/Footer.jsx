import React from 'react'
import { Link } from 'react-router-dom';

const Footer = () => (
    <footer className="bg-gray-800 text-white py-11 pb-32">
      <div className="container mx-auto px-4 grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
          <h3 className="text-xl font-bold mb-4">REC Market</h3>
          <p className="text-gray-400">Transforming renewable energy trading through blockchain technology</p>
        </div>
        <div>
          <h4 className="font-bold mb-4">Quick Links</h4>
          <ul className="space-y-2 text-gray-400">
            <li><Link to="/about">About</Link></li>
            <li><Link to="/contact">Contact</Link></li>
          </ul>
        </div>
        <div>
          <h4 className="font-bold mb-4">Connect</h4>
          <div className="flex space-x-4">
            {/* Social media icons */}
          </div>
        </div>
      </div>
    </footer>
  );

export default Footer