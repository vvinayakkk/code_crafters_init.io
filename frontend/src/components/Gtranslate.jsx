import React, { useEffect } from "react";

const GTranslate = () => {
  useEffect(() => {
    const addGoogleTranslateScript = () => {
      if (!document.querySelector("#google-translate-script")) {
        const script = document.createElement("script");
        script.id = "google-translate-script";
        script.src = "https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
        script.async = true;
        document.body.appendChild(script);
      }
    };

    window.googleTranslateElementInit = () => {
      new window.google.translate.TranslateElement(
        {
          pageLanguage: "en", // Default language
          includedLanguages: "en,hi,gu,mr,ta,fr,ja,de,ru,en,hi", // Add desired languages here
          layout: window.google.translate.TranslateElement.InlineLayout.SIMPLE,
        },
        "google_translate_element"
      );
    };

    addGoogleTranslateScript();
  }, []);

  return (
    <div
      id="google_translate_element"
      style={{
        position: 'fixed', // Changed from 'absolute' to 'fixed'
        bottom: '2rem',    // Added bottom instead of top
        right: '2rem',     // Increased right spacing
        opacity: 0.8,      // Increased opacity slightly
        zIndex: 1000,
        backgroundColor: 'white',
        borderRadius: '8px',
        padding: '0.5rem',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)', // Added shadow
        transition: 'opacity 0.2s',              // Added hover effect
        '&:hover': {
          opacity: 1
        }
      }}
    ></div>
  );
};

export default GTranslate;