// Content script injected into web pages
// Handles displaying summary results and loading states

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'ping') {
    sendResponse({ status: 'ready' });
  } else if (request.action === 'showLoading') {
    showLoadingIndicator();
  } else if (request.action === 'showSummary') {
    hideLoadingIndicator();
    displaySummary(request.data);
  } else if (request.action === 'showError') {
    hideLoadingIndicator();
    displayError(request.error);
  }
});

// Show loading indicator
function showLoadingIndicator() {
  // Remove existing indicator if any
  hideLoadingIndicator();

  const loader = document.createElement('div');
  loader.id = 'nlp-summarizer-loader';
  loader.innerHTML = `
    <div style="
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 30px 40px;
      border-radius: 12px;
      z-index: 999999;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      text-align: center;
    ">
      <div style="
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top-color: #4CAF50;
        border-radius: 50%;
        margin: 0 auto 15px;
        animation: spin 1s linear infinite;
      "></div>
      <div style="font-size: 16px; font-weight: 500;">
        Summarizing text...
      </div>
    </div>
    <style>
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
    </style>
  `;
  document.body.appendChild(loader);
}

// Hide loading indicator
function hideLoadingIndicator() {
  const loader = document.getElementById('nlp-summarizer-loader');
  if (loader) {
    loader.remove();
  }
}

// Display summary in a modal
function displaySummary(data) {
  // Remove existing modal if any
  removeSummaryModal();

  const modal = document.createElement('div');
  modal.id = 'nlp-summarizer-modal';
  
  const { summary, method, stats } = data;
  
  modal.innerHTML = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      z-index: 999998;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    " id="modal-overlay">
      <div style="
        background: white;
        border-radius: 16px;
        max-width: 700px;
        width: 90%;
        max-height: 80vh;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
      ">
        <!-- Header -->
        <div style="
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px 25px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        ">
          <div>
            <h2 style="margin: 0; font-size: 20px; font-weight: 600;">
              📝 Text Summary
            </h2>
            <p style="margin: 5px 0 0 0; font-size: 13px; opacity: 0.9;">
              Method: ${method.charAt(0).toUpperCase() + method.slice(1)}
            </p>
          </div>
          <button id="close-summary" style="
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            font-size: 24px;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
          " onmouseover="this.style.background='rgba(255, 255, 255, 0.3)'"
             onmouseout="this.style.background='rgba(255, 255, 255, 0.2)'">
            ×
          </button>
        </div>

        <!-- Stats Bar -->
        <div style="
          background: #f8f9fa;
          padding: 12px 25px;
          display: flex;
          gap: 20px;
          font-size: 13px;
          border-bottom: 1px solid #e9ecef;
        ">
          <span>
            <strong>Original:</strong> ${stats.original_words} words
          </span>
          <span>
            <strong>Summary:</strong> ${stats.summary_words} words
          </span>
          <span style="color: #28a745;">
            <strong>Reduced:</strong> ${stats.compression_ratio}%
          </span>
        </div>

        <!-- Summary Content -->
        <div style="
          padding: 25px;
          overflow-y: auto;
          flex: 1;
        ">
          <p style="
            margin: 0;
            line-height: 1.8;
            font-size: 15px;
            color: #333;
          ">${summary}</p>
        </div>

        <!-- Footer with Actions -->
        <div style="
          padding: 15px 25px;
          border-top: 1px solid #e9ecef;
          display: flex;
          gap: 10px;
          justify-content: flex-end;
        ">
          <button id="read-aloud" style="
            background: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 5px;
          " onmouseover="this.style.background='#1976D2'"
             onmouseout="this.style.background='#2196F3'">
            🔊 Read Aloud
          </button>
          <button id="copy-summary" style="
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
          " onmouseover="this.style.background='#45a049'"
             onmouseout="this.style.background='#4CAF50'">
            📋 Copy
          </button>
          <button id="close-summary-btn" style="
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
          " onmouseover="this.style.background='#5a6268'"
             onmouseout="this.style.background='#6c757d'">
            Close
          </button>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  // Event listeners
  document.getElementById('close-summary').addEventListener('click', removeSummaryModal);
  document.getElementById('close-summary-btn').addEventListener('click', removeSummaryModal);
  document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') {
      removeSummaryModal();
    }
  });

  document.getElementById('copy-summary').addEventListener('click', () => {
    navigator.clipboard.writeText(summary).then(() => {
      const btn = document.getElementById('copy-summary');
      const originalText = btn.innerHTML;
      btn.innerHTML = '✓ Copied!';
      btn.style.background = '#28a745';
      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.style.background = '#4CAF50';
      }, 2000);
    });
  });

  // Read Aloud functionality
  let speechSynthesis = window.speechSynthesis;
  let currentUtterance = null;
  let isSpeaking = false;

  document.getElementById('read-aloud').addEventListener('click', function() {
    const summaryText = summary; // Use the summary variable from the function scope
    const readBtn = document.getElementById('read-aloud');

    if (isSpeaking) {
      // Stop speaking
      speechSynthesis.cancel();
      isSpeaking = false;
      readBtn.innerHTML = '🔊 Read Aloud';
      readBtn.style.background = '#2196F3';
    } else {
      // Start speaking
      if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
      }

      currentUtterance = new SpeechSynthesisUtterance(summaryText);

      // Configure speech settings
      currentUtterance.rate = 0.9; // Slightly slower for clarity
      currentUtterance.pitch = 1;
      currentUtterance.volume = 1;

      // Set voice (prefer English voices)
      const voices = speechSynthesis.getVoices();
      const englishVoice = voices.find(voice =>
        voice.lang.startsWith('en') && voice.name.includes('Female')
      ) || voices.find(voice => voice.lang.startsWith('en')) || voices[0];

      if (englishVoice) {
        currentUtterance.voice = englishVoice;
      }

      // Handle speech events
      currentUtterance.onstart = function() {
        isSpeaking = true;
        readBtn.innerHTML = '⏸️ Stop Reading';
        readBtn.style.background = '#FF9800';
      };

      currentUtterance.onend = function() {
        isSpeaking = false;
        readBtn.innerHTML = '🔊 Read Aloud';
        readBtn.style.background = '#2196F3';
      };

      currentUtterance.onerror = function() {
        isSpeaking = false;
        readBtn.innerHTML = '🔊 Read Aloud';
        readBtn.style.background = '#2196F3';
        console.error('Speech synthesis error');
      };

      speechSynthesis.speak(currentUtterance);
    }
  });

  // Close on Escape key
  document.addEventListener('keydown', function escapeHandler(e) {
    if (e.key === 'Escape') {
      removeSummaryModal();
      document.removeEventListener('keydown', escapeHandler);
    }
  });
}

// Display error message
function displayError(errorMessage) {
  // Remove existing modal if any
  removeSummaryModal();

  const modal = document.createElement('div');
  modal.id = 'nlp-summarizer-modal';
  
  modal.innerHTML = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      z-index: 999998;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    " id="modal-overlay">
      <div style="
        background: white;
        border-radius: 16px;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
      ">
        <div style="
          background: linear-gradient(135deg, #f44336 0%, #e91e63 100%);
          color: white;
          padding: 20px 25px;
          border-radius: 16px 16px 0 0;
        ">
          <h2 style="margin: 0; font-size: 20px; font-weight: 600;">
            ❌ Error
          </h2>
        </div>

        <div style="padding: 25px;">
          <p style="margin: 0 0 15px 0; line-height: 1.6; color: #333;">
            ${errorMessage}
          </p>
          <p style="margin: 0; font-size: 13px; color: #666; line-height: 1.6;">
            <strong>To start the server:</strong><br>
            1. Open terminal in project folder<br>
            2. Activate venv: <code style="background: #f5f5f5; padding: 2px 6px; border-radius: 3px;">venv\\Scripts\\activate</code><br>
            3. Run: <code style="background: #f5f5f5; padding: 2px 6px; border-radius: 3px;">python api_server.py</code>
          </p>
        </div>

        <div style="
          padding: 15px 25px;
          border-top: 1px solid #e9ecef;
          text-align: right;
        ">
          <button id="close-error" style="
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
          ">
            Close
          </button>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  document.getElementById('close-error').addEventListener('click', removeSummaryModal);
  document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') {
      removeSummaryModal();
    }
  });
}

// Remove summary modal
function removeSummaryModal() {
  // Stop speech synthesis if active
  if (window.speechSynthesis && window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }

  const modal = document.getElementById('nlp-summarizer-modal');
  if (modal) {
    modal.remove();
  }
}

console.log('✅ Text Summarizer content script loaded');
