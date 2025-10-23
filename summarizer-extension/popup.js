// Popup script for extension settings and status

const API_URL = 'http://localhost:5000';

// Check server status on popup open
document.addEventListener('DOMContentLoaded', () => {
  checkServerStatus();
  loadSettings();
  setupEventListeners();
});

// Check if API server is running
async function checkServerStatus() {
  const statusBar = document.getElementById('statusBar');
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');

  try {
    const response = await fetch(`${API_URL}/health`);
    const data = await response.json();

    if (response.ok && data.status === 'running') {
      statusBar.className = 'status-bar online';
      statusIndicator.className = 'status-indicator online';
      statusText.textContent = '✓ Server is running';
    } else {
      throw new Error('Server not responding correctly');
    }
  } catch (error) {
    statusBar.className = 'status-bar offline';
    statusIndicator.className = 'status-indicator offline';
    statusText.textContent = '✗ Server is offline';
  }
}

// Load saved settings
function loadSettings() {
  chrome.storage.sync.get(['method', 'numSentences'], (result) => {
    if (result.method) {
      document.getElementById('methodSelect').value = result.method;
    }
    if (result.numSentences) {
      document.getElementById('sentenceCount').value = result.numSentences;
    }
  });
}

// Save settings when changed
function saveSettings() {
  const method = document.getElementById('methodSelect').value;
  const numSentences = parseInt(document.getElementById('sentenceCount').value);

  chrome.storage.sync.set({
    method: method,
    numSentences: numSentences
  }, () => {
    console.log('Settings saved:', { method, numSentences });
  });
}

// Setup event listeners
function setupEventListeners() {
  // Save settings on change
  document.getElementById('methodSelect').addEventListener('change', saveSettings);
  document.getElementById('sentenceCount').addEventListener('change', saveSettings);

  // Help button
  document.getElementById('helpBtn').addEventListener('click', showHelp);
}

// Show help dialog
function showHelp() {
  const helpMessage = `
TEXT SUMMARIZER - SETUP GUIDE

1. START THE SERVER:
   • Open terminal in project folder
   • Activate venv: venv\\Scripts\\activate
   • Run: python api_server.py
   • Keep terminal open

2. USE THE EXTENSION:
   • Select text on any webpage
   • Right-click → "Summarize"
   • View summary instantly!

3. SETTINGS:
   • Choose method (Hybrid recommended)
   • Set number of sentences (1-10)
   • Settings auto-save

Need more help? Check README.md in project folder.
  `;

  alert(helpMessage);
}

console.log('✅ Popup script loaded');
