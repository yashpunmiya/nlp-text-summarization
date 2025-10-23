// Background service worker for the extension
// Handles context menu creation and message passing

const API_URL = 'http://localhost:5000';

// Create context menu when extension is installed
chrome.runtime.onInstalled.addListener(() => {
  // Main summarize option (uses default method)
  chrome.contextMenus.create({
    id: 'summarize-hybrid',
    title: '✨ Summarize (Hybrid)',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'separator1',
    type: 'separator',
    contexts: ['selection']
  });

  // Individual methods
  chrome.contextMenus.create({
    id: 'summarize-textrank',
    title: 'Summarize with TextRank',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'summarize-tfidf',
    title: 'Summarize with TF-IDF',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'summarize-lsa',
    title: 'Summarize with LSA',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'summarize-advanced',
    title: 'Summarize with Advanced spaCy',
    contexts: ['selection']
  });

  console.log('✅ Context menus created');
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId.startsWith('summarize-')) {
    const method = info.menuItemId.replace('summarize-', '');
    const selectedText = info.selectionText;

    if (!selectedText || selectedText.trim().length === 0) {
      console.error('No text selected');
      return;
    }

    // Send message to content script to show loading indicator
    chrome.tabs.sendMessage(tab.id, {
      action: 'showLoading',
      text: selectedText
    });

    // Call API to summarize
    summarizeText(selectedText, method, tab.id);
  }
});

// Function to call the API
async function summarizeText(text, method, tabId) {
  try {
    // Get settings (sentence count)
    const settings = await chrome.storage.sync.get(['numSentences']);
    const numSentences = settings.numSentences || 3;

    console.log(`📤 Sending request: method=${method}, sentences=${numSentences}`);

    const response = await fetch(`${API_URL}/api/summarize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        method: method,
        num_sentences: numSentences
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Server error');
    }

    const data = await response.json();
    console.log('✅ Received response:', data);

    // Send result to content script
    chrome.tabs.sendMessage(tabId, {
      action: 'showSummary',
      data: data
    });

  } catch (error) {
    console.error('❌ Error:', error);
    
    let errorMessage = 'Failed to summarize text. ';
    
    if (error.message.includes('Failed to fetch')) {
      errorMessage += 'Make sure the API server is running on http://localhost:5000';
    } else {
      errorMessage += error.message;
    }

    // Send error to content script
    chrome.tabs.sendMessage(tabId, {
      action: 'showError',
      error: errorMessage
    });
  }
}

// Listen for messages from popup or content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkServer') {
    // Check if server is running
    fetch(`${API_URL}/health`)
      .then(response => response.json())
      .then(data => {
        sendResponse({ status: 'online', data: data });
      })
      .catch(error => {
        sendResponse({ status: 'offline', error: error.message });
      });
    return true; // Keep channel open for async response
  }

  if (request.action === 'summarize') {
    summarizeText(request.text, request.method, sender.tab.id);
    sendResponse({ status: 'processing' });
    return true;
  }
});

console.log('🚀 Background service worker loaded');
