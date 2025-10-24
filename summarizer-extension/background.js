// Background service worker for the extension
// Handles context menu creation and message passing

const API_URL = 'http://127.0.0.1:5000';

// Helper: promise wrapper for chrome.storage.sync.get
function getSyncStorage(keys) {
  return new Promise((resolve, reject) => {
    try {
      chrome.storage.sync.get(keys, (items) => {
        if (chrome.runtime && chrome.runtime.lastError) {
          return reject(new Error(chrome.runtime.lastError.message));
        }
        resolve(items);
      });
    } catch (err) {
      reject(err);
    }
  });
}

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
  const settings = await getSyncStorage(['numSentences']);
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
    // Debug: indicate we received the checkServer message
    console.log('🔎 checkServer requested from popup', { sender });

    // Helper: fetch with timeout
    const fetchWithTimeout = (url, timeout = 5000) => {
      return Promise.race([
        fetch(url, { cache: 'no-store' }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), timeout))
      ]);
    };

    // Try primary URL first, then fallback to 127.0.0.1 if hostname fails
    (async () => {
      try {
        console.log('🔗 Pinging', API_URL + '/health');
        const resp = await fetchWithTimeout(`${API_URL}/health`, 5000);
        const data = await resp.json();
        console.log('✅ Health check success', data);
        sendResponse({ status: 'online', data });
      } catch (err1) {
        console.warn('⚠️ Primary health check failed:', err1 && err1.message);
        try {
          console.log('🔁 Trying 127.0.0.1 fallback');
          const resp2 = await fetchWithTimeout(`http://127.0.0.1:5000/health`, 5000);
          const data2 = await resp2.json();
          console.log('✅ Fallback health check success', data2);
          sendResponse({ status: 'online', data: data2 });
        } catch (err2) {
          console.error('❌ All health checks failed', err2 && err2.message);
          sendResponse({ status: 'offline', error: (err2 && err2.message) || (err1 && err1.message) });
        }
      }
    })();

    return true; // Keep channel open for async response
  }

  if (request.action === 'summarize') {
    summarizeText(request.text, request.method, sender.tab.id);
    sendResponse({ status: 'processing' });
    return true;
  }
});

console.log('🚀 Background service worker loaded');
