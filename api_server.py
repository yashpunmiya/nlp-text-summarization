"""
Flask API Server for Text Summarization Browser Extension
Serves the advanced NLP summarization models via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from advanced_nlp_summarizer import (
    TfidfSummarizer,
    TextRankSummarizer,
    LSASummarizer,
    AdvancedSpacySummarizer,
    HybridSummarizer
)
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for browser extension

# Initialize all summarizers once at startup
print("🚀 Loading NLP models...")
summarizers = {
    'tfidf': TfidfSummarizer(),
    'textrank': TextRankSummarizer(),
    'lsa': LSASummarizer(),
    'advanced': AdvancedSpacySummarizer(),
    'hybrid': HybridSummarizer()
}
print("✅ All models loaded successfully!\n")


@app.route('/health', methods=['GET'])
def health_check():
    """Check if server is running"""
    return jsonify({
        'status': 'running',
        'message': 'Text Summarization API is active',
        'models': list(summarizers.keys())
    })


@app.route('/api/methods', methods=['GET'])
def get_methods():
    """Get available summarization methods"""
    return jsonify({
        'methods': [
            {
                'id': 'hybrid',
                'name': 'Hybrid (Best)',
                'description': 'Combines all methods for optimal results',
                'recommended': True
            },
            {
                'id': 'textrank',
                'name': 'TextRank',
                'description': 'Graph-based PageRank algorithm',
                'recommended': False
            },
            {
                'id': 'tfidf',
                'name': 'TF-IDF',
                'description': 'Term frequency-inverse document frequency',
                'recommended': False
            },
            {
                'id': 'lsa',
                'name': 'LSA',
                'description': 'Latent Semantic Analysis',
                'recommended': False
            },
            {
                'id': 'advanced',
                'name': 'Advanced spaCy',
                'description': 'NER + Dependency parsing',
                'recommended': False
            }
        ]
    })


@app.route('/api/summarize', methods=['POST'])
def summarize():
    """
    Summarize text using specified method
    
    Request body:
    {
        "text": "text to summarize",
        "method": "hybrid",  # optional, defaults to hybrid
        "num_sentences": 3   # optional, defaults to 3
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'message': 'Please provide text in the request body'
            }), 400
        
        text = data.get('text', '').strip()
        method = data.get('method', 'hybrid').lower()
        num_sentences = int(data.get('num_sentences', 3))
        
        # Validate input
        if not text:
            return jsonify({
                'error': 'Empty text',
                'message': 'The provided text is empty'
            }), 400
        
        if len(text.split()) < 10:
            return jsonify({
                'error': 'Text too short',
                'message': 'Please provide at least 10 words to summarize'
            }), 400
        
        if method not in summarizers:
            return jsonify({
                'error': 'Invalid method',
                'message': f'Method must be one of: {", ".join(summarizers.keys())}'
            }), 400
        
        if num_sentences < 1 or num_sentences > 10:
            return jsonify({
                'error': 'Invalid sentence count',
                'message': 'Number of sentences must be between 1 and 10'
            }), 400
        
        # Perform summarization
        start_time = time.time()
        summarizer = summarizers[method]
        summary = summarizer.summarize(text, num_sentences=num_sentences)
        processing_time = time.time() - start_time
        
        # Calculate statistics
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = ((original_words - summary_words) / original_words * 100) if original_words > 0 else 0
        
        # Count sentences
        original_sentences = len([s for s in text.split('.') if s.strip()])
        summary_sentences = len([s for s in summary.split('.') if s.strip()])
        
        # Return results
        return jsonify({
            'success': True,
            'summary': summary,
            'method': method,
            'stats': {
                'original_words': original_words,
                'original_sentences': original_sentences,
                'summary_words': summary_words,
                'summary_sentences': summary_sentences,
                'compression_ratio': round(compression_ratio, 1),
                'processing_time': round(processing_time, 3)
            }
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'error': 'Processing error',
            'message': str(e)
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare_methods():
    """
    Compare all summarization methods
    
    Request body:
    {
        "text": "text to summarize",
        "num_sentences": 3
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided'
            }), 400
        
        text = data.get('text', '').strip()
        num_sentences = int(data.get('num_sentences', 3))
        
        if not text or len(text.split()) < 10:
            return jsonify({
                'error': 'Text too short'
            }), 400
        
        # Run all methods
        results = {}
        for method_name, summarizer in summarizers.items():
            start_time = time.time()
            summary = summarizer.summarize(text, num_sentences=num_sentences)
            processing_time = time.time() - start_time
            
            results[method_name] = {
                'summary': summary,
                'processing_time': round(processing_time, 3),
                'summary_words': len(summary.split())
            }
        
        return jsonify({
            'success': True,
            'results': results,
            'original_words': len(text.split())
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'error': 'Processing error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 80)
    print("📝 TEXT SUMMARIZATION API SERVER")
    print("=" * 80)
    print("\n✅ Server starting on http://localhost:8000")
    print("✅ CORS enabled for browser extension")
    print("\n📌 Available endpoints:")
    print("   GET  /health           - Check server status")
    print("   GET  /api/methods      - List available methods")
    print("   POST /api/summarize    - Summarize text")
    print("   POST /api/compare      - Compare all methods")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    # Run Flask server
    app.run(host='127.0.0.1', port=8000, debug=False)
