"""
Advanced NLP Text Summarization - Pure NLP Techniques
This module implements state-of-the-art extractive summarization using:
1. TF-IDF with sentence ranking
2. TextRank algorithm (PageRank for text)
3. LSA (Latent Semantic Analysis)
4. Advanced spaCy with dependency parsing
5. Hybrid approach combining multiple methods
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import networkx as nx
from typing import List, Tuple
import re


class TfidfSummarizer:
    """TF-IDF based extractive summarization - more sophisticated than simple frequency"""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(
            max_features=None,
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
            stop_words='english'
        )
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Summarize using TF-IDF scores
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            Summarized text
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores (sum of TF-IDF values)
            sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            
            # Get top sentences while preserving order
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices_sorted = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices_sorted]
            return " ".join(summary_sentences)
        except:
            return sentences[0] if sentences else ""
    
    def get_sentence_scores(self, text: str) -> pd.DataFrame:
        """Get detailed TF-IDF scores for each sentence"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            
            return pd.DataFrame({
                'Sentence': sentences,
                'TF-IDF Score': sentence_scores
            }).sort_values('TF-IDF Score', ascending=False)
        except:
            return pd.DataFrame(columns=['Sentence', 'TF-IDF Score'])


class TextRankSummarizer:
    """
    TextRank algorithm - Uses PageRank algorithm on sentence similarity graph
    This is one of the most effective extractive summarization techniques
    """
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    
    def _preprocess_sentence(self, sentence: str) -> str:
        """Clean and preprocess sentence"""
        # Remove extra whitespace
        sentence = ' '.join(sentence.split())
        return sentence.lower()
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences using word overlap"""
        words1 = set(self._preprocess_sentence(sent1).split())
        words2 = set(self._preprocess_sentence(sent2).split())
        
        # Remove stopwords
        stopwords = set(STOP_WORDS)
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def summarize(self, text: str, num_sentences: int = 3, damping: float = 0.85) -> str:
        """
        Summarize using TextRank algorithm
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            damping: PageRank damping factor (0.85 is standard)
            
        Returns:
            Summarized text
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Build similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, alpha=damping)
        
        # Rank sentences
        ranked_sentences = sorted(
            ((scores[i], i, s) for i, s in enumerate(sentences)),
            reverse=True
        )
        
        # Get top sentences and sort by original position
        top_sentences = sorted(
            ranked_sentences[:num_sentences],
            key=lambda x: x[1]
        )
        
        return " ".join([s[2] for s in top_sentences])
    
    def get_sentence_scores(self, text: str) -> pd.DataFrame:
        """Get TextRank scores for all sentences"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if len(sentences) == 0:
            return pd.DataFrame(columns=['Sentence', 'TextRank Score'])
        
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        return pd.DataFrame({
            'Sentence': sentences,
            'TextRank Score': [scores[i] for i in range(len(sentences))]
        }).sort_values('TextRank Score', ascending=False)


class LSASummarizer:
    """
    LSA (Latent Semantic Analysis) based summarization
    Uses SVD to identify key topics and select representative sentences
    """
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    
    def summarize(self, text: str, num_sentences: int = 3, n_topics: int = 3) -> str:
        """
        Summarize using LSA
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            n_topics: Number of latent topics to extract
            
        Returns:
            Summarized text
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create term-document matrix
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        try:
            X = vectorizer.fit_transform(sentences)
            
            # Apply SVD
            n_components = min(n_topics, X.shape[0], X.shape[1])
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd_matrix = svd.fit_transform(X)
            
            # Score sentences based on their representation in topic space
            sentence_scores = np.linalg.norm(svd_matrix, axis=1)
            
            # Select top sentences
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices_sorted = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices_sorted]
            return " ".join(summary_sentences)
        except:
            return sentences[0] if sentences else ""
    
    def get_sentence_scores(self, text: str) -> pd.DataFrame:
        """Get LSA scores for sentences"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        try:
            X = vectorizer.fit_transform(sentences)
            n_components = min(3, X.shape[0], X.shape[1])
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd_matrix = svd.fit_transform(X)
            sentence_scores = np.linalg.norm(svd_matrix, axis=1)
            
            return pd.DataFrame({
                'Sentence': sentences,
                'LSA Score': sentence_scores
            }).sort_values('LSA Score', ascending=False)
        except:
            return pd.DataFrame(columns=['Sentence', 'LSA Score'])


class AdvancedSpacySummarizer:
    """
    Advanced spaCy-based summarization using:
    - Named Entity Recognition (NER)
    - Dependency parsing
    - POS tagging
    - Semantic similarity using word vectors
    """
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    
    def _calculate_advanced_score(self, sentence) -> float:
        """Calculate advanced score using multiple NLP features"""
        score = 0.0
        
        # 1. Named entities (sentences with entities are more important)
        entities_score = len(sentence.ents) * 0.3
        
        # 2. Noun phrases (more noun phrases = more informative)
        noun_chunks_score = len(list(sentence.noun_chunks)) * 0.2
        
        # 3. Proper nouns and important POS tags
        important_pos = ['PROPN', 'NOUN', 'VERB']
        pos_score = sum(1 for token in sentence if token.pos_ in important_pos) * 0.1
        
        # 4. Sentence position (first sentences often important)
        # This will be added separately
        
        # 5. Sentence length (not too short, not too long)
        length = len(sentence)
        if 10 < length < 30:
            length_score = 0.3
        elif 5 < length <= 40:
            length_score = 0.15
        else:
            length_score = 0.0
        
        # 6. Keywords from dependency parsing (root verbs are important)
        root_score = 0.2 if any(token.dep_ == 'ROOT' for token in sentence) else 0
        
        score = entities_score + noun_chunks_score + pos_score + length_score + root_score
        
        return score
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Summarize using advanced spaCy features
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            
        Returns:
            Summarized text
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate scores for each sentence
        sentence_scores = {}
        for i, sent in enumerate(sentences):
            base_score = self._calculate_advanced_score(sent)
            
            # Position bonus (first and last sentences often important)
            position_bonus = 0
            if i == 0:
                position_bonus = 0.3
            elif i == len(sentences) - 1:
                position_bonus = 0.2
            elif i < len(sentences) * 0.2:  # First 20%
                position_bonus = 0.15
            
            sentence_scores[sent.text.strip()] = base_score + position_bonus
        
        # Select top sentences
        top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        
        # Reorder by original position
        ordered_summary = []
        for sent in sentences:
            if sent.text.strip() in top_sentences:
                ordered_summary.append(sent.text.strip())
        
        return " ".join(ordered_summary)
    
    def get_sentence_scores(self, text: str) -> pd.DataFrame:
        """Get detailed scores for each sentence"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        scores_data = []
        for i, sent in enumerate(sentences):
            base_score = self._calculate_advanced_score(sent)
            position_bonus = 0.3 if i == 0 else (0.2 if i == len(sentences)-1 else 0)
            
            scores_data.append({
                'Sentence': sent.text.strip(),
                'Score': base_score + position_bonus,
                'Entities': len(sent.ents),
                'Noun Chunks': len(list(sent.noun_chunks)),
                'Length': len(sent)
            })
        
        return pd.DataFrame(scores_data).sort_values('Score', ascending=False)


class HybridSummarizer:
    """
    Hybrid approach combining multiple summarization techniques
    This often produces the best results by leveraging strengths of different methods
    """
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfSummarizer()
        self.textrank = TextRankSummarizer()
        self.lsa = LSASummarizer()
        self.advanced = AdvancedSpacySummarizer()
    
    def summarize(self, text: str, num_sentences: int = 3, 
                  weights: dict = None) -> str:
        """
        Combine multiple methods with weighted voting
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            weights: Dictionary of weights for each method
                    {'tfidf': 0.25, 'textrank': 0.35, 'lsa': 0.15, 'advanced': 0.25}
        
        Returns:
            Summarized text
        """
        if weights is None:
            weights = {
                'tfidf': 0.25,
                'textrank': 0.35,  # TextRank often performs best
                'lsa': 0.15,
                'advanced': 0.25
            }
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Get scores from each method
        combined_scores = {sent: 0.0 for sent in sentences}
        
        # TF-IDF scores
        if weights.get('tfidf', 0) > 0:
            df_tfidf = self.tfidf.get_sentence_scores(text)
            max_score = df_tfidf['TF-IDF Score'].max() if len(df_tfidf) > 0 else 1
            for _, row in df_tfidf.iterrows():
                if max_score > 0:
                    combined_scores[row['Sentence']] += (
                        row['TF-IDF Score'] / max_score * weights['tfidf']
                    )
        
        # TextRank scores
        if weights.get('textrank', 0) > 0:
            df_textrank = self.textrank.get_sentence_scores(text)
            max_score = df_textrank['TextRank Score'].max() if len(df_textrank) > 0 else 1
            for _, row in df_textrank.iterrows():
                if max_score > 0:
                    combined_scores[row['Sentence']] += (
                        row['TextRank Score'] / max_score * weights['textrank']
                    )
        
        # LSA scores
        if weights.get('lsa', 0) > 0:
            df_lsa = self.lsa.get_sentence_scores(text)
            max_score = df_lsa['LSA Score'].max() if len(df_lsa) > 0 else 1
            for _, row in df_lsa.iterrows():
                if max_score > 0:
                    combined_scores[row['Sentence']] += (
                        row['LSA Score'] / max_score * weights['lsa']
                    )
        
        # Advanced spaCy scores
        if weights.get('advanced', 0) > 0:
            df_advanced = self.advanced.get_sentence_scores(text)
            max_score = df_advanced['Score'].max() if len(df_advanced) > 0 else 1
            for _, row in df_advanced.iterrows():
                if max_score > 0:
                    combined_scores[row['Sentence']] += (
                        row['Score'] / max_score * weights['advanced']
                    )
        
        # Select top sentences and maintain order
        top_sentences = nlargest(num_sentences, combined_scores, key=combined_scores.get)
        
        ordered_summary = []
        for sent in sentences:
            if sent in top_sentences:
                ordered_summary.append(sent)
        
        return " ".join(ordered_summary)
    
    def get_all_scores(self, text: str) -> pd.DataFrame:
        """Get scores from all methods for comparison"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Get individual scores
        df_tfidf = self.tfidf.get_sentence_scores(text)
        df_textrank = self.textrank.get_sentence_scores(text)
        df_lsa = self.lsa.get_sentence_scores(text)
        df_advanced = self.advanced.get_sentence_scores(text)
        
        # Combine into single dataframe
        result = pd.DataFrame({'Sentence': sentences})
        
        # Merge scores
        result = result.merge(
            df_tfidf[['Sentence', 'TF-IDF Score']], 
            on='Sentence', how='left'
        )
        result = result.merge(
            df_textrank[['Sentence', 'TextRank Score']], 
            on='Sentence', how='left'
        )
        result = result.merge(
            df_lsa[['Sentence', 'LSA Score']], 
            on='Sentence', how='left'
        )
        result = result.merge(
            df_advanced[['Sentence', 'Score']], 
            on='Sentence', how='left'
        )
        result.rename(columns={'Score': 'Advanced Score'}, inplace=True)
        
        # Fill NaN values
        result.fillna(0, inplace=True)
        
        # Calculate combined score
        result['Combined Score'] = (
            result['TF-IDF Score'] * 0.25 +
            result['TextRank Score'] * 0.35 +
            result['LSA Score'] * 0.15 +
            result['Advanced Score'] * 0.25
        )
        
        return result.sort_values('Combined Score', ascending=False)


# Sample text for testing
SAMPLE_TEXT = """In a world often dominated by negativity, it's important to remember the power of kindness and compassion. Small acts of kindness have the ability to brighten someone's day, uplift spirits, and create a ripple effect of positivity that can spread far and wide. Whether it's a smile to a stranger, a helping hand to a friend in need, or a thoughtful gesture to a colleague, every act of kindness has the potential to make a difference in someone's life.Beyond individual actions, there is also immense power in collective efforts to create positive change. When communities come together to support one another, incredible things can happen. From grassroots initiatives to global movements, people are uniting to tackle pressing social and environmental issues, driving meaningful progress and inspiring hope for a better future.It's also important to recognize the strength that lies within each and every one of us. We all have the ability to make a positive impact, no matter how small our actions may seem. By tapping into our innate compassion and empathy, we can cultivate a culture of kindness and empathy that enriches our lives and those around us.So let's embrace the power of kindness, and strive to make the world a better place one small act at a time. Together, we can create a brighter, more compassionate future for all."""


def demo_all_methods():
    """Demonstrate all summarization methods"""
    print("=" * 100)
    print("ADVANCED NLP SUMMARIZATION - COMPARISON OF ALL METHODS")
    print("=" * 100)
    
    print(f"\nOriginal text ({len(SAMPLE_TEXT)} characters, {len(SAMPLE_TEXT.split())} words)")
    print("-" * 100)
    
    # 1. TF-IDF
    print("\n1. TF-IDF SUMMARIZATION")
    print("-" * 100)
    tfidf = TfidfSummarizer()
    summary = tfidf.summarize(SAMPLE_TEXT, num_sentences=2)
    print(f"Summary: {summary}\n")
    
    # 2. TextRank
    print("\n2. TEXTRANK SUMMARIZATION (PageRank for Text)")
    print("-" * 100)
    textrank = TextRankSummarizer()
    summary = textrank.summarize(SAMPLE_TEXT, num_sentences=2)
    print(f"Summary: {summary}\n")
    
    # 3. LSA
    print("\n3. LSA SUMMARIZATION (Latent Semantic Analysis)")
    print("-" * 100)
    lsa = LSASummarizer()
    summary = lsa.summarize(SAMPLE_TEXT, num_sentences=2)
    print(f"Summary: {summary}\n")
    
    # 4. Advanced spaCy
    print("\n4. ADVANCED SPACY SUMMARIZATION (NER + Dependency Parsing)")
    print("-" * 100)
    advanced = AdvancedSpacySummarizer()
    summary = advanced.summarize(SAMPLE_TEXT, num_sentences=2)
    print(f"Summary: {summary}\n")
    
    # 5. Hybrid (Best)
    print("\n5. HYBRID SUMMARIZATION (Combined Approach) ⭐ RECOMMENDED")
    print("-" * 100)
    hybrid = HybridSummarizer()
    summary = hybrid.summarize(SAMPLE_TEXT, num_sentences=2)
    print(f"Summary: {summary}\n")
    
    # Show detailed scores
    print("\n" + "=" * 100)
    print("DETAILED SCORE COMPARISON")
    print("=" * 100)
    df = hybrid.get_all_scores(SAMPLE_TEXT)
    print(df.to_string(index=False))
    print("\n")


if __name__ == "__main__":
    demo_all_methods()
