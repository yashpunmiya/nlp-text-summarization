# NLP Summarization Methods - Technical Flowcharts

## 1. TF-IDF Summarization
**What it is:** TF-IDF (Term Frequency-Inverse Document Frequency) ranks sentences by their word importance. Words that appear frequently in a sentence but rarely across the document get high scores.

**Logic:** Statistical approach that identifies informative terms by balancing local frequency with global rarity.

**Technical Implementation:**
- **Library:** `sklearn.feature_extraction.text.TfidfVectorizer`
- **Key Methods:** `fit_transform()`, `sum()` (numpy)
- **Parameters:** `ngram_range=(1,2)`, `stop_words='english'`, `max_features=None`
- **Output:** Sparse TF-IDF matrix, sentence scores via row sums

```
┌─────────────────┐
│   INPUT TEXT    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SPLIT INTO      │
│ SENTENCES       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ TF-IDF VECTOR   │
│ FIT_TRANSFORM   │
│ (ngram_range=   │
│  (1,2), stop_   │
│  words='english')│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ CALCULATE       │
│ SENTENCE SCORES │
│ sum(TF-IDF      │
│ weights per     │
│ sentence)       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SELECT TOP N    │
│ SENTENCES       │
│ (by TF-IDF sum) │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ REORDER BY      │
│ ORIGINAL        │
│ POSITION        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ JOIN INTO       │
│ SUMMARY TEXT    │
└─────────────────┘
```

---

## 2. TextRank Summarization
**What it is:** TextRank adapts Google's PageRank algorithm for text. Sentences vote for similar sentences, creating a graph where important sentences get higher scores through network connectivity.

**Logic:** Graph-based ranking where sentence similarity creates edges, and iterative PageRank propagation identifies globally important sentences.

**Technical Implementation:**
- **Libraries:** `networkx` (graph operations), `numpy` (matrix operations), `spacy` (sentence splitting)
- **Key Methods:** `nx.from_numpy_array()`, `nx.pagerank()`, `set()` (for Jaccard), `STOP_WORDS`
- **Parameters:** `alpha=0.85` (damping factor), similarity threshold implicit
- **Data Structures:** N×N similarity matrix, NetworkX graph object

```
┌─────────────────┐
│   INPUT TEXT    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SPLIT INTO      │
│ SENTENCES       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ BUILD SIMILARITY│
│ MATRIX          │
│ (N×N array)     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ CALCULATE PAIR  │
│ SIMILARITIES    │
│ Jaccard:        │
│ |A∩B| / |A∪B|   │
│ (remove stop    │
│  words)         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ CREATE GRAPH    │
│ NetworkX from   │
│ similarity      │
│ matrix          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ APPLY PAGERANK  │
│ nx.pagerank()   │
│ alpha=0.85      │
│ (damping factor)│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SELECT TOP N    │
│ SENTENCES       │
│ (by PageRank    │
│ score)          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ REORDER BY      │
│ ORIGINAL        │
│ POSITION        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ JOIN INTO       │
│ SUMMARY TEXT    │
└─────────────────┘
```

---

## 3. LSA Summarization
**What it is:** LSA (Latent Semantic Analysis) uses linear algebra to discover hidden topics in text. Sentences are projected into a lower-dimensional topic space and scored by their topic strength.

**Logic:** SVD decomposition reveals latent topics, and sentences are scored by their representation strength in the reduced topic space.

**Technical Implementation:**
- **Libraries:** `sklearn.feature_extraction.text.CountVectorizer`, `sklearn.decomposition.TruncatedSVD`, `numpy.linalg`
- **Key Methods:** `CountVectorizer.fit_transform()`, `TruncatedSVD.fit_transform()`, `numpy.linalg.norm()`
- **Parameters:** `n_components=3` (topics), `random_state=42`, `max_features=1000`
- **Math:** X = U × Σ × V^T (SVD), topic_space = X × V_k, score = ||topic_vector||₂

```
┌─────────────────┐
│   INPUT TEXT    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SPLIT INTO      │
│ SENTENCES       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ CREATE TERM-    │
│ DOCUMENT MATRIX │
│ CountVectorizer │
│ (stop_words=    │
│ 'english')      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ APPLY TRUNCATED │
│ SVD              │
│ n_components=3   │
│ (latent topics)  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ PROJECT TO      │
│ TOPIC SPACE     │
│ svd.transform() │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ CALCULATE L2    │
│ NORM SCORES     │
│ ||vector||₂     │
│ (topic strength)│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SELECT TOP N    │
│ SENTENCES       │
│ (by L2 norm)    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ REORDER BY      │
│ ORIGINAL        │
│ POSITION        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ JOIN INTO       │
│ SUMMARY TEXT    │
└─────────────────┘
```

---

## 4. Advanced spaCy Summarization
**What it is:** Advanced spaCy leverages linguistic analysis including named entities, dependency parsing, POS tagging, and sentence structure to score sentences by their informational content.

**Logic:** Multi-feature scoring combines NER entities, noun phrases, grammatical structure, and positional importance for comprehensive sentence evaluation.

**Technical Implementation:**
- **Library:** `spacy` (en_core_web_sm model)
- **Key Methods:** `spacy.load()`, `.ents` (NER), `.noun_chunks`, `.pos_` (POS tags), `.dep_` (dependencies)
- **Features Used:** Named entities (×0.3), noun chunks (×0.2), POS tags (×0.1), root verbs (+0.2), length bonus, position bonus
- **Scoring Formula:** entities×0.3 + chunks×0.2 + pos×0.1 + length_bonus + root_score + position_bonus

```
┌─────────────────┐
│   INPUT TEXT    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ LOAD SPACY      │
│ MODEL           │
│ en_core_web_sm  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ PROCESS TEXT    │
│ doc = nlp(text) │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ FOR EACH        │
│ SENTENCE:       │
│ • NER entities  │
│ • Noun chunks   │
│ • POS tags      │
│ • Dependencies  │
│ • Length bonus  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ CALCULATE       │
│ SCORES:         │
│ entities × 0.3  │
│ + chunks × 0.2  │
│ + pos × 0.1     │
│ + length bonus  │
│ + root verbs    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ ADD POSITION    │
│ BONUS:          │
│ First: +0.3     │
│ Last: +0.2      │
│ Early: +0.15    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SELECT TOP N    │
│ SENTENCES       │
│ (by total score)│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ REORDER BY      │
│ ORIGINAL        │
│ POSITION        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ JOIN INTO       │
│ SUMMARY TEXT    │
└─────────────────┘
```

---

## Method Comparison Summary

| Method | Approach | Key Algorithm | Libraries/Methods | Best For |
|--------|----------|---------------|-------------------|----------|
| TF-IDF | Statistical | Vector weighting | `sklearn.TfidfVectorizer`, `numpy.sum` | Keyword-rich content |
| TextRank | Graph-based | PageRank | `networkx.pagerank`, `numpy.zeros` | Connected narratives |
| LSA | Linear Algebra | SVD decomposition | `sklearn.TruncatedSVD`, `numpy.linalg.norm` | Multi-topic documents |
| Advanced spaCy | Linguistic | Multi-feature scoring | `spacy` (NER, POS, deps) | Entity-heavy content |
| Hybrid | Ensemble | Weighted combination | All above + normalization | General purpose |

Each method extracts sentences differently, and the hybrid combines their strengths for optimal results.</content>
<parameter name="filePath">all_methods.md