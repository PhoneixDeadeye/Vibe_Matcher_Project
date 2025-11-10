# Vibe Matcher Project

A semantic search system for fashion products using **Google Gemini embeddings** and **Pinecone vector database**. This project demonstrates how to match user queries (vibes) with product descriptions using normalized embeddings and cosine similarity.

## 🚀 Features

- **Gemini Embeddings**: Uses Google's `text-embedding-004` model (768 dimensions)
- **Vector Normalization**: All embeddings normalized to unit length (L2 norm = 1.0)
- **Pinecone Integration**: Serverless vector database with cosine similarity
- **Semantic Search**: Match abstract queries to specific fashion products
- **Performance Metrics**: Tracks latency and similarity scores
- **Visualization**: Bar charts showing query latency

## 📋 Requirements

- Python 3.11+
- Google Gemini API key
- Pinecone API key

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PhoneixDeadeye/Vibe_Matcher_Project.git
   cd Vibe_Matcher_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

   Or set environment variables:
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your_gemini_api_key_here"
   $env:PINECONE_API_KEY="your_pinecone_api_key_here"
   
   # Linux/Mac
   export GEMINI_API_KEY="your_gemini_api_key_here"
   export PINECONE_API_KEY="your_pinecone_api_key_here"
   ```

## 🎯 Usage

Open and run `vibe_check.ipynb` in Jupyter Notebook or VS Code:

```bash
jupyter notebook vibe_check.ipynb
```

### Notebook Structure

1. **Cell 1**: Install dependencies (Pinecone, Google Generative AI, pandas, matplotlib)
2. **Cell 2**: Setup API keys and initialize clients
3. **Cell 3**: Define product dataset (15 fashion items with descriptions and tags)
4. **Cell 4**: Create embedding function with vector normalization
5. **Cell 5**: Create Pinecone index and upsert normalized embeddings
6. **Cell 6**: Define search function with cosine similarity
7. **Cell 7**: Run tests and evaluate performance

## 🔍 How It Works

### 1. Vector Normalization
All embeddings (both documents and queries) are normalized to unit length:
```python
def normalize_vector(v):
    v_array = np.array(v)
    norm = np.linalg.norm(v_array)
    normalized = v_array / norm
    return normalized.tolist()
```

### 2. Embedding Generation
Uses Gemini's `text-embedding-004` model with optional title parameter:
```python
embedding = get_gemini_embedding(
    text=product_description,
    task_type="RETRIEVAL_DOCUMENT",
    title=product_name
)
```

### 3. Similarity Search
Query Pinecone with normalized query embeddings:
```python
results = find_vibe_matches_pinecone(
    query="energetic urban chic",
    index=index,
    no_match_threshold=0.7
)
```

## 📊 Dataset

The project includes 15 curated fashion products across different styles:
- Boho Maxi Dress
- Streetwear Graphic Hoodie
- Tailored Linen Suit
- Cozy Cable-Knit Sweater
- Performance Active Leggings
- Vintage Trucker Jacket
- Mulberry Silk Pajama Set
- Techwear Cargo Pants
- Minimalist Court Sneaker
- Preppy Argyle Vest
- Gothic Lace Blouse
- Utility Field Jacket
- Sequin Party Dress
- Y2K Velour Tracksuit
- Coastal-Chic Cardigan

## 🎨 Example Queries

```python
# Abstract style queries
"energetic urban chic"           # → Streetwear Graphic Hoodie
"something cozy for a cold night" # → Cozy Cable-Knit Sweater
"what to wear to a summer wedding" # → Tailored Linen Suit

# Specific product queries (higher scores)
"hoodie for streetwear"           # → Score: 0.72+
"cozy winter sweater"             # → Score: 0.72+
"linen suit for summer wedding"   # → Score: 0.77+
```

## 📈 Performance

- **Average Latency**: ~900-1200ms per query
- **Index Size**: 15 vectors (768 dimensions each)
- **Similarity Threshold**: 0.7 (configurable)
- **Success Rate**: Varies based on query specificity

### Query Tips
- **Specific queries** ("hoodie for streetwear") → Higher scores (0.70-0.80)
- **Abstract queries** ("energetic urban chic") → Lower scores (0.50-0.65)
- Include product types and style keywords for best results

## 🛠️ Configuration

### Pinecone Index Settings
- **Name**: `vibe-matcher-gemini-cosine`
- **Dimension**: 768 (native Gemini dimension)
- **Metric**: Cosine similarity
- **Cloud**: AWS (us-east-1)
- **Type**: Serverless

### Embedding Model
- **Model**: `models/text-embedding-004`
- **Provider**: Google Gemini
- **Output**: 768-dimensional vectors (native, not reduced)

## 🔐 Security

- ⚠️ **Never commit `.env` files** - Already in `.gitignore`
- Store API keys securely in environment variables
- Use GitHub secrets for CI/CD pipelines

## 📝 Notes

- **Normalization**: All vectors are normalized before upserting to Pinecone and before querying
- **Cosine Similarity**: With normalized vectors, cosine similarity = dot product
- **Threshold Tuning**: Adjust `no_match_threshold` based on your use case (0.6-0.8 typical)

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add more fashion products to the dataset
- Experiment with different embedding models
- Improve query preprocessing
- Add more evaluation metrics

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

- [Google Gemini API](https://ai.google.dev/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Repository](https://github.com/PhoneixDeadeye/Vibe_Matcher_Project)

---

**Built with ❤️ using Google Gemini and Pinecone**
