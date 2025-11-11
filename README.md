# Vibe Matcher Project

A semantic search system for fashion products using **Google Gemini embeddings** and **Pinecone vector database** with gRPC. This project demonstrates how to match user queries (vibes) with product descriptions using normalized embeddings and cosine similarity.

## 🚀 Features

- **Gemini Embeddings**: Uses Google's `text-embedding-004` model (768 dimensions)
- **Vector Normalization**: All embeddings normalized to unit length (L2 norm = 1.0)
- **Pinecone gRPC Integration**: High-performance serverless vector database with cosine similarity
- **Semantic Search**: Match abstract queries to specific fashion products
- **Performance Metrics**: Tracks latency, similarity scores, and match quality
- **Rich Visualization**: Dual plots showing match scores and search speed with quality thresholds

## 📋 Requirements

- Python 3.8+
- Google Gemini API key
- Pinecone API key
- Jupyter Notebook or VS Code with Python extension

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
   
   The `requirements.txt` includes:
   - `pinecone[grpc]>=7.3.0` - Pinecone vector database with gRPC support
   - `google-generativeai` - Google Gemini API
   - `python-dotenv` - Environment variable management
   - `pandas` - Data manipulation
   - `numpy` - Numerical operations
   - `matplotlib` - Visualization
   - `tabulate>=0.9.0` - Markdown table formatting

3. **Set up API keys**
   
   Create a `.env` file in the project root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

   The notebook will automatically load these from the `.env` file.

## 🎯 Usage

Open and run `vibe_check.ipynb` in Jupyter Notebook or VS Code:

```bash
jupyter notebook vibe_check.ipynb
# or open in VS Code
```

**Important**: Run cells sequentially from top to bottom.

### Notebook Structure

The notebook contains 6 cells that should be executed in order:

1. **Cell 1: Setup & API Keys**
   - Imports all required libraries
   - Loads API keys from `.env` file using relative path
   - Initializes Gemini and Pinecone gRPC clients
   - Verifies API connectivity

2. **Cell 2: Define Product Data**
   - Creates a DataFrame with 15 fashion products
   - Each product has name, description, and vibe tags
   - Products span diverse styles (boho, streetwear, formal, cozy, etc.)

3. **Cell 3: Create Embedding Functions**
   - `get_gemini_embedding()`: Generates 768-dimensional embeddings
   - `normalize_vector()`: Normalizes vectors to unit length (L2 norm = 1.0)
   - Tests embedding generation and normalization

4. **Cell 4: Create & Populate Pinecone Index**
   - Creates/recreates the `vibe-matcher-gemini-cosine` index
   - Generates embeddings for all products (~9-14 seconds)
   - Uploads normalized vectors to Pinecone
   - Displays progress with checkmarks and timing

5. **Cell 5: Create Search Function**
   - `find_vibe_matches_pinecone()`: Searches for similar products
   - Returns top N matches with similarity scores
   - Handles edge cases (no matches, low scores)

6. **Cell 6: Test & Evaluate**
   - Runs 3 test queries with different vibes
   - Measures search latency and similarity scores
   - Classifies matches as Strong Match (≥0.70), Good Match (≥0.55), or No Match
   - Generates visualizations with match scores and search speed
   - Displays results in markdown table format

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
# Test queries included in the notebook
"energetic urban chic"              # → Streetwear Graphic Hoodie, Techwear Cargo Pants
"something cozy for a cold night"   # → Cozy Cable-Knit Sweater, Mulberry Silk Pajama Set
"a formal outfit for hot weather"   # → Tailored Linen Suit

# You can add your own queries
results = find_vibe_matches_pinecone(
    query="gothic evening wear",
    index=index,
    top_n=3,
    no_match_threshold=0.55
)
```

### Match Quality Thresholds
- 🔥 **Strong Match**: Score ≥ 0.70
- 👍 **Good Match**: Score ≥ 0.55 and < 0.70
- ❌ **No Match**: Score < 0.55

## 📈 Performance

- **Index Creation**: ~5-6 seconds
- **Embedding Generation**: ~9-14 seconds for 15 products (~0.6s per item)
- **Search Latency**: ~1.8-2.2 seconds per query
- **Total Setup Time**: ~14-20 seconds (one-time operation)
- **Vector Dimensions**: 768 (native Gemini embedding size)
- **Match Accuracy**: Dependent on query specificity and product descriptions

### Typical Results
- **Abstract vibes** ("energetic urban chic"): Good Match (0.55-0.70)
- **Specific descriptions** ("cozy for cold night"): Strong Match (0.70+)
- **Formal queries** ("formal hot weather"): Strong Match (0.70+)

### Platform Notes
- ✅ **Windows**: Fully tested with Pinecone gRPC v7.3.0
- ✅ **Linux/Mac**: Compatible (standard installation)
- 💡 **Tip**: Use `pinecone[grpc]` package for best performance

## 🛠️ Configuration

### Pinecone Index Settings
- **Name**: `vibe-matcher-gemini-cosine`
- **Dimension**: 768 (native Gemini embedding dimension)
- **Metric**: Cosine similarity
- **Spec**: Serverless (AWS us-east-1)
- **Connection**: gRPC for high performance

### Embedding Model
- **Model**: `models/text-embedding-004`
- **Provider**: Google Gemini API
- **Task Types**: 
  - `RETRIEVAL_DOCUMENT` for product embeddings (with optional title)
  - `RETRIEVAL_QUERY` for search queries
- **Output**: 768-dimensional vectors (normalized to unit length)

### Search Configuration
- **Default Top Results**: 3 matches
- **Strong Match Threshold**: 0.70
- **Good Match Threshold**: 0.55
- **Adjustable**: Modify thresholds in Cell 6 based on your use case

## 🔐 Security

- ⚠️ **Never commit `.env` files** - Already in `.gitignore`
- Store API keys securely in environment variables
- Use GitHub secrets for CI/CD pipelines

## 📝 Notes

- **Vector Normalization**: All vectors are normalized to unit length (L2 norm = 1.0) before storage and search
- **Cosine Similarity**: With normalized vectors, cosine similarity equals the dot product
- **Threshold Tuning**: Adjust match thresholds (0.55-0.70) based on your domain and requirements
- **gRPC Performance**: Using `pinecone[grpc]` provides faster connections and lower latency than REST API
- **Relative Paths**: The notebook uses relative paths for `.env` file for better portability across systems
- **Windows Compatibility**: Fixed gRPC connection issues specific to Windows environments

## 🐛 Troubleshooting

### Common Issues

1. **"PINECONE_API_KEY not found"**
   - Ensure `.env` file is in the project root directory
   - Check that variable names match exactly: `GEMINI_API_KEY` and `PINECONE_API_KEY`

2. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Ensure you're using `pinecone[grpc]>=7.3.0` (not older `pinecone-client`)

3. **Index creation hangs or times out**
   - This was fixed in v7.3.0 with proper gRPC support
   - If still occurring, check your internet connection and Pinecone service status

4. **Low similarity scores**
   - Try more specific queries with product types and style keywords
   - Consider enriching product descriptions with more detail
   - Adjust the `no_match_threshold` parameter (default: 0.7)

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add more fashion products to the dataset
- Experiment with different embedding models or dimensions
- Improve query preprocessing and enrichment
- Add more evaluation metrics and visualizations
- Optimize performance and latency
- Test on different platforms and environments

## 📄 License

This project is open source and available under the MIT License.

## � Acknowledgments

- **Google Gemini API** for powerful text embeddings
- **Pinecone** for scalable vector search infrastructure
- **Python Data Science Stack** (pandas, numpy, matplotlib)

## �🔗 Links

- [Google Gemini API Documentation](https://ai.google.dev/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Project Repository](https://github.com/PhoneixDeadeye/Vibe_Matcher_Project)
- [Pinecone gRPC Installation Guide](https://docs.pinecone.io/guides/get-started/quickstart)

---

**Built with ❤️ using Google Gemini embeddings and Pinecone vector database**

