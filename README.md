# AI Photo Gallery SnapSeek

A powerful terminal-based AI system that finds photos using natural language descriptions. Instead of manually browsing through thousands of photos, simply describe what you're looking for and let AI find it for you!

## ✨ Features

- **Semantic Search**: Find photos using natural language descriptions like "sunset at the beach" or "group of people laughing"
- **Smart AI Vision**: Uses OpenAI's CLIP model to understand both images and text
- **Object Detection**: Automatically detects people, objects, scenes, and concepts in your photos
- **Color Analysis**: Extracts dominant colors from images for enhanced search
- **Intelligent Caching**: Process photos once, search instantly forever
- **Recursive Scanning**: Automatically finds photos in all subdirectories
- **Interactive Mode**: Search multiple queries without restarting
- **Rich Results**: Shows similarity scores, metadata, detected objects, and colors

## Quick Start

### 1. Installation

```bash
# Clone or download the script
wget https://raw.githubusercontent.com/your-repo/ai_gallery.py

# Install dependencies (recommended: use virtual environment)
pip install "numpy<2"
pip install torch torchvision pillow
pip install git+https://github.com/openai/CLIP.git
pip install sentence-transformers
```

**Note**: We recommend NumPy < 2.0 for better compatibility with ML libraries.

### 2. Process Your Photos

```bash
# Scan your photo collection (one-time setup)
python ai_gallery.py /path/to/your/photos --scan
```

### 3. Start Searching

```bash
# Interactive mode (recommended)
python ai_gallery.py /path/to/your/photos --interactive

# Or search directly
python ai_gallery.py /path/to/your/photos --search "cats playing in garden"
```

## 📖 Usage Guide

### Command Line Options

```bash
# Basic usage
python ai_gallery.py <photos_directory> [options]

# Available options:
--scan          # Scan and process photos (first time only)
--rescan        # Force rescan all photos (if you added new ones)
--search "query"  # Search for specific photos
--top N         # Number of results to show (default: 10)
--interactive   # Interactive search mode
```

### Example Commands

```bash
# First time setup
python ai_gallery.py ~/Pictures --scan

# Search for specific photos
python ai_gallery.py ~/Pictures --search "dog running in park"
python ai_gallery.py ~/Pictures --search "birthday party with cake" --top 5

# Interactive search (best experience)
python ai_gallery.py ~/Pictures --interactive

# Added new photos? Rescan to include them
python ai_gallery.py ~/Pictures --rescan
```

### Search Query Examples

The AI understands complex, natural descriptions:

- **People**: "smiling person", "group of friends", "children playing"
- **Animals**: "cat sitting on windowsill", "dog catching frisbee"
- **Scenes**: "sunset over mountains", "city skyline at night", "beach vacation"
- **Objects**: "red car in driveway", "birthday cake with candles"
- **Activities**: "wedding ceremony", "hiking in forest", "cooking in kitchen"
- **Colors/Style**: "person wearing blue shirt", "black and white photo"
- **Locations**: "indoor restaurant", "outdoor concert", "living room"

## 🛠️ Installation Guide

### Option 1: Quick Install
```bash
pip install "numpy<2" torch torchvision pillow
pip install git+https://github.com/openai/CLIP.git
pip install sentence-transformers
```

### Option 2: Clean Environment (Recommended)
```bash
# Create fresh environment
conda create -n ai_gallery python=3.9
conda activate ai_gallery

# Install dependencies in order
pip install numpy==1.24.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow
pip install git+https://github.com/openai/CLIP.git
pip install sentence-transformers
```

### Option 3: With GPU Support
```bash
# For NVIDIA GPU acceleration
pip install "numpy<2"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pillow
pip install git+https://github.com/openai/CLIP.git
pip install sentence-transformers
```

## 📁 Supported File Formats

- **JPEG**: .jpg, .jpeg
- **PNG**: .png
- **BMP**: .bmp
- **GIF**: .gif
- **TIFF**: .tiff
- **WebP**: .webp

## 🔧 How It Works

1. **Image Analysis**: Uses CLIP (Contrastive Language-Image Pre-training) to understand image content
2. **Feature Extraction**: Extracts visual embeddings, detects objects, analyzes colors
3. **Smart Caching**: Saves processed data locally - only processes new/changed photos
4. **Semantic Matching**: Compares your text query with image features using AI similarity
5. **Ranked Results**: Returns most relevant photos with confidence scores

## 📊 Example Output

```
Searching for: 'sunset beach'

Found 8 matching photos:
================================================================================

1. IMG_2041.jpg
   Path: /Users/john/Photos/vacation/IMG_2041.jpg
   Similarity: 0.847
   Resolution: 4032x3024
   Size: 2456 KB
   Detected: sunset, beach, water, outdoor scene, sky
   Colors: #ff6b35, #ffa500, #4682b4

2. beach_sunset.png
   Path: /Users/john/Photos/2023/beach_sunset.png
   Similarity: 0.832
   Resolution: 1920x1080
   Size: 1834 KB
   Detected: sunset, water, outdoor scene, sky
   Colors: #ff4500, #ff8c00, #1e90ff
```

## ⚙️ Advanced Features

### Custom Cache Location
```bash
# The system creates a .ai_gallery_cache folder in the script directory
# This contains:
# - photo_metadata.json (file info, detected objects, colors)
# - embeddings.npy (AI feature vectors for fast search)
```

### Performance Tips

- **GPU Acceleration**: Automatically uses CUDA if available for faster processing
- **Incremental Processing**: Only processes new or modified photos
- **Efficient Storage**: Compressed embeddings and smart metadata caching
- **Memory Management**: Processes photos in batches for large collections

### Directory Structure

The system recursively scans directories:
```
Photos/
├── 2023/
│   ├── vacation/
│   │   ├── beach1.jpg    ✅ Found
│   │   └── sunset.png    ✅ Found
│   └── family/
│       └── reunion.jpg   ✅ Found
├── pets/
│   ├── cat.jpg          ✅ Found
│   └── dog.webp         ✅ Found
└── old_photos/
    └── vintage.tiff     ✅ Found
```

## 🐛 Troubleshooting

### Common Issues

**NumPy Compatibility Error**:
```bash
pip install "numpy<2"
```

**CUDA Out of Memory**:
```bash
# The system automatically falls back to CPU
# Or reduce batch size in the code
```

**No Photos Found**:
- Check if photos are in supported formats
- Verify directory path is correct
- Ensure photos aren't corrupted

**Poor Search Results**:
- Try more descriptive queries
- Use different keywords
- Check if photos were processed correctly during scan

### Performance Issues

**Slow Scanning**:
- Enable GPU acceleration
- Process smaller batches
- Check disk I/O speed

**Large Collections**:
- The system is tested with 10,000+ photos
- Uses efficient caching and indexing
- Memory usage scales linearly

## 🤝 Contributing

Feel free to improve the code:

- Add support for more file formats
- Implement advanced filtering options
- Add face recognition features
- Create GUI interface
- Optimize performance

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) - Amazing vision-language model
- [Sentence Transformers](https://www.sbert.net/) - Text embedding capabilities
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

**Happy Photo Searching! 📸✨**

*Made with ❤️ for photo enthusiasts who have too many pictures and too little time*
