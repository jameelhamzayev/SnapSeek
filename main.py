import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import clip
    import numpy as np
    from sentence_transformers import SentenceTransformer
    print("All required libraries loaded successfully!")
except ImportError as e:
    print(f"Missing required library: {e}")
    print("\nInstall required packages:")
    print("pip install torch torchvision pillow")
    print("pip install git+https://github.com/openai/CLIP.git")
    print("pip install sentence-transformers")
    exit(1)

@dataclass
class PhotoMetadata:
    file_path: str
    file_name: str
    file_size: int
    modified_time: str
    image_hash: str
    clip_embedding: Optional[List[float]] = None
    detected_objects: Optional[List[str]] = None
    color_palette: Optional[List[str]] = None
    resolution: Optional[Tuple[int, int]] = None

class AIPhotoGallery:
    
    def __init__(self, photos_dir: str, cache_dir: str = ".ai_gallery_cache"):
        self.photos_dir = Path(photos_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.metadata_cache = self.cache_dir / "photo_metadata.json"
        self.embeddings_cache = self.cache_dir / "embeddings.npy"
        
        print("Loading AI models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        self.photo_metadata: List[PhotoMetadata] = []
        self.photo_embeddings = None
        
        print("AI models loaded successfully!")
    
    def get_image_hash(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def extract_colors(self, image: Image.Image, num_colors: int = 5) -> List[str]:
        image = image.resize((50, 50))
        image = image.convert('RGB')
        pixels = np.array(image).reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        if len(unique_colors) > num_colors:
            unique_colors = unique_colors[:num_colors]
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in unique_colors]
        return hex_colors
    
    def detect_basic_objects(self, image: Image.Image) -> List[str]:
        object_queries = [
            "a person", "people", "a face", "faces",
            "a car", "cars", "a building", "buildings",
            "a tree", "trees", "flowers", "animals",
            "food", "a beach", "mountains", "sky",
            "indoor scene", "outdoor scene", "night scene",
            "a dog", "a cat", "water", "sunset", "city"
        ]
        
        detected = []
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_tokens = clip.tokenize(object_queries).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            similarities = torch.cosine_similarity(image_features, text_features)
            threshold = 0.25
            for i, similarity in enumerate(similarities):
                if similarity > threshold:
                    detected.append(object_queries[i])
        
        return detected
    
    def process_image(self, image_path: str) -> Optional[PhotoMetadata]:
        try:
            image = Image.open(image_path).convert('RGB')
            file_stat = os.stat(image_path)
            file_hash = self.get_image_hash(image_path)
            metadata = PhotoMetadata(
                file_path=str(image_path),
                file_name=Path(image_path).name,
                file_size=file_stat.st_size,
                modified_time=datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                image_hash=file_hash,
                resolution=(image.width, image.height)
            )
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                metadata.clip_embedding = image_features.cpu().numpy().flatten().tolist()
            metadata.color_palette = self.extract_colors(image)
            metadata.detected_objects = self.detect_basic_objects(image)
            return metadata
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def scan_photos(self, force_rescan: bool = False) -> None:
        print(f"Scanning photos in: {self.photos_dir}")
        existing_metadata = {}
        if self.metadata_cache.exists() and not force_rescan:
            try:
                with open(self.metadata_cache, 'r') as f:
                    cached_data = json.load(f)
                    for item in cached_data:
                        existing_metadata[item['file_path']] = PhotoMetadata(**item)
                print(f"Loaded {len(existing_metadata)} cached entries")
            except Exception as e:
                print(f"Error loading cache: {e}")
                existing_metadata = {}
        image_files = []
        for root, dirs, files in os.walk(self.photos_dir):
            for file in files:
                if Path(file).suffix.lower() in self.supported_formats:
                    image_files.append(os.path.join(root, file))
        print(f"Found {len(image_files)} image files")
        processed_metadata = []
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {Path(image_path).name}")
            if (image_path in existing_metadata and 
                not force_rescan and 
                existing_metadata[image_path].image_hash == self.get_image_hash(image_path)):
                processed_metadata.append(existing_metadata[image_path])
            else:
                metadata = self.process_image(image_path)
                if metadata:
                    processed_metadata.append(metadata)
        self.photo_metadata = processed_metadata
        embeddings = []
        for metadata in self.photo_metadata:
            if metadata.clip_embedding:
                embeddings.append(metadata.clip_embedding)
        if embeddings:
            self.photo_embeddings = np.array(embeddings)
        self.save_cache()
        print(f"Successfully processed {len(self.photo_metadata)} photos")
    
    def save_cache(self) -> None:
        try:
            with open(self.metadata_cache, 'w') as f:
                json.dump([asdict(metadata) for metadata in self.photo_metadata], f, indent=2)
            if self.photo_embeddings is not None:
                np.save(self.embeddings_cache, self.photo_embeddings)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_cache(self) -> bool:
        try:
            if not self.metadata_cache.exists():
                return False
            with open(self.metadata_cache, 'r') as f:
                cached_data = json.load(f)
                self.photo_metadata = [PhotoMetadata(**item) for item in cached_data]
            if self.embeddings_cache.exists():
                self.photo_embeddings = np.load(self.embeddings_cache)
            print(f"Loaded {len(self.photo_metadata)} photos from cache")
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def search_photos(self, query: str, top_k: int = 10) -> List[Tuple[PhotoMetadata, float]]:
        if not self.photo_metadata or self.photo_embeddings is None:
            print("No photos processed. Please run scan first.")
            return []
        print(f"Searching for: '{query}'")
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            query_embedding = text_features.cpu().numpy().flatten()
        similarities = np.dot(self.photo_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append((self.photo_metadata[idx], float(similarities[idx])))
        return results
    
    def print_search_results(self, results: List[Tuple[PhotoMetadata, float]]) -> None:
        if not results:
            print("No matching photos found.")
            return
        print(f"\nFound {len(results)} matching photos:")
        print("=" * 80)
        for i, (metadata, score) in enumerate(results, 1):
            print(f"\n{i}. {metadata.file_name}")
            print(f"   Path: {metadata.file_path}")
            print(f"   Similarity: {score:.3f}")
            print(f"   Resolution: {metadata.resolution[0]}x{metadata.resolution[1]}")
            print(f"   Size: {metadata.file_size // 1024} KB")
            if metadata.detected_objects:
                print(f"   Detected: {', '.join(metadata.detected_objects[:5])}")
            if metadata.color_palette:
                print(f"   Colors: {', '.join(metadata.color_palette[:3])}")

def main():
    parser = argparse.ArgumentParser(description="AI Photo Gallery - Smart Photo Search")
    parser.add_argument("photos_dir", help="Directory containing photos to index")
    parser.add_argument("--scan", action="store_true", help="Scan and process photos")
    parser.add_argument("--rescan", action="store_true", help="Force rescan all photos")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--top", type=int, default=10, help="Number of results to show")
    parser.add_argument("--interactive", action="store_true", help="Interactive search mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.photos_dir):
        print(f"Error: Photos directory '{args.photos_dir}' does not exist.")
        return
    
    gallery = AIPhotoGallery(args.photos_dir)
    
    cache_loaded = gallery.load_cache()
    
    if args.scan or args.rescan or not cache_loaded:
        gallery.scan_photos(force_rescan=args.rescan)
    
    if args.search:
        results = gallery.search_photos(args.search, args.top)
        gallery.print_search_results(results)
    
    elif args.interactive:
        print("\nInteractive Search Mode")
        print("Type your search queries or 'quit' to exit.")
        print("Example queries:")
        print("  - 'person smiling outdoors'")
        print("  - 'sunset beach'")
        print("  - 'cat sitting on chair'")
        print("  - 'group of people at party'")
        
        while True:
            try:
                query = input("\nSearch query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    results = gallery.search_photos(query, args.top)
                    gallery.print_search_results(results)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        print("Use --scan to process photos, --search 'query' to search, or --interactive for interactive mode")
        print("Example: python ai_gallery.py /path/to/photos --interactive")

if __name__ == "__main__":
    main()
