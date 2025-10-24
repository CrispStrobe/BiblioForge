# extractors/mlx_vlm_extractor.py
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import json
import re

class MLXVLMExtractor:
    """
    Vision Language Model extractor using mlx-vlm (optimized for Apple Silicon).
    Much faster than llama-mtmd-cli on Mac hardware.
    """
    
    def __init__(self, import_cache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._debug = debug
        self._import_cache = import_cache
        self._binary_paths = binary_paths or {}
        
        # MLX-VLM components
        self._mlx_vlm = None
        self._model = None
        self._processor = None
        self._config = None
        
        # Configuration
        self.model_id = None
        self.max_tokens = 512
        self.temperature = 0.0  # 0.0 for deterministic OCR
        
        # Check for mlx-vlm availability
        self._check_mlx_vlm()
    
    def _check_mlx_vlm(self):
        """Check if mlx-vlm is available."""
        try:
            import mlx.core as mx
            self._mlx_available = True
            if self._debug:
                logging.debug("MLX is available")
        except ImportError:
            self._mlx_available = False
            logging.warning("mlx-vlm not available. Install with: pip install mlx-vlm")
    
    def set_model(self, model_id: str):
        """
        Set the HuggingFace model ID to use.
        For MLX models: 'mlx-community/LFM2-VL-3B-8bit'
        """
        self.model_id = model_id
        if self._debug:
            logging.debug(f"MLXVLMExtractor: Set model to {model_id}")
    
    def _load_model(self):
        """Lazy load the MLX-VLM model."""
        if self._model is not None:
            return True
        
        if not self._mlx_available:
            logging.error("MLX not available")
            return False
        
        if not self.model_id:
            logging.error("No model specified for MLXVLMExtractor")
            return False
        
        try:
            if self._debug:
                logging.info(f"Loading MLX-VLM model: {self.model_id}")
            
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            
            # Load model and processor
            self._model, self._processor = load(self.model_id)
            self._config = load_config(self.model_id)
            
            if self._debug:
                logging.info(f"MLX-VLM model loaded successfully")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load MLX-VLM model: {e}", exc_info=self._debug)
            return False
    
    def extract_text(self, input_path: str, 
                    progress_callback: Optional[Callable] = None,
                    **kwargs) -> str:
        """
        Extract text from document using MLX-VLM model.
        """
        if not self._load_model():
            return ""
        
        try:
            file_ext = os.path.splitext(input_path)[1].lower()
            
            if file_ext == '.pdf':
                return self._extract_from_pdf(input_path, progress_callback)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']:
                return self._extract_from_image(input_path)
            else:
                logging.warning(f"Unsupported file type for MLX-VLM extraction: {file_ext}")
                return ""
                
        except Exception as e:
            logging.error(f"MLX-VLM extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def _extract_from_pdf(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Extract text from PDF by converting to images and processing each page."""
        pdf2image = self._import_cache.import_module('pdf2image')
        if not pdf2image:
            logging.error("pdf2image required for PDF processing with VL models")
            return ""
        
        text_parts = []
        poppler_path = self._binary_paths.get('pdftoppm')
        
        try:
            # Convert PDF to images
            conversion_kwargs = {
                'dpi': 200,
                'fmt': 'jpeg',
                'thread_count': 1
            }
            
            if poppler_path:
                poppler_dir = os.path.dirname(poppler_path)
                conversion_kwargs['poppler_path'] = poppler_dir
            
            images = pdf2image.convert_from_path(pdf_path, **conversion_kwargs)
            
            if self._debug:
                logging.debug(f"Converted PDF to {len(images)} images")
            
            # Process each page
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, image in enumerate(images):
                    temp_image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                    image.save(temp_image_path, 'JPEG', quality=90)
                    
                    page_text = self._extract_from_image(temp_image_path)
                    if page_text.strip():
                        text_parts.append(page_text.strip())
                    
                    if progress_callback:
                        progress_callback(1)
                    
                    # Clean up
                    image.close()
            
            return "\n\n---PAGE BREAK---\n\n".join(text_parts)
            
        except Exception as e:
            logging.error(f"PDF to MLX-VLM extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def _extract_from_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Extract text from a single image using MLX-VLM.
        """
        if not prompt:
            prompt = (
                "Extract all text from this image. "
                "Transcribe everything you see, maintaining the original layout and structure. "
                "Include headings, paragraphs, lists, and any other text elements."
            )
        
        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
            
            # Prepare image
            image = [image_path]
            
            # Apply chat template
            formatted_prompt = apply_chat_template(
                self._processor, 
                self._config, 
                prompt, 
                num_images=len(image)
            )
            
            if self._debug:
                logging.debug(f"Processing image: {image_path}")
            
            # Generate output
            output = generate(
                self._model, 
                self._processor, 
                formatted_prompt, 
                image, 
                verbose=False,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if self._debug:
                logging.debug(f"MLX-VLM output length: {len(output)} chars")
            
            return output.strip()
            
        except Exception as e:
            logging.error(f"MLX-VLM image extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def extract_metadata_direct(self, input_path: str) -> Dict[str, Any]:
        """
        Extract metadata directly using MLX-VLM with targeted prompting.
        Returns dict with title, author, year, etc.
        """
        if not self._load_model():
            return {}
        
        try:
            file_ext = os.path.splitext(input_path)[1].lower()
            
            if file_ext == '.pdf':
                images = self._get_pdf_sample_images(input_path, max_pages=3)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']:
                images = [input_path]
            else:
                return {}
            
            if not images:
                return {}
            
            # Metadata extraction prompt
            metadata_prompt = """Analyze this document image and extract bibliographic metadata.

Return ONLY a JSON object with these fields:
{
  "title": "the document title",
  "author": "author name(s)",
  "year": "publication year (YYYY format)",
  "publisher": "publisher name",
  "isbn": "ISBN if visible",
  "doi": "DOI if visible",
  "document_type": "book|article|paper|report|other"
}

If a field is not visible or unclear, use null. Do not add explanatory text, only return the JSON."""
            
            # Try first image
            metadata_text = self._extract_from_image(images[0], prompt=metadata_prompt)
            
            # Parse JSON response
            metadata = self._parse_metadata_json(metadata_text)
            
            # Clean up temp images
            for img in images:
                if img != input_path and os.path.exists(img):
                    try:
                        os.unlink(img)
                    except:
                        pass
            
            return metadata
            
        except Exception as e:
            logging.error(f"MLX-VLM metadata extraction failed: {e}", exc_info=self._debug)
            return {}
    
    def _get_pdf_sample_images(self, pdf_path: str, max_pages: int = 3) -> List[str]:
        """Extract first few pages of PDF as temp image files."""
        pdf2image = self._import_cache.import_module('pdf2image')
        if not pdf2image:
            return []
        
        temp_images = []
        temp_dir = tempfile.mkdtemp(prefix='mlx_vlm_')
        
        try:
            poppler_path = self._binary_paths.get('pdftoppm')
            conversion_kwargs = {
                'dpi': 200,
                'fmt': 'jpeg',
                'first_page': 1,
                'last_page': max_pages
            }
            
            if poppler_path:
                conversion_kwargs['poppler_path'] = os.path.dirname(poppler_path)
            
            images = pdf2image.convert_from_path(pdf_path, **conversion_kwargs)
            
            for i, image in enumerate(images):
                temp_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                image.save(temp_path, 'JPEG', quality=90)
                temp_images.append(temp_path)
                image.close()
            
            return temp_images
            
        except Exception as e:
            logging.error(f"Failed to extract PDF sample images: {e}")
            for img in temp_images:
                try:
                    os.unlink(img)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
            return []
    
    def _parse_metadata_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON metadata from VL model output."""
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                metadata = json.loads(json_str)
                
                # Validate and clean
                cleaned = {}
                if 'title' in metadata and metadata['title']:
                    cleaned['title'] = str(metadata['title']).strip()
                if 'author' in metadata and metadata['author']:
                    cleaned['author'] = str(metadata['author']).strip()
                if 'year' in metadata and metadata['year']:
                    year_str = str(metadata['year']).strip()
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
                    if year_match:
                        cleaned['year'] = year_match.group(0)
                if 'publisher' in metadata and metadata['publisher']:
                    cleaned['publisher'] = str(metadata['publisher']).strip()
                if 'isbn' in metadata and metadata['isbn']:
                    cleaned['isbn'] = str(metadata['isbn']).strip()
                if 'doi' in metadata and metadata['doi']:
                    cleaned['doi'] = str(metadata['doi']).strip()
                if 'document_type' in metadata and metadata['document_type']:
                    cleaned['document_type'] = str(metadata['document_type']).strip().lower()
                
                return cleaned
            
            return {}
            
        except json.JSONDecodeError as e:
            if self._debug:
                logging.debug(f"Failed to parse JSON metadata: {e}")
            return {}
        except Exception as e:
            logging.error(f"Metadata parsing error: {e}")
            return {}