# extractors/llama_mtmd_vl_extractor.py - Complete fixed version

import os
import logging
import subprocess
import tempfile
import json
import signal
import platform
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import shutil
import re

class LlamaMtmdVLExtractor:
    """
    Vision Language Model extractor using llama-mtmd-cli.
    NOTE: Has known issues with large images. Use MLXVLMExtractor instead for better performance.
    """
    
    def __init__(self, import_cache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._debug = debug
        self._import_cache = import_cache
        self._binary_paths = binary_paths or {}
        
        # Configuration
        self.model_id = None
        self.max_tokens = 2048  # REDUCED from 4096
        self.temperature = 0.1
        self.min_p = 0.15
        self.repetition_penalty = 1.05
        self._custom_sampling = False
        self.use_gpu = True
        self.ctx_size = 8192  # NEW: Set context size
        
        # Check for llama-mtmd-cli
        self._cli_path = shutil.which('llama-mtmd-cli')
        if not self._cli_path:
            logging.warning("llama-mtmd-cli not found in PATH")
        
        # Import shutdown handling
        try:
            from utils import shutdown_flag
            self._shutdown_flag = shutdown_flag
        except ImportError:
            self._shutdown_flag = None
            logging.warning("Could not import shutdown_flag, Ctrl+C may not work properly")
    
    def set_model(self, model_id: str):
        """Set the HuggingFace model ID to use."""
        self.model_id = model_id
        if self._debug:
            logging.debug(f"LlamaMtmdVLExtractor: Set model to {model_id}")
    
    def set_sampling_params(self, temperature: Optional[float] = None, 
                           min_p: Optional[float] = None,
                           repetition_penalty: Optional[float] = None,
                           max_tokens: Optional[int] = None):
        """Set custom sampling parameters."""
        if temperature is not None:
            self.temperature = temperature
        if min_p is not None:
            self.min_p = min_p
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty
        if max_tokens is not None:
            self.max_tokens = max_tokens
        self._custom_sampling = True
    
    def extract_text(self, input_path: str, 
                    progress_callback: Optional[Callable] = None,
                    **kwargs) -> str:
        """Extract text from document using VL model as OCR."""
        if not self._cli_path:
            logging.error("llama-mtmd-cli not available")
            return ""
        
        if not self.model_id:
            logging.error("No model specified for LlamaMtmdVLExtractor")
            return ""
        
        try:
            file_ext = os.path.splitext(input_path)[1].lower()
            
            if file_ext == '.pdf':
                return self._extract_from_pdf(input_path, progress_callback)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']:
                return self._extract_from_image(input_path)
            else:
                logging.warning(f"Unsupported file type for VL extraction: {file_ext}")
                return ""
                
        except Exception as e:
            logging.error(f"LlamaMtmdVL extraction failed: {e}", exc_info=self._debug)
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
            # CRITICAL: Reduce DPI to reduce image tokens
            conversion_kwargs = {
                'dpi': 150,  # REDUCED from 200
                'fmt': 'jpeg',
                'thread_count': 1,
                'jpegopt': {'quality': 85}  # REDUCED from 90
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
                    # Check shutdown flag
                    if self._shutdown_flag and self._shutdown_flag.is_set():
                        logging.info("Shutdown flag set, stopping PDF extraction")
                        break
                    
                    temp_image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                    image.save(temp_image_path, 'JPEG', quality=85)
                    
                    page_text = self._extract_from_image(temp_image_path)
                    if page_text.strip():
                        text_parts.append(page_text.strip())
                    
                    if progress_callback:
                        progress_callback(1)
                    
                    # Clean up
                    image.close()
            
            return "\n\n---PAGE BREAK---\n\n".join(text_parts)
            
        except Exception as e:
            logging.error(f"PDF to VL extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def _extract_from_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Extract text from a single image using llama-mtmd-cli."""
        if not prompt:
            prompt = (
                "Extract all text from this image. "
                "Transcribe everything you see, maintaining the original layout and structure. "
                "Include headings, paragraphs, lists, and any other text elements."
            )
        
        try:
            # Check shutdown before starting
            if self._shutdown_flag and self._shutdown_flag.is_set():
                return ""
            
            # Build command with CRITICAL fixes
            cmd = [
                self._cli_path,
                '-hf', self.model_id,
                '--image', image_path,
                '-p', prompt,
                '--temp', str(self.temperature),
                '-n', str(self.max_tokens),
                '--min-p', str(self.min_p),
                '--repeat-penalty', str(self.repetition_penalty),
                # CRITICAL: Add context size
                '-c', str(self.ctx_size),
                # CRITICAL: Reduce batch size
                '-b', '512',
                '-ub', '256'
            ]
            
            # Add GPU control if configured
            if not self.use_gpu:
                cmd.append('--no-mmproj-offload')
            
            if self._debug:
                logging.debug(f"Running: {' '.join(cmd)}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=None if platform.system() == 'Windows' else os.setsid
            )
            
            # CRITICAL: Poll with timeout and check shutdown flag
            poll_interval = 0.5  # Check every 0.5 seconds
            timeout = 300  # 5 minute total timeout
            elapsed = 0
            
            while elapsed < timeout:
                # Check if process finished
                if process.poll() is not None:
                    break
                
                # Check shutdown flag
                if self._shutdown_flag and self._shutdown_flag.is_set():
                    logging.info("Shutdown detected, killing llama-mtmd-cli")
                    try:
                        if platform.system() != 'Windows':
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        else:
                            process.terminate()
                        process.wait(timeout=2)
                    except:
                        process.kill()
                    return ""
                
                # Wait a bit
                import time
                time.sleep(poll_interval)
                elapsed += poll_interval
            
            # Get output
            if process.poll() is None:
                # Timeout
                logging.error(f"llama-mtmd-cli timeout for {image_path}")
                try:
                    if platform.system() != 'Windows':
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    else:
                        process.terminate()
                    process.wait(timeout=2)
                except:
                    process.kill()
                return ""
            
            # Read output
            stdout, stderr = process.communicate(timeout=1)
            returncode = process.returncode
            
            if returncode != 0:
                logging.error(f"llama-mtmd-cli failed with return code {returncode}")
                try:
                    stderr_text = stderr.decode('utf-8', errors='replace')
                    # Extract just the error lines
                    error_lines = [l for l in stderr_text.split('\n') 
                                 if 'error' in l.lower() or 'failed' in l.lower()]
                    if error_lines:
                        logging.error(f"Error details: {'; '.join(error_lines[:5])}")
                    else:
                        # Show last few lines
                        lines = stderr_text.strip().split('\n')
                        logging.error(f"Last stderr lines: {'; '.join(lines[-5:])}")
                except:
                    logging.error(f"Could not decode stderr")
                return ""
            
            # Decode stdout
            try:
                output = stdout.decode('utf-8', errors='replace')
            except Exception as e:
                logging.error(f"Failed to decode output: {e}")
                return ""
            
            output = output.strip()
            
            if not output:
                logging.warning(f"llama-mtmd-cli returned empty output for {image_path}")
                return ""
            
            # Filter system messages
            lines = output.split('\n')
            content_lines = []
            skip_patterns = [
                'load_backend', 'llama_model_load', 'llm_load', 'llama_init_from_gpt_params',
                'llama_kv_cache_init', 'llama_new_context_with_model', 'system info:',
                'llama_perf', 'ggml_backend', 'compute buffer size', 'BLAS =',
                'sampling:', 'generate:', 'prompt eval time', 'eval time',
                'sampled tokens', 'encode:', 'decode:', 'ggml_metal'
            ]
            
            for line in lines:
                line_lower = line.lower()
                if any(pattern in line_lower for pattern in skip_patterns):
                    continue
                if not content_lines and not line.strip():
                    continue
                content_lines.append(line)
            
            output = '\n'.join(content_lines).strip()
            
            if self._debug:
                logging.debug(f"VL output length: {len(output)} chars")
            
            return output
            
        except Exception as e:
            logging.error(f"Image VL extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def extract_metadata_direct(self, input_path: str) -> Dict[str, Any]:
        """
        Extract metadata directly using VL model with targeted prompting.
        Returns dict with title, author, year, etc.
        """
        if not self._cli_path or not self.model_id:
            return {}
        
        try:
            # Get representative image(s) from document
            file_ext = os.path.splitext(input_path)[1].lower()
            
            if file_ext == '.pdf':
                # Extract first few pages as images
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
            
            # Try first image (usually has most metadata)
            metadata_text = self._extract_from_image(images[0], prompt=metadata_prompt)
            
            # Parse JSON response
            metadata = self._parse_metadata_json(metadata_text)
            
            # Clean up temp images if we created them
            for img in images:
                if img != input_path and os.path.exists(img):
                    try:
                        os.unlink(img)
                    except:
                        pass
            
            return metadata
            
        except Exception as e:
            logging.error(f"Direct metadata extraction failed: {e}", exc_info=self._debug)
            return {}
    
    def _get_pdf_sample_images(self, pdf_path: str, max_pages: int = 3) -> List[str]:
        """Extract first few pages of PDF as temp image files."""
        pdf2image = self._import_cache.import_module('pdf2image')
        if not pdf2image:
            return []
        
        temp_images = []
        temp_dir = tempfile.mkdtemp(prefix='llama_mtmd_')
        
        try:
            poppler_path = self._binary_paths.get('pdftoppm')
            conversion_kwargs = {
                'dpi': 200,
                'fmt': 'jpeg',
                'first_page': 1,
                'last_page': max_pages,
                'jpegopt': {'quality': 90}
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
            # Clean up on error
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
            # Try to find JSON in the output
            # Look for {...} pattern
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
                    # Extract 4-digit year
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