# extractors/nanonets_ocr2_extractor.py
import os
import logging
import tempfile
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from tqdm import tqdm

try:
    from utils import ImportCache, shutdown_flag
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py: {e}")
    raise

class NanonetsOCR2Extractor:
    """
    Nanonets-OCR2 extractor with full support for:
    - Transformers (HuggingFace)
    - vLLM server
    - llama.cpp GGUF (with mmproj)
    - Ollama
    """
    
    # Official HuggingFace models
    DEFAULT_MODEL = "nanonets/Nanonets-OCR2-3B"
    ALTERNATIVE_MODELS = {
        "3b": "nanonets/Nanonets-OCR2-3B",
        "1.5b": "nanonets/Nanonets-OCR2-1.5B-exp"
    }
    
    # GGUF models from mradermacher (quantized versions)
    GGUF_MODELS = {
        "3b_q4": "mradermacher/Nanonets-OCR2-3B-GGUF",
        "3b_q8": "mradermacher/Nanonets-OCR2-3B-GGUF",
        "3b_f16": "mradermacher/Nanonets-OCR2-3B-GGUF"
    }
    
    GGUF_FILES = {
        # Model files
        "3b_q4": "Nanonets-OCR2-3B.Q4_K_M.gguf",
        "3b_q8": "Nanonets-OCR2-3B.Q8_0.gguf",
        "3b_f16": "Nanonets-OCR2-3B.f16.gguf",
        # Vision projection files (critical for multimodal)
        "mmproj_q8": "Nanonets-OCR2-3B.mmproj-Q8_0.gguf",
        "mmproj_f16": "Nanonets-OCR2-3B.mmproj-f16.gguf"
    }
    
    # Official Ollama models from benhaotang
    OLLAMA_MODELS = {
        "nanonets_s": "benhaotang/Nanonets-OCR-s:latest",
        "nanonets_s_q4": "benhaotang/Nanonets-OCR-s:q4_k_m",
        "nanonets_s_f16": "benhaotang/Nanonets-OCR-s:F16"
    }
    
    # Official prompt template from documentation
    DEFAULT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False,
                 model_name: Optional[str] = None,
                 use_gguf: bool = False,
                 # gguf_model_path is intentionally removed; model_name is used for repo_id or local path
                 gguf_mmproj_path: Optional[str] = None,
                 gguf_quantization: str = "q4",
                 gguf_chat_format: str = "llava-1.5",
                 use_ollama: bool = False,
                 ollama_model: Optional[str] = None,
                 ollama_host: Optional[str] = None,
                 use_vllm_server: bool = False,
                 vllm_base_url: Optional[str] = None,
                 device_map: str = "auto",
                 n_ctx: int = 8192,
                 n_gpu_layers: int = -1,
                 binary_paths: Optional[Dict[str, str]] = None):
        
        self._import_cache = import_cache
        self._debug = debug
        self._model_name = model_name or self.DEFAULT_MODEL
        
        # GGUF support
        self._use_gguf = use_gguf
        self._gguf_model_path = None  # Resolved later in _init_llama_cpp_model to ensure consistency
        self._gguf_mmproj_path = gguf_mmproj_path
        self._gguf_quantization = gguf_quantization
        self._gguf_chat_format = gguf_chat_format
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        
        # Ollama support
        self._use_ollama = use_ollama
        self._ollama_model = ollama_model or self.OLLAMA_MODELS["nanonets_s"]
        self._ollama_host = ollama_host
        
        # Other modes
        self._use_vllm_server = use_vllm_server
        self._vllm_base_url = vllm_base_url or "http://localhost:8000/v1"
        self._device_map = device_map
        self._binary_paths = binary_paths or {}
        
        # Model components (loaded lazily)
        self._model = None
        self._processor = None
        self._vllm_client = None
        self._llama_cpp_model = None
        self._ollama_client = None
        
        self._available_methods: Optional[Dict[str, bool]] = None
        
        if self._debug:
            logging.debug(f"NanonetsOCR2Extractor initialized:")
            logging.debug(f"  Model: {self._model_name}")
            logging.debug(f"  GGUF mode: {self._use_gguf} (quant: {self._gguf_quantization}, chat: {self._gguf_chat_format})")
            logging.debug(f"  Ollama mode: {self._use_ollama}")
            logging.debug(f"  vLLM server mode: {self._use_vllm_server}")

            # Auto-detect Ollama model format
            if not self._use_ollama and not self._use_gguf:
                # Check if model name looks like an Ollama model
                if model_name and ':' in model_name and '/' in model_name:
                    # Format like "benhaotang/Nanonets-OCR-s:q4_k_m"
                    self._use_ollama = True
                    self._ollama_model = model_name
                    if self._debug:
                        logging.debug(f"  Auto-detected Ollama model format: {model_name}")
    
    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            self._available_methods = {
                'transformers': self._check_transformers_available(),
                'vllm_server': self._check_vllm_server_available(),
                'llama_cpp_gguf': self._check_llama_cpp_available(),
                'ollama': self._check_ollama_available(),
                'pdf2image': self._check_pdf2image_available()
            }
            if self._debug:
                logging.debug(f"NanonetsOCR2 available methods: {self._available_methods}")
        return self._available_methods
    
    def _check_transformers_available(self) -> bool:
        """Check if transformers and required dependencies are available"""
        if self._use_gguf or self._use_ollama:
            return False
        return (self._import_cache.is_available('transformers') and
                self._import_cache.is_available('PIL') and
                self._import_cache.is_available('torch'))
    
    def _check_vllm_server_available(self) -> bool:
        """Check if vLLM server is reachable"""
        if not self._use_vllm_server:
            return False
        try:
            import requests
            health_url = self._vllm_base_url.rstrip('/v1') + '/health'
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _check_llama_cpp_available(self) -> bool:
        """Check if llama.cpp with vision support is available"""
        if not self._use_gguf:
            return False
        return self._import_cache.is_available('llama_cpp')
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        if not self._use_ollama:
            return False
        return self._import_cache.is_available('ollama')
    
    def _check_pdf2image_available(self) -> bool:
        """Check if pdf2image is available"""
        return self._import_cache.is_available('pdf2image')
    
    def _init_transformers_model(self):
        """Initialize transformers model"""
        if self._model is not None:
            return True
        
        if not self.available_methods['transformers']:
            logging.error("Transformers not available")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
            import torch
            
            logging.info(f"Loading Nanonets-OCR2 transformers model: {self._model_name}")
            
            # Determine torch dtype
            torch_dtype = "auto"
            if torch.cuda.is_available():
                torch_dtype = torch.float16
            elif torch.backends.mps.is_available():
                torch_dtype = torch.float16
            
            # Load model
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                device_map=self._device_map,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )
            self._model.eval()
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            
            logging.info("Nanonets-OCR2 transformers model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load transformers model: {e}", exc_info=self._debug)
            return False
    
    def _init_llama_cpp_model(self):
        """Initialize llama.cpp model with vision support (mmproj)"""
        if self._llama_cpp_model is not None:
            return True
        
        if not self.available_methods['llama_cpp_gguf']:
            logging.error("llama-cpp-python not available")
            return False
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            
            # Download GGUF models if not provided
            if not self._gguf_model_path:
                self._gguf_model_path = self._download_gguf_model()
            
            if not self._gguf_mmproj_path:
                self._gguf_mmproj_path = self._download_mmproj_model()
            
            if not self._gguf_model_path or not self._gguf_mmproj_path:
                logging.error("Failed to obtain GGUF model files")
                return False
            
            logging.info(f"Loading Nanonets-OCR2 GGUF model ({self._gguf_quantization})...")
            logging.info(f"  Model: {self._gguf_model_path}")
            logging.info(f"  MMProj: {self._gguf_mmproj_path}")
            
            # Auto-detect GPU layers
            n_gpu_layers = self._n_gpu_layers
            if n_gpu_layers == -1:
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        n_gpu_layers = 33  # All layers for 3B model
                        logging.info("  Using Metal (MPS) acceleration")
                    elif torch.cuda.is_available():
                        # Check available VRAM
                        try:
                            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            if vram_gb >= 8:
                                n_gpu_layers = 33  # All layers
                            elif vram_gb >= 4:
                                n_gpu_layers = 20  # Partial offload
                            else:
                                n_gpu_layers = 10  # Minimal offload
                            logging.info(f"  Using CUDA acceleration ({vram_gb:.1f}GB VRAM, {n_gpu_layers} layers)")
                        except:
                            n_gpu_layers = 33  # Default to all
                            logging.info("  Using CUDA acceleration")
                    else:
                        n_gpu_layers = 0
                        logging.info("  Using CPU only (this will be slow)")
                except ImportError:
                    n_gpu_layers = 0
                    logging.info("  PyTorch not available, using CPU only")
            
            # Initialize chat handler with mmproj (critical for vision)
            chat_handler = Llava15ChatHandler(
                clip_model_path=self._gguf_mmproj_path,
                verbose=self._debug
            )
            
            # Load model
            self._llama_cpp_model = Llama(
                model_path=self._gguf_model_path,
                chat_handler=chat_handler,
                n_ctx=self._n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=self._debug,
                logits_all=True,
                n_threads=8
            )
            
            logging.info("Nanonets-OCR2 GGUF model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load GGUF model: {e}", exc_info=self._debug)
            return False
    
    def _download_gguf_model(self) -> Optional[str]:
        """Download GGUF model from HuggingFace"""
        try:
            from huggingface_hub import hf_hub_download
            
            cache_dir = Path.home() / ".cache" / "biblioforge_nanonets_gguf"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Select quantization
            quant_key = f"3b_{self._gguf_quantization}"
            if quant_key not in self.GGUF_FILES:
                logging.warning(f"Unknown quantization: {self._gguf_quantization}, using q4")
                quant_key = "3b_q4"
            
            repo_id = self.GGUF_MODELS["3b_q4"]  # Same repo for all quants
            filename = self.GGUF_FILES[quant_key]
            
            logging.info(f"Downloading GGUF model: {repo_id}/{filename}")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                resume_download=True
            )
            return model_path
        except Exception as e:
            logging.error(f"Failed to download GGUF model: {e}")
            return None
    
    def _download_mmproj_model(self) -> Optional[str]:
        """Download mmproj (vision projection) model - CRITICAL for multimodal"""
        try:
            from huggingface_hub import hf_hub_download
            
            cache_dir = Path.home() / ".cache" / "biblioforge_nanonets_gguf"
            
            repo_id = self.GGUF_MODELS["3b_q4"]
            
            # Match mmproj quantization to model quantization
            if self._gguf_quantization == "f16":
                filename = self.GGUF_FILES["mmproj_f16"]
            else:
                filename = self.GGUF_FILES["mmproj_q8"]  # Default to Q8 for Q4/Q8 models
            
            logging.info(f"Downloading MMProj (vision) model: {repo_id}/{filename}")
            logging.info("  This is required for image/document processing...")
            
            try:
                mmproj_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                    resume_download=True
                )
                logging.info(f"✓ MMProj downloaded to: {mmproj_path}")
                return mmproj_path
            except KeyboardInterrupt:
                logging.warning("MMProj download interrupted by user")
                raise
            except Exception as e:
                logging.error(f"Failed to download MMProj model: {e}")
                logging.error("  Without MMProj, vision models cannot process images!")
                return None
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error(f"Error in MMProj download setup: {e}")
            return None
    
    def _init_ollama_client(self):
        """Initialize Ollama client and ensure model is available"""
        if self._ollama_client is not None:
            return True
        
        try:
            import ollama
            
            client_kwargs = {'host': self._ollama_host} if self._ollama_host else {}
            
            # Check if model exists - simple approach that works with all formats
            try:
                # Try to show the model - if it exists, this succeeds
                model_info = ollama.show(self._ollama_model, **client_kwargs)
                logging.info(f"Ollama model '{self._ollama_model}' is available")
                self._ollama_client = ollama
                return True
            except Exception as show_error:
                # Model doesn't exist, need to pull it
                logging.info(f"Pulling Ollama model '{self._ollama_model}'...")
                
                # Pull with progress tracking
                try:
                    pull_stream = ollama.pull(self._ollama_model, stream=True, **client_kwargs)
                    
                    last_status = None
                    for chunk in pull_stream:
                        if shutdown_flag.is_set():
                            raise InterruptedError("Model pull cancelled by user")
                        
                        # Log progress
                        if isinstance(chunk, dict):
                            status = chunk.get('status', '')
                            if status != last_status:
                                logging.info(f"  {status}")
                                last_status = status
                    
                    logging.info(f"✓ Model '{self._ollama_model}' pulled successfully")
                    self._ollama_client = ollama
                    return True
                    
                except InterruptedError:
                    logging.warning("Model pull interrupted by user")
                    raise
                except Exception as pull_error:
                    logging.error(f"Failed to pull model '{self._ollama_model}': {pull_error}")
                    return False
        
        except InterruptedError:
            raise
        except Exception as e:
            logging.error(f"Failed to initialize Ollama: {e}", exc_info=self._debug)
            return False
    
    def _init_vllm_client(self):
        """Initialize vLLM OpenAI-compatible client"""
        if self._vllm_client is not None:
            return True
        
        try:
            from openai import OpenAI
            
            self._vllm_client = OpenAI(
                api_key="EMPTY",  # vLLM doesn't require API key
                base_url=self._vllm_base_url
            )
            
            logging.info(f"vLLM client initialized: {self._vllm_base_url}")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize vLLM client: {e}", exc_info=self._debug)
            return False
    
    def extract_text(self, file_path: str,
                    preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None,
                    max_tokens: int = 15000,
                    prompt_template: Optional[str] = None,
                    **kwargs) -> str:
        """Extract text using Nanonets-OCR2"""
        
        # Validate file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""
        
        file_basename = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Use official prompt if none provided
        if prompt_template is None:
            prompt_template = self.DEFAULT_PROMPT
        
        # Determine processing method
        if self._use_ollama and self.available_methods['ollama']:
            method = 'ollama'
        elif self._use_gguf and self.available_methods['llama_cpp_gguf']:
            method = 'llama_cpp_gguf'
        elif self._use_vllm_server and self.available_methods['vllm_server']:
            method = 'vllm_server'
        elif self.available_methods['transformers']:
            method = 'transformers'
        else:
            logging.error(f"No available method for Nanonets-OCR2 extraction")
            return ""
        
        if self._debug:
            logging.debug(f"Using {method} method for {file_basename}")
        
        try:
            # Handle PDF files
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_path, method, progress_callback, 
                                              max_tokens, prompt_template)
            else:
                return self._extract_from_image(file_path, method, progress_callback,
                                               max_tokens, prompt_template)
        
        except Exception as e:
            logging.error(f"Nanonets-OCR2 extraction failed for {file_basename}: {e}",
                         exc_info=self._debug)
            return ""
    
    def _extract_from_image(self, image_path: str, method: str,
                           progress_callback: Optional[Callable],
                           max_tokens: int,
                           prompt_template: str) -> str:
        """Extract text from a single image"""
        
        if progress_callback:
            progress_callback(1, f"nanonets_ocr2_{method}_processing")
        
        if method == 'ollama':
            result = self._extract_with_ollama(image_path, max_tokens, prompt_template)
        elif method == 'llama_cpp_gguf':
            result = self._extract_with_llama_cpp(image_path, max_tokens, prompt_template)
        elif method == 'vllm_server':
            result = self._extract_with_vllm(image_path, max_tokens, prompt_template)
        else:  # transformers
            result = self._extract_with_transformers(image_path, max_tokens, prompt_template)
        
        if progress_callback:
            progress_callback(1, f"nanonets_ocr2_{method}_complete")
        
        return result
    
    def _extract_with_llama_cpp(self, image_path: str, max_tokens: int,
                                prompt_template: str) -> str:
        """Extract using llama.cpp GGUF"""
        
        if not self._init_llama_cpp_model():
            return ""
        
        try:
            # Prepare message with image (llama.cpp format)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                        {"type": "text", "text": prompt_template}
                    ]
                }
            ]
            
            # Generate response
            response = self._llama_cpp_model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0  # Deterministic for OCR
            )
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            logging.error(f"llama.cpp extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def _extract_with_ollama(self, image_path: str, max_tokens: int,
                            prompt_template: str) -> str:
        """Extract using Ollama"""
        
        if not self._init_ollama_client():
            return ""
        
        try:
            client_kwargs = {'host': self._ollama_host} if self._ollama_host else {}
            
            # Verify image exists
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                return ""
            
            if self._debug:
                logging.debug(f"Ollama: Processing {os.path.basename(image_path)} with model {self._ollama_model}")
            
            # Generate response
            response = self._ollama_client.generate(
                model=self._ollama_model,
                prompt=prompt_template,
                images=[image_path],
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.0  # Deterministic for OCR
                },
                **client_kwargs
            )
            
            # Extract text from response
            if isinstance(response, dict) and 'response' in response:
                result_text = response['response']
                if self._debug:
                    logging.debug(f"Ollama: Extracted {len(result_text)} characters")
                return result_text
            else:
                logging.warning(f"Ollama: Unexpected response format: {type(response)}")
                return str(response) if response else ""
            
        except Exception as e:
            logging.error(f"Ollama extraction failed for {os.path.basename(image_path)}: {e}", 
                        exc_info=self._debug)
            return ""
    
    def _extract_with_transformers(self, image_path: str, max_tokens: int,
                                   prompt_template: str) -> str:
        """Extract using transformers library"""
        
        if not self._init_transformers_model():
            return ""
        
        try:
            from PIL import Image
            
            image = Image.open(image_path)
            
            # Official message format from documentation
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt_template},
                ]},
            ]
            
            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self._processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self._model.device)
            
            # Generate
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False  # Deterministic for OCR
            )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            output_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return output_text[0]
            
        except Exception as e:
            logging.error(f"Transformers extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def _extract_with_vllm(self, image_path: str, max_tokens: int,
                          prompt_template: str) -> str:
        """Extract using vLLM server"""
        
        if not self._init_vllm_client():
            return ""
        
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            
            response = self._vllm_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                            {
                                "type": "text",
                                "text": prompt_template,
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"vLLM extraction failed: {e}", exc_info=self._debug)
            return ""
    
    def _extract_from_pdf(self, pdf_path: str, method: str,
                         progress_callback: Optional[Callable],
                         max_tokens: int,
                         prompt_template: str) -> str:
        """Extract text from PDF by converting pages to images"""
        
        if not self.available_methods['pdf2image']:
            logging.error("pdf2image not available for PDF processing")
            logging.error("Install with: pip install pdf2image")
            logging.error("Also requires poppler: https://pdf2image.readthedocs.io/en/latest/installation.html")
            return ""
        
        try:
            from pdf2image import convert_from_path
            
            # Get poppler path if available
            poppler_path = None
            if 'pdftoppm' in self._binary_paths:
                poppler_path = os.path.dirname(self._binary_paths['pdftoppm'])
            
            logging.info(f"Converting PDF to images: {pdf_path}")
            
            images = convert_from_path(
                pdf_path,
                dpi=300,
                poppler_path=poppler_path
            )
            
            total_pages = len(images)
            text_parts = []
            
            with tqdm(total=total_pages, desc="Nanonets-OCR2 Pages", 
                     unit="page", leave=False, position=2) as pbar:
                
                for page_num, image in enumerate(images, 1):
                    if shutdown_flag.is_set():
                        break
                    
                    pbar.set_description(f"Nanonets-OCR2 Page {page_num}/{total_pages}")
                    
                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        temp_image_path = tmp.name
                        image.save(temp_image_path, 'PNG')
                    
                    try:
                        page_text = self._extract_from_image(
                            temp_image_path, method, None, max_tokens, prompt_template
                        )
                        
                        if page_text and page_text.strip():
                            text_parts.append(f"## Page {page_num}\n\n{page_text}")
                        
                    finally:
                        try:
                            os.unlink(temp_image_path)
                        except:
                            pass
                    
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(1)
            
            return "\n\n---\n\n".join(text_parts)
            
        except Exception as e:
            logging.error(f"PDF processing failed: {e}", exc_info=self._debug)
            return ""
    
    def extract_metadata_fields(self, image_path: str, 
                               fields: List[str]) -> Dict[str, Any]:
        """
        Extract specific metadata fields using VQA capabilities.
        """
        try:
            results = {}
            
            for field in fields:
                question = self._build_vqa_question(field)
                
                # Use the same extraction pipeline but with VQA prompt
                if self._use_ollama and self.available_methods['ollama']:
                    answer = self._extract_with_ollama(image_path, 100, question)
                elif self._use_gguf and self.available_methods['llama_cpp_gguf']:
                    answer = self._extract_with_llama_cpp(image_path, 100, question)
                elif self._use_vllm_server and self.available_methods['vllm_server']:
                    answer = self._extract_with_vllm(image_path, 100, question)
                else:
                    answer = self._extract_with_transformers(image_path, 100, question)
                
                if answer and answer.lower() != "not mentioned":
                    results[field] = answer
            
            return results
            
        except Exception as e:
            logging.error(f"Metadata extraction failed: {e}", exc_info=self._debug)
            return {}
    
    def _build_vqa_question(self, field: str) -> str:
        """Build appropriate VQA question for a metadata field"""
        question_templates = {
            'title': "What is the title of this document?",
            'author': "Who is the author of this document?",
            'authors': "Who are the authors of this document?",
            'year': "What is the publication year of this document?",
            'date': "What is the publication date of this document?",
            'publisher': "Who is the publisher of this document?",
            'language': "What language is this document written in?",
        }
        
        question = question_templates.get(field.lower(), 
                                         f"What is the {field} of this document?")
        return f"{question} If the information is not present in the document, respond with 'Not mentioned'."