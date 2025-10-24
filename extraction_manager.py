import os
import logging
import shutil
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable, Tuple
from contextlib import contextmanager
import traceback
import pkg_resources # For version checking
import threading

from tqdm import tqdm

# Imports from our new modules
from extractors import ( # Direct import of the 'extractors' package/module
    PDFExtractor, EPUBExtractor, DJVUExtractor, 
    MOBIExtractor, TextExtractor, HTMLExtractor, PPTXExtractor, LlamaMtmdVLExtractor
)
from utils import (     # Direct import of 'utils' module
    ImportCache, parse_metadata, sanitize_filename, 
    add_rename_command, validate_and_fix_year, file_lock, shutdown_flag
)
from llm_providers import ( # Direct import
    extract_metadata as llm_extract_metadata, 
    sort_author_names, 
    get_llm_provider, 
    LLMProvider 
)

class ExtractionManager:
    """Central manager for text extraction operations"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF', '.epub': 'EPUB', '.djvu': 'DJVU', '.djv': 'DJVU',
        '.mobi': 'MOBI', '.azw': 'MOBI', '.azw3': 'MOBI', '.azw4': 'MOBI',
        '.txt': 'Text', '.text': 'Text', '.md': 'Text',
        '.html': 'HTML', '.htm': 'HTML', '.xhtml': 'HTML',
        '.docx': 'Text', '.doc': 'Text', '.rtf': 'Text', '.fb2': 'Text',
        '.pdb': 'Text', '.lit': 'Text', '.odt': 'Text', '.lrf': 'Text',
        '.cbz': 'Text', '.cbr': 'Text', '.chm': 'Text', '.snb': 'Text', '.tcr': 'Text',
        '.pptx': 'PPTX', '.ppt': 'PPTX',
    }

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._binary_paths, self._binaries = self._check_system_dependencies()
        self._versions = self._check_python_package_versions()
        self._extractors_cache: Dict[str, Any] = {} 
        self._import_cache = ImportCache()
        self._extractor_init_lock = threading.Lock()

    def _check_system_dependencies(self) -> Tuple[Dict[str, str], Dict[str, bool]]:
        # (This method should be fully implemented here as it was in the original script,
        #  or in utils.py and called from here. Assuming a complete implementation.)
        binary_paths_found: Dict[str, Optional[str]] = {
            'tesseract': None, 'pdftoppm': None, 'gs': None,
            'djvutxt': None, 'ddjvu': None, 'ebook-converter': None
        }
        executable_names = {
            'tesseract': ['tesseract', 'tesseract.exe'],
            'pdftoppm': ['pdftoppm', 'pdftoppm.exe'],
            'gs': ['gs', 'gswin64c', 'gswin64c.exe', 'gswin32c.exe'],
            'djvutxt': ['djvutxt', 'djvutxt.exe'],
            'ddjvu': ['ddjvu', 'ddjvu.exe'],
            'ebook-converter': ['ebook-converter', 'ebook-convert', 'ebook-converter.exe', 'ebook-convert.exe']
        }
        
        # Simplified check for brevity - use your full _check_system_dependencies logic here
        for binary_key, names in executable_names.items():
            for name in names:
                path = shutil.which(name)
                if path:
                    binary_paths_found[binary_key] = path
                    if self._debug: logging.debug(f"ExtractionManager: Found {binary_key} at {path}")
                    break
        
        # Filter out None values for the path dictionary to be returned
        # The class attribute self._binary_paths will store these.
        # This method needs to return Dict[str, str] for paths where key exists only if path is found.
        final_binary_paths: Dict[str, str] = {k: v for k, v in binary_paths_found.items() if v is not None}

        binaries_bool = {
            'tesseract': bool(final_binary_paths.get('tesseract')),
            'poppler': bool(final_binary_paths.get('pdftoppm')),
            'ghostscript': bool(final_binary_paths.get('gs')),
            'djvulibre': bool(final_binary_paths.get('djvutxt')) or bool(final_binary_paths.get('ddjvu')),
            'calibre': bool(final_binary_paths.get('ebook-converter'))
        }
        return final_binary_paths, binaries_bool


    def _check_python_package_versions(self) -> Dict[str, str]:
        # (Implementation from previous step, ensure it's correct)
        versions = {}
        packages = [
            'pymupdf', 'pdfplumber', 'pypdf', 'pdfminer.six', 'pytesseract', 
            'pdf2image', 'easyocr', 'paddleocr', 'python-doctr', 'ocrmypdf', 
            'camelot-py', 'ebooklib', 'beautifulsoup4', 'html2text', 'kraken',
            'mobi', 'kindleunpack', 'chardet', 'ftfy', 'lxml', 
            'openai', 'requests', 'groq', 'cohere', 'huggingface_hub'
        ]
        # Using importlib.metadata (preferred) with a fallback to pkg_resources
        try:
            from importlib.metadata import version as get_pkg_version, PackageNotFoundError
            for package in packages:
                try:
                    versions[package] = get_pkg_version(package)
                except PackageNotFoundError:
                    if self._debug: logging.debug(f"Package {package} not found by importlib.metadata.")
        except ImportError: # Fallback for Python < 3.8 or if pkg_resources is still needed
            import pkg_resources
            for package in packages:
                try:
                    versions[package] = pkg_resources.get_distribution(package).version
                except pkg_resources.DistributionNotFound:
                    if self._debug: logging.debug(f"Package {package} not found by pkg_resources.")
                except Exception as e_pkg:
                    if self._debug: logging.debug(f"Error getting version for {package} using pkg_resources: {e_pkg}")
        
        if self._debug:
            for pkg, ver in versions.items():
                    logging.debug(f"Found {pkg} version {ver}")
        return versions


    def _get_extractor(self, file_path: str,
                   use_nanonets_ocr2: bool = False,
                   nanonets_config: Optional[Dict] = None,
                   use_docstrange: bool = False,
                   docstrange_config: Optional[Dict] = None,
                   use_llama_mtmd: bool = False, 
                   llama_mtmd_config: Optional[Dict] = None,
                   use_mlx_vlm: bool = False,
                   mlx_vlm_config: Optional[Dict] = None) -> Optional[Any]: 

        """
        Get appropriate extractor for the file with robust, thread-safe caching.
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # 1. Determine the unique cache key based on the configuration
        # Set cache_key based on extractor type FIRST
        cache_key = file_ext  # Default for standard extractors
        extractor_type_str = "standard"  # For logging
        
        if self._debug:
            logging.debug(f"_get_extractor: Determining extractor for '{file_path}'")
            logging.debug(f"_get_extractor: use_mlx_vlm={use_mlx_vlm}, use_llama_mtmd={use_llama_mtmd}, use_docstrange={use_docstrange}, use_nanonets={use_nanonets_ocr2}")
            
        # Determine cache key and extractor type
        # PRIORITY ORDER: mlx-vlm > llama-mtmd > docstrange > nanonets > standard
        if use_mlx_vlm and mlx_vlm_config:
            model_id = mlx_vlm_config.get('model_id', 'mlx-community/LFM2-VL-3B-8bit')
            cache_key = f"mlx_vlm_{model_id.replace('/', '_').replace('-', '_')}"
            extractor_type_str = "mlx-vlm"
            if self._debug:
                logging.debug(f"_get_extractor: Will use MLXVLMExtractor with cache_key='{cache_key}'")
        elif use_llama_mtmd and llama_mtmd_config:
            model_id = llama_mtmd_config.get('model_id', 'LiquidAI/LFM2-VL-3B-GGUF:Q4_0')
            cache_key = f"llama_mtmd_{model_id.replace('/', '_').replace(':', '_')}"
            extractor_type_str = "llama-mtmd-vl"
            if self._debug:
                logging.debug(f"_get_extractor: Will use LlamaMtmdVLExtractor with cache_key='{cache_key}'")
        elif use_docstrange and docstrange_config:
            mode = docstrange_config.get('mode', 'cloud')
            api_key_hash = hash(docstrange_config.get('api_key', 'none'))
            cache_key = f"docstrange_{mode}_{api_key_hash}"
            extractor_type_str = "docstrange"
            if self._debug:
                logging.debug(f"_get_extractor: Will use DocStrangeExtractor with cache_key='{cache_key}'")
        elif use_nanonets_ocr2 and nanonets_config:
            parts = ["nanonets"]
            if nanonets_config.get('use_gguf'):
                parts.extend(['gguf', nanonets_config.get('gguf_quantization', 'q4')])
            elif nanonets_config.get('use_ollama'):
                parts.extend(['ollama', nanonets_config.get('ollama_model', 'default')])
            elif nanonets_config.get('use_vllm_server'):
                parts.extend(['vllm', nanonets_config.get('vllm_base_url')])
            else:
                parts.extend(['transformers', nanonets_config.get('model_name', '3b')])
            cache_key = '_'.join(parts)
            extractor_type_str = "nanonets"
            if self._debug:
                logging.debug(f"_get_extractor: Will use NanonetsOCR2Extractor with cache_key='{cache_key}'")
        else:
            # Standard extractor
            extractor_file_type = self.SUPPORTED_EXTENSIONS.get(file_ext, 'Unknown')
            extractor_type_str = f"standard-{extractor_file_type}"
            if self._debug:
                logging.debug(f"_get_extractor: Will use standard {extractor_file_type}Extractor with cache_key='{cache_key}'")

        # 2. First check (fast path): Return immediately if already cached.
        if cache_key in self._extractors_cache:
            if self._debug:
                logging.debug(f"_get_extractor: Using cached extractor for key '{cache_key}' (type: {extractor_type_str})")
            return self._extractors_cache[cache_key]

        if self._debug:
            logging.debug(f"_get_extractor: Cache miss for key '{cache_key}', will initialize {extractor_type_str} extractor")

        # 3. Lock: Only one thread can proceed beyond this point to initialize.
        with self._extractor_init_lock:
            # 4. Second check (double-check): Another thread might have created it while we waited.
            if cache_key in self._extractors_cache:
                if self._debug:
                    logging.debug(f"_get_extractor: Found cached extractor after acquiring lock for key '{cache_key}'")
                return self._extractors_cache[cache_key]

            # 5. Initialization: This thread is now solely responsible for creating the extractor.
            if self._debug:
                logging.debug(f"_get_extractor: Initializing {extractor_type_str} extractor for cache_key '{cache_key}'...")

            extractor_instance = None
            
            try:
                # We use if/elif/else chain to ensure only ONE extractor is initialized
                
                # PRIORITY 1: llama-mtmd VL extractor
                if use_mlx_vlm and mlx_vlm_config:
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: mlx-vlm] Initializing MLXVLMExtractor...")
                    
                    from extractors import MLXVLMExtractor
                    extractor_instance = MLXVLMExtractor(
                        import_cache=self._import_cache,
                        debug=self._debug,
                        binary_paths=self._binary_paths
                    )
                    
                    model_id = mlx_vlm_config.get('model_id', 'mlx-community/LFM2-VL-3B-8bit')
                    extractor_instance.set_model(model_id)
                    extractor_instance.max_tokens = mlx_vlm_config.get('max_tokens', 512)
                    extractor_instance.temperature = mlx_vlm_config.get('temperature', 0.0)
                    
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: mlx-vlm] ✓ Successfully initialized MLXVLMExtractor with model {model_id}")
                
                elif use_llama_mtmd and llama_mtmd_config:
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: llama-mtmd] Initializing LlamaMtmdVLExtractor...")
                    
                    from extractors import LlamaMtmdVLExtractor
                    extractor_instance = LlamaMtmdVLExtractor(
                        import_cache=self._import_cache,
                        debug=self._debug,
                        binary_paths=self._binary_paths
                    )
                    
                    # Configure the model
                    model_id = llama_mtmd_config.get('model_id', 'LiquidAI/LFM2-VL-3B-GGUF:Q4_0')
                    extractor_instance.set_model(model_id)
                    
                    # Set sampling parameters
                    extractor_instance.set_sampling_params(
                        temperature=llama_mtmd_config.get('temperature'),
                        min_p=llama_mtmd_config.get('min_p'),
                        repetition_penalty=llama_mtmd_config.get('repetition_penalty'),
                        max_tokens=llama_mtmd_config.get('max_tokens')
                    )
                    
                    # GPU control
                    if 'use_gpu' in llama_mtmd_config:
                        extractor_instance.use_gpu = llama_mtmd_config['use_gpu']
                        
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: llama-mtmd] ✓ Successfully initialized LlamaMtmdVLExtractor with model {model_id}")
                
                # PRIORITY 2: DocStrange extractor
                elif use_docstrange and docstrange_config:
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: docstrange] Initializing DocStrangeExtractor...")
                    
                    from extractors import DocStrangeExtractor
                    extractor_instance = DocStrangeExtractor(
                        import_cache=self._import_cache,
                        debug=self._debug,
                        mode=docstrange_config.get('mode', 'cloud'),
                        api_key=docstrange_config.get('api_key'),
                        binary_paths=self._binary_paths
                    )
                    
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: docstrange] ✓ Successfully initialized DocStrangeExtractor")
                
                # PRIORITY 3: Nanonets extractor
                elif use_nanonets_ocr2 and nanonets_config:
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: nanonets] Initializing NanonetsOCR2Extractor...")
                    
                    from extractors import NanonetsOCR2Extractor
                    extractor_instance = NanonetsOCR2Extractor(
                        import_cache=self._import_cache,
                        debug=self._debug,
                        model_name=nanonets_config.get('model_name'),
                        use_gguf=nanonets_config.get('use_gguf', False),
                        gguf_quantization=nanonets_config.get('gguf_quantization', 'q4'),
                        gguf_mmproj_path=nanonets_config.get('gguf_mmproj_path'),
                        gguf_chat_format=nanonets_config.get('gguf_chat_format', 'llava-1.5'),
                        use_ollama=nanonets_config.get('use_ollama', False),
                        ollama_model=nanonets_config.get('ollama_model'),
                        ollama_host=nanonets_config.get('ollama_host'),
                        use_vllm_server=nanonets_config.get('use_vllm_server', False),
                        vllm_base_url=nanonets_config.get('vllm_base_url'),
                        device_map=nanonets_config.get('device_map', 'auto'),
                        n_ctx=nanonets_config.get('n_ctx', 8192),
                        n_gpu_layers=nanonets_config.get('n_gpu_layers', -1),
                        binary_paths=self._binary_paths
                    )
                    
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: nanonets] ✓ Successfully initialized NanonetsOCR2Extractor")
                
                # PRIORITY 4: Standard extractors based on file extension
                else:
                    if self._debug:
                        logging.debug(f"_get_extractor: [BRANCH: standard] Initializing standard extractor for extension '{file_ext}'...")
                    
                    extractor_type = self.SUPPORTED_EXTENSIONS.get(file_ext)
                    if extractor_type:
                        if extractor_type == 'PDF': 
                            from extractors import PDFExtractor
                            extractor_class = PDFExtractor
                        elif extractor_type == 'EPUB': 
                            from extractors import EPUBExtractor
                            extractor_class = EPUBExtractor
                        elif extractor_type == 'DJVU': 
                            from extractors import DJVUExtractor
                            extractor_class = DJVUExtractor
                        elif extractor_type == 'MOBI': 
                            from extractors import MOBIExtractor
                            extractor_class = MOBIExtractor
                        elif extractor_type == 'Text': 
                            from extractors import TextExtractor
                            extractor_class = TextExtractor
                        elif extractor_type == 'HTML': 
                            from extractors import HTMLExtractor
                            extractor_class = HTMLExtractor
                        elif extractor_type == 'PPTX': 
                            from extractors import PPTXExtractor
                            extractor_class = PPTXExtractor
                        else: 
                            extractor_class = None

                        if extractor_class:
                            extractor_instance = extractor_class(
                                import_cache=self._import_cache,
                                debug=self._debug,
                                binary_paths=self._binary_paths
                            )
                            
                            if self._debug:
                                logging.debug(f"_get_extractor: [BRANCH: standard] ✓ Successfully initialized {extractor_class.__name__}")
                        else:
                            if self._debug:
                                logging.debug(f"_get_extractor: [BRANCH: standard] ✗ No extractor class for type '{extractor_type}'")
                    else:
                        if self._debug:
                            logging.debug(f"_get_extractor: [BRANCH: standard] ✗ Unsupported file extension '{file_ext}'")
                            
            except Exception as e:
                logging.error(f"_get_extractor: EXCEPTION while initializing extractor for key '{cache_key}': {e}", exc_info=self._debug)
                extractor_instance = None

            # 6. Cache the result (even if it's None) and release the lock.
            self._extractors_cache[cache_key] = extractor_instance
            
            if self._debug:
                if extractor_instance:
                    logging.debug(f"_get_extractor: ✓ Cached extractor for key '{cache_key}' (type: {type(extractor_instance).__name__})")
                else:
                    logging.debug(f"_get_extractor: ✗ Caching None for key '{cache_key}' (initialization failed)")

        if self._debug:
            logging.debug(f"_get_extractor: Returning extractor for key '{cache_key}': {type(extractor_instance).__name__ if extractor_instance else 'None'}")
        
        return self._extractors_cache[cache_key]

    def extract(self, input_path: str, 
                output_path: Optional[str] = None, 
                method: Optional[str] = None,
                ocr_method: Optional[str] = None,
                password: Optional[str] = None,
                extract_tables: bool = False,
                force_ocr: bool = False,
                sort: bool = False, 
                llm_provider_arg: Optional[Any] = None, 
                rename_script_path: Optional[str] = None,
                temperature: float = 0.5, 
                max_tokens: int = 250,
                **kwargs) -> Dict[str, Any]:
        
        result_data: Dict[str, Any] = {
            "success": False, "text": "", "output_path": output_path, "skipped": False, 
            "error": None, "tables": [], "metadata_llm": None, "renamed_info": None
        }
        
        try:
            # Log which file is being extracted
            logging.info(f"Extracting: {os.path.basename(input_path)}")
            
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext not in self.SUPPORTED_EXTENSIONS:
                result_data["error"] = f"Unsupported file type: {input_path}"
                return result_data

            # Extract configs from kwargs
            nanonets_config = kwargs.get('nanonets_config')
            docstrange_config = kwargs.get('docstrange_config')
            llama_mtmd_config = kwargs.get('llama_mtmd_config')
            mlx_vlm_config = kwargs.get('mlx_vlm_config') 

            use_nanonets = nanonets_config is not None
            use_docstrange = docstrange_config is not None
            use_llama_mtmd = llama_mtmd_config is not None  
            use_mlx_vlm = mlx_vlm_config is not None 
            
            # Get extractor with proper config passing
            extractor = self._get_extractor(
                input_path,
                use_nanonets_ocr2=use_nanonets,
                nanonets_config=nanonets_config,
                use_docstrange=use_docstrange,
                docstrange_config=docstrange_config,
                use_llama_mtmd=use_llama_mtmd,
                llama_mtmd_config=llama_mtmd_config,
                use_mlx_vlm=use_mlx_vlm, 
                mlx_vlm_config=mlx_vlm_config 
            )
            
            if not extractor:
                result_data["error"] = f"No suitable extractor found for {input_path}"
                return result_data

            if hasattr(extractor, 'set_password') and password:
                extractor.set_password(password)
            
            def _internal_progress_cb(n=1, engine_info=None):
                if self._debug and engine_info:
                    logging.debug(f"Extractor ({engine_info}) progress step: {n}")

            # Build extractor kwargs
            extractor_kwargs = {'preferred_method': method}
            
            # For PDF extractor
            from extractors import PDFExtractor
            if isinstance(extractor, PDFExtractor):
                extractor_kwargs.update({
                    'ocr_method': ocr_method,
                    'force_ocr': force_ocr,
                    'extract_tables': extract_tables
                })
            
            # For Nanonets/DocStrange/LlamaMtmd, pass additional params
            from extractors import NanonetsOCR2Extractor, DocStrangeExtractor, LlamaMtmdVLExtractor  # FIXED: Added LlamaMtmdVLExtractor
            if isinstance(extractor, NanonetsOCR2Extractor):
                extractor_kwargs.update({
                    'max_tokens': kwargs.get('max_tokens', 15000),
                    'prompt_template': kwargs.get('prompt_template')
                })
            elif isinstance(extractor, DocStrangeExtractor):
                extractor_kwargs.update({
                    'output_format': docstrange_config.get('output_format', 'markdown') if docstrange_config else 'markdown'
                })
            # No special extractor_kwargs needed for LlamaMtmdVLExtractor - configuration is done during initialization
            
            # Extract text
            extracted_text = extractor.extract_text(
                input_path, 
                progress_callback=_internal_progress_cb,
                **extractor_kwargs
            )

            if extracted_text and extracted_text.strip():
                result_data["text"] = extracted_text
                result_data["success"] = True
                if self._debug: 
                    logging.debug(f"Successfully extracted {len(extracted_text)} chars from {input_path}")
                
                # Handle tables for PDF extractor
                if isinstance(extractor, PDFExtractor) and extract_tables:
                    tables_df_list = extractor.get_last_extracted_tables()
                    if tables_df_list:
                        result_data["tables"] = [df.to_dict('records') for df in tables_df_list]

                # Write output file
                if output_path:
                    try:
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(extracted_text)
                        result_data["output_path"] = output_path
                    except Exception as e_write:
                        result_data["error"] = f"Text file write error: {e_write}"
                        result_data["success"] = False 
            else:
                result_data["error"] = f"No text extracted from {input_path}"
                result_data["success"] = False

            # Handle direct metadata extraction 
            # This is placed OUTSIDE the text extraction success block so it can run even if OCR fails
            # But it's still inside the main try block for proper error handling
            if isinstance(extractor, LlamaMtmdVLExtractor) and llama_mtmd_config.get('extract_metadata'):
                try:
                    if self._debug:
                        logging.debug(f"Attempting VL metadata extraction for {input_path}")
                    vl_metadata = extractor.extract_metadata_direct(input_path)
                    if vl_metadata:
                        result_data['metadata_vl'] = vl_metadata
                        if self._debug:
                            logging.debug(f"VL metadata extracted: {vl_metadata}")
                    else:
                        if self._debug:
                            logging.debug(f"No VL metadata extracted from {input_path}")
                except Exception as e:
                    # Don't fail the whole extraction if metadata extraction fails
                    logging.warning(f"VL metadata extraction failed for {input_path}: {e}")
                    if self._debug:
                        logging.debug(f"VL metadata extraction error details:", exc_info=True)

            # Handle direct metadata extraction for MLX-VLM
            from extractors import MLXVLMExtractor
            if isinstance(extractor, MLXVLMExtractor) and mlx_vlm_config.get('extract_metadata'):
                try:
                    if self._debug:
                        logging.debug(f"Attempting MLX-VLM metadata extraction for {input_path}")
                    vl_metadata = extractor.extract_metadata_direct(input_path)
                    if vl_metadata:
                        result_data['metadata_vl'] = vl_metadata
                        if self._debug:
                            logging.debug(f"MLX-VLM metadata extracted: {vl_metadata}")
                except Exception as e:
                    logging.warning(f"MLX-VLM metadata extraction failed for {input_path}: {e}")

        except Exception as e:
            result_data["error"] = f"ExtractionManager.extract failed: {str(e)}"
            logging.error(result_data["error"], exc_info=self._debug)
            result_data["success"] = False
        
        return result_data

    def clear_extractor_cache(self, extractor_type: Optional[str] = None):
        """
        Clear cached extractors to free memory or force reinitialization.
        
        Args:
            extractor_type: If specified, only clear this type (e.g., 'nanonets', 'docstrange')
                        If None, clear all cached extractors
        """
        if extractor_type:
            # Clear specific type
            keys_to_remove = [k for k in self._extractors_cache.keys() 
                            if extractor_type.lower() in k.lower()]
            for key in keys_to_remove:
                if self._debug:
                    logging.debug(f"Clearing cached extractor: {key}")
                del self._extractors_cache[key]
        else:
            # Clear all
            if self._debug:
                logging.debug(f"Clearing all {len(self._extractors_cache)} cached extractors")
            self._extractors_cache.clear()