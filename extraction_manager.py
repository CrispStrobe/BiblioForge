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

from tqdm import tqdm

# Imports from our new modules
from extractors import ( # Direct import of the 'extractors' package/module
    PDFExtractor, EPUBExtractor, DJVUExtractor, 
    MOBIExtractor, TextExtractor, HTMLExtractor, PPTXExtractor
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
                    docstrange_config: Optional[Dict] = None) -> Optional[Any]:
        """Get appropriate extractor for file with improved caching"""
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Priority 1: DocStrange (if enabled)
        if use_docstrange and docstrange_config:
            # FIXED: Better cache key that includes all relevant config
            mode = docstrange_config.get('mode', 'cloud')
            api_key_hash = hash(docstrange_config.get('api_key', 'none'))  # Hash for privacy
            cache_key = f"docstrange_{mode}_{api_key_hash}"
            
            if cache_key not in self._extractors_cache:
                try:
                    from extractors import DocStrangeExtractor
                    
                    if self._debug:
                        logging.debug(f"Initializing DocStrange extractor (mode: {mode})")
                    
                    self._extractors_cache[cache_key] = DocStrangeExtractor(
                        import_cache=self._import_cache,
                        debug=self._debug,
                        mode=mode,
                        api_key=docstrange_config.get('api_key'),
                        binary_paths=self._binary_paths
                    )
                    
                    if self._debug:
                        logging.debug(f"DocStrange initialized successfully with cache key: {cache_key}")
                        
                except Exception as e:
                    logging.error(f"Failed to initialize DocStrange: {e}", exc_info=self._debug)
                    self._extractors_cache[cache_key] = None
            
            if self._extractors_cache.get(cache_key):
                if self._debug:
                    logging.debug(f"Using cached DocStrange extractor: {cache_key}")
                return self._extractors_cache[cache_key]
        
        # Priority 2: Nanonets-OCR2 (if enabled)
        if use_nanonets_ocr2 and nanonets_config:
            # FIXED: Better cache key generation with all relevant config
            cache_key_parts = ["nanonets"]
            
            if nanonets_config.get('use_ollama'):
                ollama_model = nanonets_config.get('ollama_model', 'default')
                ollama_host = nanonets_config.get('ollama_host', 'localhost')
                cache_key_parts.extend(['ollama', ollama_model, ollama_host])
                
            elif nanonets_config.get('use_gguf'):
                quant = nanonets_config.get('gguf_quantization', 'q4')
                n_ctx = nanonets_config.get('n_ctx', 8192)  # Default to 8192
                n_gpu = nanonets_config.get('n_gpu_layers', -1)
                chat_format = nanonets_config.get('gguf_chat_format', 'llava-1.5')
                cache_key_parts.extend(['gguf', quant, str(n_ctx), str(n_gpu), chat_format])
                
            elif nanonets_config.get('use_vllm_server'):
                vllm_url = nanonets_config.get('vllm_base_url', 'localhost:8000')
                cache_key_parts.extend(['vllm', vllm_url])
                
            else:
                model_name = nanonets_config.get('model_name', '3b')
                device = nanonets_config.get('device_map', 'auto')
                cache_key_parts.extend(['transformers', model_name, device])
            
            cache_key = '_'.join(cache_key_parts)
            
            if cache_key not in self._extractors_cache:
                try:
                    from extractors import NanonetsOCR2Extractor
                    
                    if self._debug:
                        logging.debug(f"Initializing Nanonets-OCR2 extractor with config: {nanonets_config}")
                    
                    self._extractors_cache[cache_key] = NanonetsOCR2Extractor(
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
                        logging.debug(f"Nanonets-OCR2 initialized successfully with cache key: {cache_key}")
                        
                except Exception as e:
                    logging.error(f"Failed to initialize Nanonets-OCR2: {e}", exc_info=self._debug)
                    self._extractors_cache[cache_key] = None
            
            if self._extractors_cache.get(cache_key):
                if self._debug:
                    logging.debug(f"Using cached Nanonets-OCR2 extractor: {cache_key}")
                return self._extractors_cache[cache_key]
        
        # Priority 3: Standard extractors based on file extension
        if file_ext not in self._extractors_cache:
            extractor_type = self.SUPPORTED_EXTENSIONS.get(file_ext)
            if not extractor_type:
                logging.warning(f"Unsupported file type for extraction: {file_path} (ext: {file_ext})")
                return None
            
            extractor_class = None
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
            
            if extractor_class:
                try:
                    if self._debug:
                        logging.debug(f"Initializing {extractor_type} extractor for {file_ext}")
                    
                    self._extractors_cache[file_ext] = extractor_class(
                        import_cache=self._import_cache,
                        debug=self._debug,
                        binary_paths=self._binary_paths 
                    )
                    
                    if self._debug:
                        logging.debug(f"{extractor_type} extractor initialized successfully")
                        
                except Exception as e:
                    logging.error(f"Failed to initialize extractor '{extractor_type}' for '{file_ext}': {e}", exc_info=self._debug)
                    self._extractors_cache[file_ext] = None
            else:
                self._extractors_cache[file_ext] = None
        
        return self._extractors_cache.get(file_ext)

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

            # FIXED: Extract Nanonets and DocStrange configs from kwargs
            nanonets_config = kwargs.get('nanonets_config')
            docstrange_config = kwargs.get('docstrange_config')
            use_nanonets = nanonets_config is not None
            use_docstrange = docstrange_config is not None
            
            # Get extractor with proper config passing
            extractor = self._get_extractor(
                input_path,
                use_nanonets_ocr2=use_nanonets,
                nanonets_config=nanonets_config,
                use_docstrange=use_docstrange,
                docstrange_config=docstrange_config
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
            
            # FIXED: For Nanonets/DocStrange, pass additional params
            from extractors import NanonetsOCR2Extractor, DocStrangeExtractor
            if isinstance(extractor, NanonetsOCR2Extractor):
                extractor_kwargs.update({
                    'max_tokens': kwargs.get('max_tokens', 15000),
                    'prompt_template': kwargs.get('prompt_template')
                })
            elif isinstance(extractor, DocStrangeExtractor):
                extractor_kwargs.update({
                    'output_format': docstrange_config.get('output_format', 'markdown') if docstrange_config else 'markdown'
                })
            
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