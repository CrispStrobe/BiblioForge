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
    MOBIExtractor, TextExtractor, HTMLExtractor
)
from utils import (      # Direct import of 'utils' module
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


    def _get_extractor(self, file_path: str) -> Optional[Any]:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self._extractors_cache:
            extractor_type = self.SUPPORTED_EXTENSIONS.get(file_ext)
            if not extractor_type:
                logging.warning(f"Unsupported file type for extraction: {file_path} (ext: {file_ext})")
                return None
            
            extractor_class = None
            if extractor_type == 'PDF': extractor_class = PDFExtractor
            elif extractor_type == 'EPUB': extractor_class = EPUBExtractor
            elif extractor_type == 'DJVU': extractor_class = DJVUExtractor
            elif extractor_type == 'MOBI': extractor_class = MOBIExtractor
            elif extractor_type == 'Text': extractor_class = TextExtractor
            elif extractor_type == 'HTML': extractor_class = HTMLExtractor
            
            if extractor_class:
                try:
                    # Pass the central ImportCache and detected binary_paths
                    self._extractors_cache[file_ext] = extractor_class(
                        import_cache=self._import_cache, # Pass the central ImportCache
                        debug=self._debug,
                        binary_paths=self._binary_paths 
                    )
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
                # Added for passing to LLM functions if needed by sort logic here
                temperature: float = 0.5, 
                max_tokens: int = 250,
                **kwargs) -> Dict[str, Any]:
        
        result_data: Dict[str, Any] = { # Ensure type for result_data
            "success": False, "text": "", "output_path": output_path, "skipped": False, 
            "error": None, "tables": [], "metadata_llm": None, "renamed_info": None
        }
        
        try:
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext not in self.SUPPORTED_EXTENSIONS:
                result_data["error"] = f"Unsupported file type: {input_path}"
                return result_data

            extractor = self._get_extractor(input_path)
            if not extractor:
                result_data["error"] = f"No suitable extractor found for {input_path}"
                return result_data

            if hasattr(extractor, 'set_password') and password:
                extractor.set_password(password)
            
            # Simplified progress callback for internal extractor use
            def _internal_progress_cb(n=1, engine_info=None):
                if self._debug and engine_info:
                    logging.debug(f"Extractor ({engine_info}) progress step: {n}")

            extractor_kwargs = {'preferred_method': method}
            if isinstance(extractor, PDFExtractor): # Check specific type for PDF args
                extractor_kwargs.update({
                    'ocr_method': ocr_method,
                    'force_ocr': force_ocr,
                    'extract_tables': extract_tables
                })
            
            extracted_text = extractor.extract_text(
                input_path, 
                progress_callback=_internal_progress_cb,
                **extractor_kwargs
            )

            if extracted_text and extracted_text.strip():
                result_data["text"] = extracted_text
                result_data["success"] = True
                if self._debug: logging.debug(f"Successfully extracted {len(extracted_text)} chars from {input_path}")
                
                if isinstance(extractor, PDFExtractor) and extract_tables:
                    tables_df_list = extractor.get_last_extracted_tables()
                    if tables_df_list:
                        result_data["tables"] = [df.to_dict('records') for df in tables_df_list]
                        if self._debug: logging.info(f"Stored {len(tables_df_list)} tables from {input_path}.")

                if output_path: # If an output path for the .txt file is provided
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
                result_data["success"] = False # Ensure success is false if no text

            # LLM-based sorting (moved most of this to DocumentProcessor)
            # This manager's extract method focuses on text/table extraction.
            # Sorting and renaming logic is better handled at a higher level (DocumentProcessor)
            # after extraction is complete.
            # However, if this `extract` method is called per file with sort=True, it might need to do it.
            # Let's assume for now that the sorting logic in _process_single_file of DocumentProcessor
            # calls this extract method and then handles sorting.
            # So, this `extract` method doesn't need the full sorting logic itself.
            # It just needs to return the text and success.

        except Exception as e:
            result_data["error"] = f"ExtractionManager.extract for {input_path} failed: {str(e)}"
            logging.error(result_data["error"], exc_info=self._debug)
            result_data["success"] = False
        
        return result_data