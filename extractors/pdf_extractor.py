# extractors/pdf_extractor.py

import os
import logging
import shutil
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from tqdm import tqdm
import threading
import time
import shlex
import traceback 

# Direct import for utilities, assuming 'utils.py' is in PYTHONPATH
# (which it will be if running from the root directory)
try:
    from utils import ImportCache, run_process, shutdown_flag # Added shutdown_flag
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py in pdf_extractor.py: {e}")
    # Depending on how critical these are, you might re-raise or provide dummy objects
    # For now, let's assume they are critical.
    raise

# Imports for specific PDF libraries, will be handled by ImportCache or try-except
# import fitz (pymupdf)
# import pdfplumber
# import pypdf
# from pdfminer import high_level
# import pytesseract
# import pdf2image
# import easyocr
# from paddleocr import PaddleOCR
# import doctr
# import kraken
# from PIL import Image
# import cv2
# import numpy as np
# import camelot

# import for utilities from the parent directory's utils.py
try:
    from utils import ImportCache, run_process #, shutdown_flag (if directly used)
except ImportError:
    # Fallback for direct execution or different project structure
    # This might happen if pdf_extractor.py is run standalone for testing (not recommended for this structure)
    logging.warning("Could not perform relative import for utils. Assuming utils.py is in PYTHONPATH.")

class TableExtractor:
    """PDF table extraction using Camelot"""
    
    def __init__(self, import_cache: ImportCache):
        self._import_cache = import_cache
        self._camelot = None
        
    def extract_tables(self, pdf_path: str, password: Optional[str] = None) -> List[Any]:
        """Extract tables using multiple methods"""
        if not self._init_camelot():
            logging.warning("Camelot not initialized, cannot extract tables.")
            return []
            
        tables = []
        # Camelot expects 'pages' as a string, e.g., '1', '1,3-5', 'all'
        # For password protected PDFs, camelot needs the password.
        # Common flavors: 'lattice' (for tables with clear lines) and 'stream' (for tables without lines)
        methods_params = [
            ('lattice', {'password': password, 'line_scale': 40}),
            ('stream', {'password': password, 'edge_tol': 500}) 
        ]
        
        logging.info(f"Attempting to extract tables from {pdf_path} using Camelot.")
        for method, params in methods_params:
            try:
                logging.debug(f"Camelot: Trying method '{method}' with params {params}")
                # Ensure camelot is available
                if not self._import_cache.is_available('camelot'):
                    logging.warning("Camelot library not available for table extraction.")
                    return []

                current_tables_report = self._camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor=method,
                    **params
                )
                logging.info(f"Camelot method '{method}' found {current_tables_report.n} tables.")
                
                # Add successfully parsed tables (as DataFrames) to the list
                for table in current_tables_report:
                    tables.append(table.df) # Store the pandas DataFrame
                    
            except Exception as e:
                # Catch specific errors if possible, e.g., for encrypted PDFs without password
                if "pdfisencryptedexception" in str(e).lower() and not password:
                    logging.warning(f"Table extraction with Camelot ({method}) failed: PDF is encrypted and no password provided.")
                else:
                    logging.debug(f"Table extraction with Camelot ({method}) failed: {e}")
                continue # Try next method
                
        return tables
        
    def _init_camelot(self) -> bool:
        """Initialize Camelot library only when needed."""
        if self._camelot is None:
            if self._import_cache.is_available('camelot'):
                try:
                    self._camelot = self._import_cache.import_module('camelot')
                    # Check for Ghostscript, a Camelot dependency for 'lattice'
                    if not (self._import_cache.is_available('ghostscript') or shutil.which('gs') or shutil.which('gswin64c')):
                         logging.warning("Ghostscript not found. 'lattice' table extraction in Camelot might fail.")
                    return True
                except Exception as e:
                    logging.error(f"Failed to initialize Camelot: {e}")
                    return False
            else:
                logging.warning("Camelot-py package not found. Table extraction will be skipped.")
                return False
        return True

    
class PDFExtractor:
    """Enhanced PDF text extraction with lazy loading and multiple fallback methods"""

    TEXT_METHODS = [
        'pymupdf',      # Fast native PDF parsing
        'pdfplumber',   # Good balance of speed and accuracy
        'calibre',      # proven
        'pypdf',        # Simple but reliable
        'pdfminer',     # Good layout preservation
        'tesseract',    # OCR support
        'easyocr',      # Alternative OCR
        'paddleocr',    # multilingual
        'doctr',        # Deep learning OCR 
        'kraken_cli',   # Kraken CLI method
        'kraken',       # Kraken API method (lower priority)
        'crispembed'    # CrispEmbed single-pass VLM OCR (opt-in; needs a model)
    ]
    CORE_METHODS = ['pymupdf', 'calibre', 'pdfplumber', 'pypdf', 'pdfminer']
    OCR_METHODS = ['tesseract', 'easyocr', 'paddleocr', 'doctr', 'kraken', 'kraken_cli', 'crispembed']
    
    # TABLE_METHODS = ['camelot'] # TableExtractor handles this

    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths: Optional[Dict[str,str]] = None): # Added import_cache parameter
        self._debug = debug
        self._import_cache = import_cache # Use the passed ImportCache instance
        self._initialized_methods = set()
        self._password = None
        self._current_doc = None 
        self._ocr_initialized: Dict[str, bool] = {} 
        self._available_methods: Optional[Dict[str, bool]] = None
        self._ocr_failed_methods = set()

        # Optional pre-OCR scan cleanup (CrispEmbed). Configured via set_scan_cleanup().
        self._scan_cleanup_config = None
        self._scan_cleaner = None          # cached CrispScanCleanup instance
        self._scan_cleaner_failed = False  # True once we've given up loading it
        self._scan_cleanup_logged = False  # log activation only once

        # Optional pre-OCR super-resolution (CrispEmbed). Configured via set_super_resolution().
        self._sr_config = None
        self._super_resolver = None
        self._sr_failed = False
        self._sr_logged = False

        # Optional CrispEmbed single-pass OCR backend. Configured via set_crispembed_ocr().
        self._crispembed_ocr_config = None
        self._crispembed_ocr = None

        self._setup_windows_paths() 

        if binary_paths:
            self._binary_paths = binary_paths
            self._binaries = {
                'tesseract': bool(binary_paths.get('tesseract')),
                'poppler': bool(binary_paths.get('pdftoppm')), 
                'ghostscript': bool(binary_paths.get('gs')),
                'djvulibre': bool(binary_paths.get('djvutxt')) or bool(binary_paths.get('ddjvu')),
                'calibre': bool(binary_paths.get('ebook-converter'))
            }
            if self._debug: logging.debug("PDFExtractor: Using provided binary paths.")
        else:
            if self._debug: logging.warning("PDFExtractor: No binary paths provided, detecting binaries.")
            # _check_system_dependencies should return Dict[str, str] for paths, Dict[str, bool] for bools
            # And self._binary_paths should store the Dict[str, str] part
            actual_binary_paths, self._binaries = self._check_system_dependencies()
            self._binary_paths = actual_binary_paths # Store the paths correctly
        
        self._check_core_dependencies()
        if self._debug or (hasattr(self, '_binaries') and self._binaries.get('tesseract', False)):
            self._check_ocr_dependencies()
        
        if self._debug:
            available = sorted(list(self._initialized_methods))
            logging.debug(f"PDFExtractor: Available text methods after init: {', '.join(available)}")
        
        # Initialize TableExtractor using the (now passed-in) import_cache
        self._table_extractor = TableExtractor(self._import_cache)


    def _setup_windows_paths(self):
        if platform.system() == 'Windows':
            paths_to_check = [
                r'C:\Program Files\gs\gs10.04.0\bin', r'C:\Program Files\gs\gs10.02.0\bin',
                r'C:\Program Files (x86)\gs\gs10.04.0\bin', r'C:\Program Files (x86)\gs\gs10.02.0\bin',
                r'C:\Users\stc\Downloads\code\poppler-24.08.0\Library\bin', # User specific, remove or generalize
                r'C:\Program Files\poppler-24.02.0\Library\bin',
                r'C:\Program Files\Tesseract-OCR', r'C:\Program Files (x86)\Tesseract-OCR',
            ]
            current_path = os.environ.get('PATH', '')
            paths_added_count = 0
            for path_dir in paths_to_check:
                if os.path.isdir(path_dir) and path_dir not in current_path:
                    os.environ['PATH'] = path_dir + os.pathsep + current_path
                    current_path = os.environ['PATH'] # Update current_path for next check
                    paths_added_count +=1
            if paths_added_count > 0 and self._debug:
                logging.debug(f"PDFExtractor: Added {paths_added_count} paths to system PATH for Windows.")

    def _check_calibre_available(self) -> bool:
        if hasattr(self, '_binary_paths') and self._binary_paths and self._binary_paths.get('ebook-converter'):
            if os.path.exists(self._binary_paths['ebook-converter']):
                if self._debug: logging.debug(f"Calibre found via _binary_paths: {self._binary_paths['ebook-converter']}")
                return True
        
        for bin_name in ['ebook-converter', 'ebook-convert']:
            calibre_path = shutil.which(bin_name)
            if calibre_path:
                if self._debug: logging.debug(f"Found Calibre in PATH: {calibre_path}")
                if not (hasattr(self, '_binary_paths') and self._binary_paths.get('ebook-converter')):
                    if not hasattr(self, '_binary_paths') or self._binary_paths is None: self._binary_paths = {}
                    self._binary_paths['ebook-converter'] = calibre_path # Store if found via PATH
                return True
        if self._debug: logging.debug("Calibre (ebook-converter/ebook-convert) not found.")
        return False
    
    def _safe_import(self, module_name: str, attribute_name: Optional[str] = None) -> Any:
        try:
            module = self._import_cache.import_module(module_name)
            if attribute_name:
                return getattr(module, attribute_name)
            return module
        except ImportError:
            if self._debug: logging.debug(f"Cannot import {module_name}{'.' + attribute_name if attribute_name else ''}")
        except AttributeError:
             if self._debug: logging.debug(f"Attribute {attribute_name} not found in {module_name}")
        except Exception as e:
            if self._debug: logging.debug(f"Error importing {module_name}: {e}")
        return None

    @property
    def languages(self) -> List[str]: # Tesseract specific, might need adjustment for other OCRs
        # Default, can be expanded by specific OCR engines if they support language listing
        return ['eng'] 

    def get_binary_path(self, binary_name: str) -> Optional[str]:
        binary_map = {
            'tesseract': 'tesseract', 'pdftoppm': 'pdftoppm', 'poppler': 'pdftoppm',
            'gs': 'gs', 'ghostscript': 'gs',
            'djvutxt': 'djvutxt', 'ddjvu': 'ddjvu', 'djvulibre': 'djvutxt',
            'ebook-converter': 'ebook-converter', 'ebook-convert': 'ebook-converter', 'calibre': 'ebook-converter'
        }
        binary_key = binary_map.get(binary_name.lower(), binary_name.lower())
        binary_paths_dict = getattr(self, '_binary_paths', {})
        return binary_paths_dict.get(binary_key)

    def _check_system_dependencies(self) -> Tuple[Dict[str, str], Dict[str, bool]]:
        logging.debug("PDFExtractor: Inner _check_system_dependencies called.")
        binary_paths = {
            'tesseract': None, 'pdftoppm': None, 'gs': None,
            'djvutxt': None, 'ddjvu': None, 'ebook-converter': None
        }
        executable_names = {
            'tesseract': ['tesseract', 'tesseract.exe'], 'pdftoppm': ['pdftoppm', 'pdftoppm.exe'],
            'gs': ['gs', 'gswin64c', 'gswin64c.exe', 'gswin32c.exe'],
            'djvutxt': ['djvutxt', 'djvutxt.exe'], 'ddjvu': ['ddjvu', 'ddjvu.exe'],
            'ebook-converter': ['ebook-converter', 'ebook-convert', 'ebook-converter.exe', 'ebook-convert.exe']
        }
        # ... (rest of the logic from main script's _check_system_dependencies, simplified for brevity here)
        # This method should ideally be called by ExtractionManager and paths passed in.
        # For now, a simplified version for PDFExtractor if it has to run it.
        for binary_key_in_loop, names in executable_names.items():
            for name in names:
                path = shutil.which(name)
                if path:
                    binary_paths[binary_key_in_loop] = path
                    if self._debug: logging.debug(f"PDFExtractor: Found {binary_key_in_loop} in PATH: {path}")
                    break
        
        binaries_bool = {
            'tesseract': bool(binary_paths.get('tesseract')),
            'poppler': bool(binary_paths.get('pdftoppm')),
            'ghostscript': bool(binary_paths.get('gs')),
            'djvulibre': bool(binary_paths.get('djvutxt')) or bool(binary_paths.get('ddjvu')),
            'calibre': bool(binary_paths.get('ebook-converter'))
        }
        return binary_paths, binaries_bool # Corrected to return tuple

    def _check_core_dependencies(self):
        if self._check_calibre_available():
            self._initialized_methods.add('calibre')
            if self._debug: logging.debug("Calibre ebook-converter available for PDFExtractor")
        
        if self._safe_import('fitz'): self._initialized_methods.add('pymupdf')
        if self._safe_import('pdfplumber'): self._initialized_methods.add('pdfplumber')
        if self._safe_import('pypdf'): self._initialized_methods.add('pypdf')
        if self._safe_import('pdfminer.high_level'): self._initialized_methods.add('pdfminer')
        
        if self._debug:
            core_init = sorted(list(self._initialized_methods.intersection(self.CORE_METHODS)))
            logging.debug(f"PDFExtractor: Initialized core methods: {', '.join(core_init)}")

    def _check_ocr_dependencies(self):
        if self._debug: logging.debug("PDFExtractor: Checking OCR dependencies.")
        # Tesseract
        if self.get_binary_path('tesseract') and \
           self._safe_import('pytesseract') and \
           self._safe_import('pdf2image'):
            try:
                pytesseract_module = self._import_cache.import_module('pytesseract')
                version = pytesseract_module.get_tesseract_version()
                self._pytesseract = pytesseract_module # Store for use
                self._pdf2image = self._import_cache.import_module('pdf2image') # Store for use
                self._initialized_methods.add('tesseract')
                if self._debug: logging.debug(f"Tesseract ({version}) available for PDFExtractor.")
            except Exception as e:
                if self._debug: logging.debug(f"Tesseract verification failed: {e}")
        
        # Other OCRs are lazily initialized in _init_ocr to save resources
        # For now, just check if their main package is importable for self.available_methods
        if self._import_cache.is_available('paddleocr'): self._initialized_methods.add('paddleocr')
        if self._import_cache.is_available('doctr'): self._initialized_methods.add('doctr')
        if self._import_cache.is_available('easyocr'): self._initialized_methods.add('easyocr')
        if self._import_cache.is_available('kraken'): self._initialized_methods.add('kraken')
        if shutil.which('kraken'): self._initialized_methods.add('kraken_cli')


    def _is_method_available(self, method: str) -> bool:
        if method in self._initialized_methods: return True # Already checked and available

        if method == 'calibre':
            is_avail = self._check_calibre_available()
            if is_avail: self._initialized_methods.add('calibre')
            return is_avail
        
        # Core methods (Python libraries)
        core_map = {'pymupdf': 'fitz', 'pdfplumber': 'pdfplumber', 'pypdf': 'pypdf', 'pdfminer': 'pdfminer.high_level'}
        if method in core_map:
            is_avail = bool(self._safe_import(core_map[method]))
            if is_avail: self._initialized_methods.add(method)
            return is_avail

        # OCR methods (need _init_ocr to fully check/load models)
        if method in self.OCR_METHODS:
            # Basic check for now, _init_ocr will do the full initialization
            if method == 'crispembed':
                # Opt-in only: available when explicitly configured AND CrispEmbed loads.
                is_avail = False
                if self._crispembed_ocr_config:
                    try:
                        import crispembed_adapter as ca
                        is_avail = ca.is_available()
                    except Exception:
                        is_avail = False
                if is_avail: self._initialized_methods.add('crispembed')
                return is_avail
            if method == 'tesseract':
                is_avail = (self.get_binary_path('tesseract') is not None and
                            self._import_cache.is_available('pytesseract') and
                            self._import_cache.is_available('pdf2image'))
            elif method == 'kraken_cli':
                is_avail = shutil.which('kraken') is not None
            else: # easyocr, paddleocr, doctr, kraken (API)
                is_avail = self._import_cache.is_available(method)
            
            if is_avail: self._initialized_methods.add(method) # Mark as potentially available
            return is_avail
        return False

    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            self._available_methods = {
                m: self._is_method_available(m) for m in self.TEXT_METHODS
            }
            # Table methods are handled by TableExtractor
        return self._available_methods

    def set_password(self, password: str):
        self._password = password

    def _safe_library_call(self, func, *args, **kwargs):
        """Wrapper to protect logging around library calls"""
        # Store current logging state
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = root_logger.handlers.copy()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore logging state if it was changed
            current_level = root_logger.level
            current_handlers = root_logger.handlers
            
            if (current_level != original_level or 
                len(current_handlers) != len(original_handlers)):
                
                if self._debug:
                    logging.warning(f"Library call to {func.__name__} changed logging configuration. Restoring...")
                
                # Restore level
                root_logger.setLevel(original_level)
                
                # Restore handlers if they were removed
                if len(current_handlers) < len(original_handlers):
                    for handler in original_handlers:
                        if handler not in current_handlers:
                            root_logger.addHandler(handler)

    def extract_text(self, pdf_path: str, preferred_method: Optional[str] = None,
                     ocr_method: Optional[str] = None, force_ocr: bool = False,
                     progress_callback: Optional[Callable] = None,
                     extract_tables: bool = False, **kwargs) -> str:
        if not os.path.exists(pdf_path):
            # This check should ideally be even earlier, e.g., in DocumentProcessor,
            # but it's good to have it here too as a safeguard.
            logging.error(f"PDFExtractor: File not found: {pdf_path}")
            raise FileNotFoundError(f"File not found: {pdf_path}")

        if self._debug: logging.debug(f"PDFExtractor: Starting extraction for {pdf_path}")

        final_text = ""
        used_method = "None"
        
        # --- NEW: Trackers for core method failures ---
        core_method_total_attempts = 0
        core_method_fatal_structure_failures = 0
        # Keywords indicating a PDF is likely unrecoverable for image conversion / OCR
        fatal_error_keywords = [
            "empty file", "no /root object", "document stream is empty",
            "no pages found", "cannot open empty file", "is this really a pdf"
        ]

        # Determine effective CORE methods to try
        core_methods_to_try = []
        if not force_ocr: # If not forcing OCR, try core methods first
            if preferred_method and preferred_method in self.CORE_METHODS and self.available_methods.get(preferred_method):
                core_methods_to_try.append(preferred_method)
            core_methods_to_try.extend([
                m for m in self.CORE_METHODS 
                if m != preferred_method and self.available_methods.get(m)
            ])
        
        if self._debug and not force_ocr: 
            logging.debug(f"PDFExtractor: Core methods to try for '{os.path.basename(pdf_path)}': {core_methods_to_try}")
        elif self._debug and force_ocr:
            logging.debug(f"PDFExtractor: Force OCR enabled for '{os.path.basename(pdf_path)}'. Skipping direct core text extraction attempts first.")


        # Attempt text layer extraction (CORE METHODS), only if not force_ocr
        if not force_ocr and core_methods_to_try:
            for method_name in core_methods_to_try:
                if shutdown_flag.is_set():
                    logging.info(f"PDFExtractor: Shutdown signal during core method processing for {pdf_path}.")
                    break
                
                if not hasattr(self, f'extract_with_{method_name}'):
                    if self._debug: logging.debug(f"PDFExtractor: Method {method_name} not implemented.")
                    continue

                core_method_total_attempts += 1
                if self._debug: logging.debug(f"PDFExtractor: Trying core method {method_name} for {pdf_path}")
                if progress_callback: progress_callback(1, f"pdf_{method_name}")
                
                try:
                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    current_text = extraction_func(pdf_path, progress_callback=progress_callback)
                    if current_text and current_text.strip():
                        quality = self._assess_text_quality(current_text)
                        if self._debug: logging.debug(f"PDFExtractor: Method {method_name} extracted {len(current_text)} chars, quality: {quality:.2f} for {pdf_path}")
                        if quality > 0.5: # Threshold for "good enough" text
                            final_text = current_text.strip()
                            used_method = method_name
                            break # Successful extraction from core method
                        else:
                            if not final_text: # Keep first non-empty result if all are low quality
                                final_text = current_text.strip()
                                used_method = method_name
                except KeyboardInterrupt:
                    logging.info(f"PDFExtractor: Extraction with {method_name} interrupted by user for {pdf_path}.")
                    raise
                except Exception as e:
                    error_str_lower = str(e).lower()
                    logging.warning(f"PDFExtractor: Core method {method_name} failed for {pdf_path}: {e}")
                    if any(keyword in error_str_lower for keyword in fatal_error_keywords):
                        core_method_fatal_structure_failures += 1
                        if self._debug: logging.debug(f"PDFExtractor: Core method {method_name} encountered a fatal structure error for {pdf_path}.")
                    if self._debug: traceback.print_exc()
            
            if final_text: # If a core method succeeded with good quality text
                 if self._debug: logging.info(f"PDFExtractor: Core extraction successful with '{used_method}' for {pdf_path}.")

        # --- MODIFIED OCR DECISION LOGIC ---
        # Proceed to OCR if:
        # 1. force_ocr is true (user explicitly wants OCR)
        # OR
        # 2. No usable text was found from core methods (`not final_text`)
        #    AND we haven't encountered too many fatal structural errors from core methods.
        
        proceed_to_ocr = False
        if force_ocr:
            proceed_to_ocr = True
            if self._debug: logging.debug(f"PDFExtractor: Force OCR is enabled for {pdf_path}. Proceeding to OCR.")
        elif not final_text: # No good text from core methods
            # Heuristic: If most/all attempted core methods failed with fatal structural errors,
            # it's unlikely OCR (which relies on converting PDF to image) will work.
            # Threshold: if more than half of attempted core methods had fatal errors, or at least 2 such errors.
            min_fatal_failures_to_skip_ocr = 2
            if core_method_total_attempts > 0 and \
               core_method_fatal_structure_failures >= min_fatal_failures_to_skip_ocr and \
               core_method_fatal_structure_failures >= core_method_total_attempts / 2:
                if self._debug:
                    logging.warning(f"PDFExtractor: Skipping OCR for {pdf_path}. {core_method_fatal_structure_failures}/{core_method_total_attempts} "
                                    f"core methods failed with fatal structural errors. File likely too corrupt for OCR.")
            else:
                proceed_to_ocr = True
                if self._debug: logging.debug(f"PDFExtractor: No usable text from core methods for {pdf_path} (or few fatal errors). Proceeding to OCR.")
        
        if proceed_to_ocr:
            ocr_methods_to_try = []
            # Prioritize user's specific ocr_method if valid and available
            if ocr_method and ocr_method != 'auto' and self.available_methods.get(ocr_method) and ocr_method in self.OCR_METHODS:
                 ocr_methods_to_try.append(ocr_method)
            # Add other available OCR methods, ensuring no duplicates and preferred one is first
            ocr_methods_to_try.extend([
                m for m in self.OCR_METHODS 
                if m not in ocr_methods_to_try and self.available_methods.get(m)
            ])

            if self._debug: logging.debug(f"PDFExtractor: OCR methods to try for '{os.path.basename(pdf_path)}': {ocr_methods_to_try}")

            for method_name in ocr_methods_to_try:
                if shutdown_flag.is_set():
                    logging.info(f"PDFExtractor: Shutdown signal during OCR method processing for {pdf_path}.")
                    break
                
                if not hasattr(self, f'extract_with_{method_name}'):
                    if self._debug: logging.debug(f"PDFExtractor: OCR Method {method_name} not implemented.")
                    continue
                
                if not self._init_ocr(method_name): # Ensure OCR engine is ready
                    if self._debug: logging.debug(f"PDFExtractor: OCR Method {method_name} failed to initialize for {pdf_path}.")
                    continue

                if self._debug: logging.debug(f"PDFExtractor: Trying OCR method {method_name} for {pdf_path}")
                if progress_callback: progress_callback(1, f"pdf_ocr_{method_name}")
                
                try:
                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    current_text = extraction_func(pdf_path, progress_callback=progress_callback) # Pass callback
                    if current_text and current_text.strip():
                        # Assuming OCR text is valuable if found, could add quality check here too
                        final_text = current_text.strip()
                        used_method = f"{method_name} (OCR)"
                        if self._debug: logging.info(f"PDFExtractor: OCR successful with '{used_method}' for {pdf_path}.")
                        break # Successful OCR
                except KeyboardInterrupt:
                    logging.info(f"PDFExtractor: OCR Extraction with {method_name} interrupted by user for {pdf_path}.")
                    raise
                except Exception as e:
                    logging.warning(f"PDFExtractor: OCR method {method_name} failed for {pdf_path}: {e}")
                    if self._debug: traceback.print_exc()
                    self._ocr_failed_methods.add(method_name)
        
        if not final_text and self._debug:
             logging.warning(f"PDFExtractor: All attempted methods failed to extract text from {pdf_path}.")


        if self._debug: logging.info(f"PDFExtractor: Finished extraction for {pdf_path} using {used_method}. Length: {len(final_text)}")

        if extract_tables:
            if self._debug: logging.info(f"PDFExtractor: Attempting table extraction for {pdf_path}")
            try:
                tables_data = self._table_extractor.extract_tables(pdf_path, password=self._password)
                if tables_data:
                    if self._debug: logging.info(f"PDFExtractor: Extracted {len(tables_data)} tables from {pdf_path}.")
                    self._last_extracted_tables = tables_data
                else:
                    self._last_extracted_tables = [] # Ensure it's reset if no tables found
            except Exception as te:
                logging.error(f"Table extraction failed for {pdf_path} in PDFExtractor: {te}", exc_info=self._debug)
                self._last_extracted_tables = []

        self._cleanup()
        return final_text

    def get_last_extracted_tables(self) -> List[Any]:
        """Returns tables extracted in the last call to extract_text, if any."""
        return getattr(self, "_last_extracted_tables", [])

    # Placeholder for other _assess_text_quality, _might_need_ocr, etc.
    def _assess_text_quality(self, text: str) -> float:
        # Simplified version for now
        if not text or len(text.strip()) < 50: return 0.0
        # Count non-printable characters (excluding common whitespace)
        text_len = len(text)
        if text_len == 0: return 0.0
        
        non_printable = sum(1 for char in text if not char.isprintable() and char not in '\n\r\t ')
        printable_ratio = 1.0 - (non_printable / text_len)
        
        # Check for very short average word length (often a sign of bad OCR or encoding)
        words = text.split()
        if not words: return 0.0 * printable_ratio # No words, bad quality
        
        avg_word_length = sum(len(w) for w in words) / len(words)
        word_length_score = 0.0
        if avg_word_length < 2.5: # Very short words
            word_length_score = 0.1
        elif avg_word_length < 3.5:
            word_length_score = 0.5
        else:
            word_length_score = 1.0

        # Combine scores (heuristic)
        return (printable_ratio * 0.7) + (word_length_score * 0.3)

    def _might_need_ocr(self, pdf_path: str) -> bool:
        # Simplified: if less than 100 chars on first page via pymupdf, likely needs OCR
        if 'pymupdf' in self._initialized_methods:
            fitz = self._safe_import('fitz')
            if fitz:
                doc = None
                try:
                    doc = fitz.open(pdf_path)
                    if doc.needs_pass and self._password: doc.authenticate(self._password)
                    if not doc.needs_pass and len(doc) > 0:
                        first_page_text = doc[0].get_text("text")
                        return len(first_page_text.strip()) < 100
                except Exception as e:
                    if self._debug: logging.debug(f"Error in _might_need_ocr: {e}")
                finally:
                    if doc: doc.close()
        return True # Default to true if unsure

    def _cleanup(self):
        
        import gc
        gc.collect()
        self._clear_gpu_memory()


    def _clear_gpu_memory(self):
        try:
            torch_module = self._safe_import('torch')
            if torch_module and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception: pass # Ignore if torch or cuda not available/fails
        try:
            tf_module = self._safe_import('tensorflow')
            if tf_module and hasattr(tf_module.keras.backend, 'clear_session'):
                 tf_module.keras.backend.clear_session()
        except Exception: pass


    def _init_ocr(self, method: str) -> bool:
        # Simplified version: Actual initialization should load models etc.
        # This method now primarily checks if the dependencies are met
        # and sets up any specific instances if not already done.
        if method in self._ocr_initialized and self._ocr_initialized[method]:
            return True
        if method in self._ocr_failed_methods:
            return False

        success = False
        try:
            if method == 'tesseract':
                if self.get_binary_path('tesseract') and self._safe_import('pytesseract') and self._safe_import('pdf2image'):
                    self._pytesseract = self._import_cache.import_module('pytesseract')
                    self._pdf2image = self._import_cache.import_module('pdf2image')
                    # Further check: try get_tesseract_version
                    if hasattr(self._pytesseract, 'get_tesseract_version'):
                        self._pytesseract.get_tesseract_version() 
                        success = True
            elif method == 'paddleocr':
                PaddleOCR_class = self._safe_import('paddleocr', 'PaddleOCR')
                if PaddleOCR_class:
                    # Lazy load model if not already loaded
                    if not hasattr(self, '_paddleocr') or self._paddleocr is None:
                         # Ensure PyTorch is configured for security before loading models
                        self._configure_torch_security()
                        self._paddleocr = PaddleOCR_class(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4', show_log=self._debug, use_gpu=self._is_gpu_available_for_paddle())
                    success = True
            elif method == 'doctr':
                doctr_module = self._safe_import('doctr.models', 'ocr_predictor')
                if doctr_module:
                     if not hasattr(self, '_doctr_predictor') or self._doctr_predictor is None:
                        self._configure_torch_security()
                        # Default model, can be configured further in extract_with_doctr
                        self._doctr_predictor = doctr_module(pretrained=True)
                     self._doctr = self._import_cache.import_module('doctr') # For other doctr utilities
                     success = True
            elif method == 'easyocr':
                easyocr_module = self._safe_import('easyocr', 'Reader')
                if easyocr_module:
                    if not hasattr(self, '_easyocr_reader') or self._easyocr_reader is None:
                        self._configure_torch_security()
                        torch_module = self._safe_import('torch')
                        use_gpu = torch_module is not None and torch_module.cuda.is_available()
                        self._easyocr_reader = easyocr_module(['en'], gpu=use_gpu) # Default to English
                    self._easyocr = self._import_cache.import_module('easyocr') # For other easyocr utilities
                    success = True
            elif method == 'kraken':
                kraken_recognition = self._safe_import('kraken.rpred') # Or kraken.recognition
                kraken_binarization = self._safe_import('kraken.binarization')
                kraken_pageseg = self._safe_import('kraken.pageseg')
                kraken_models = self._safe_import('kraken.lib.models')

                if kraken_recognition and kraken_binarization and kraken_pageseg and kraken_models:
                    self._kraken_rpred = kraken_recognition 
                    self._kraken_binarization = kraken_binarization
                    self._kraken_pageseg = kraken_pageseg
                    self._kraken_models = kraken_models
                    # Try to load default model to confirm setup
                    # model = self._kraken_models.load_any(self._kraken_rpred.get_default_model())
                    success = True # Assume success if imports work, model loading is deferred
            elif method == 'kraken_cli':
                success = shutil.which('kraken') is not None
            elif method == 'crispembed':
                if self._crispembed_ocr is not None:
                    success = True
                else:
                    cfg = self._crispembed_ocr_config or {}
                    import crispembed_adapter as ca
                    if ca.is_available():
                        model_name = cfg.get('model', 'got-ocr2')
                        logging.info(f"Loading CrispEmbed OCR model '{model_name}' (first use may download)...")
                        self._crispembed_ocr = ca.get_ocr_model(model_name, force_cpu=cfg.get('force_cpu', False))
                        success = self._crispembed_ocr is not None

            if success:
                self._ocr_initialized[method] = True
                if self._debug: logging.debug(f"PDFExtractor: OCR method '{method}' initialized.")
                return True
            else:
                raise RuntimeError(f"{method} core components not available.")

        except Exception as e:
            if self._debug: logging.warning(f"PDFExtractor: Failed to initialize OCR method '{method}': {e}")
            self._ocr_failed_methods.add(method)
            self._ocr_initialized[method] = False
            return False
        
    def _is_gpu_available_for_paddle(self) -> bool:
        """Checks if GPU is available and configured for PaddlePaddle."""
        try:
            paddle = self._safe_import('paddle')
            if paddle and hasattr(paddle.device, 'is_compiled_with_cuda') and paddle.device.is_compiled_with_cuda():
                gpu_count = paddle.device.cuda.device_count()
                if gpu_count > 0:
                    if self._debug: logging.debug(f"PaddleOCR: CUDA available with {gpu_count} device(s).")
                    return True
        except Exception as e:
            if self._debug: logging.debug(f"PaddleOCR: GPU check error: {e}")
        if self._debug: logging.debug("PaddleOCR: GPU not available or not compiled with CUDA support.")
        return False

    def _configure_torch_security(self):
        torch_module = self._safe_import('torch')
        if torch_module:
            try:
                # These settings might affect determinism or performance but are generally safe.
                torch_module.backends.cudnn.benchmark = False # False for more determinism if needed
                torch_module.backends.cudnn.deterministic = True # If reproducibility is key
                if self._debug: logging.debug("PyTorch security/determinism settings applied.")
            except Exception as e:
                if self._debug: logging.debug(f"Error applying PyTorch settings: {e}")
    
    # --- Individual Extraction Methods ---

    def extract_with_pymupdf(self, pdf_path: str, progress_callback=None) -> str:
        fitz = self._safe_import('fitz')
        if not fitz: return ""
        text_parts = []
        doc = None
        try:
            doc = fitz.open(pdf_path)
            # REMOVED: self._current_doc = doc
            if doc.needs_pass:
                if not self._password or not doc.authenticate(self._password):
                    raise ValueError("PyMuPDF: Invalid PDF password or no password provided.")
            
            total_pages = len(doc)
            for page_num in range(total_pages):
                if shutdown_flag.is_set(): break
                page = doc.load_page(page_num)
                page_text = page.get_text("text", sort=True)
                if not page_text.strip():
                    page_text_dict = page.get_text("dict", sort=True)
                    page_text = self._process_text_dict(page_text_dict)

                if page_text.strip(): text_parts.append(page_text.strip())
                if progress_callback: progress_callback(1)
            return "\n\n".join(text_parts)
        except Exception as e:
            if self._debug: logging.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return ""
        finally:
            if doc: 
                try: doc.close()
                except: pass

    def _process_text_dict(self, text_dict: Dict) -> str:
        # (Copied from main script, ensure it's robust)
        text_parts = []
        try:
            for block in text_dict.get('blocks', []):
                if 'lines' in block:
                    for line in block.get('lines', []):
                        line_text = ' '.join(span.get('text', '') for span in line.get('spans', []))
                        if line_text.strip():
                            text_parts.append(line_text)
        except Exception as e:
            logging.debug(f"PyMuPDF _process_text_dict failed: {e}")
        return '\n'.join(text_parts)

    def extract_with_pdfplumber(self, pdf_path: str, progress_callback=None) -> str:
        pdfplumber_module = self._safe_import('pdfplumber')
        if not pdfplumber_module: return ""
        text_parts = []
        
        try:
            if self._debug: logging.debug(f"PdfPlumber: >>> BEFORE pdfplumber.open for {pdf_path}")
            with pdfplumber_module.open(pdf_path, password=self._password) as pdf:
                # REMOVED: self._current_doc = pdf
                if self._debug: logging.debug(f"PdfPlumber: <<< AFTER pdfplumber.open, processing pages for {pdf_path}")
                
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    if shutdown_flag.is_set(): break
                    if self._debug: logging.debug(f"PdfPlumber: >>> BEFORE page.extract_text for page {i+1} of {pdf_path}")
                    
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=True, keep_blank_chars=False)
                    
                    if self._debug: logging.debug(f"PdfPlumber: <<< AFTER page.extract_text (layout=True) for page {i+1} of {pdf_path}")
                    if not page_text or not page_text.strip():
                        if self._debug: logging.debug(f"PdfPlumber: >>> BEFORE page.extract_text (fallback) for page {i+1} of {pdf_path}")
                        page_text = page.extract_text(keep_blank_chars=False)
                        if self._debug: logging.debug(f"PdfPlumber: <<< AFTER page.extract_text (fallback) for page {i+1} of {pdf_path}")

                    if page_text and page_text.strip(): text_parts.append(page_text.strip())
                    if progress_callback: progress_callback(1)
            
            if self._debug: logging.debug(f"PdfPlumber: Finished processing pages for {pdf_path}")
            return "\n\n".join(text_parts)
        except Exception as e:
            if self._debug: logging.error(f"pdfplumber extraction failed for {pdf_path}: {e}", exc_info=True)
            return ""
        finally:
            if self._debug: logging.debug(f"PdfPlumber: Cleanup complete for {pdf_path}")

    def extract_with_pypdf(self, pdf_path: str, progress_callback=None) -> str:
        pypdf_module = self._safe_import('pypdf', 'PdfReader')
        if not pypdf_module: return ""
        text_parts = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf_module(f)
                if reader.is_encrypted:
                    if not self._password or not reader.decrypt(self._password): # Pass empty string if no password
                        raise ValueError("PyPDF: PDF encrypted, password needed or incorrect.")
                
                total_pages = len(reader.pages)
                for i, page in enumerate(reader.pages):
                    if shutdown_flag.is_set(): break
                    page_text = page.extract_text()
                    if page_text and page_text.strip(): text_parts.append(page_text.strip())
                    if progress_callback: progress_callback(1)
            return "\n\n".join(text_parts)
        except Exception as e:
            if self._debug: logging.error(f"pypdf extraction failed for {pdf_path}: {e}")
            return ""

    def extract_with_pdfminer(self, pdf_path: str, progress_callback=None) -> str:
        extract_text_to_fp = self._safe_import('pdfminer.high_level', 'extract_text_to_fp')
        LAParams_class = self._safe_import('pdfminer.layout', 'LAParams')
        StringIO_class = self._safe_import('io', 'StringIO')

        if not all([extract_text_to_fp, LAParams_class, StringIO_class]): return ""
        
        output_string = StringIO_class()
        try:
            with open(pdf_path, 'rb') as fp:
                laparams = LAParams_class() # Use default LAParams or make configurable
                extract_text_to_fp(fp, output_string, laparams=laparams, 
                                   password=self._password or "", codec='utf-8')
            text = output_string.getvalue()
            if progress_callback: progress_callback(1) # Simplified progress for pdfminer
            return text.strip()
        except Exception as e:
            if self._debug: logging.error(f"pdfminer.six extraction failed for {pdf_path}: {e}")
            return ""
        finally:
            output_string.close()

    def extract_with_calibre(self, pdf_path: str, progress_callback=None) -> str:
        calibre_bin = self.get_binary_path('ebook-converter')
        if not calibre_bin:
            if self._debug: logging.warning("Calibre (ebook-converter) not found for PDF extraction.")
            return ""

        temp_output_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                temp_output_file = tmp.name
            
            cmd = [calibre_bin, pdf_path, temp_output_file]
            # Calibre might have PDF specific options, e.g. --pdf-no-images
            # Add options like --input-encoding=utf-8 --output-encoding=utf-8 if needed
            if self._debug: logging.debug(f"Running Calibre for PDF: {' '.join(cmd)}")
            
            result = run_process(cmd, timeout=120) # Use run_process from utils

            if result.returncode != 0:
                logging.warning(f"Calibre PDF conversion failed (code {result.returncode}). Stderr: {result.stderr.strip()[:200]}")
                return ""
            
            if os.path.exists(temp_output_file):
                with open(temp_output_file, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read().strip()
                if self._debug: logging.info(f"Calibre extracted {len(text)} chars from PDF.")
                if progress_callback: progress_callback(1)
                return text
            else:
                logging.warning(f"Calibre output file not created: {temp_output_file}")
                return ""
        except Exception as e:
            logging.error(f"Exception during Calibre PDF extraction for {pdf_path}: {e}")
            return ""
        finally:
            if temp_output_file and os.path.exists(temp_output_file):
                try: os.unlink(temp_output_file)
                except Exception as e_unlink: logging.debug(f"Error unlinking temp Calibre output {temp_output_file}: {e_unlink}")
    
    # --- OCR Methods (Tesseract, EasyOCR, PaddleOCR, DocTR, Kraken) ---
    # These will be similar to their existing implementations but using self._safe_import, etc.
    # For brevity, I'll show Tesseract and a placeholder for others.

    # --- Optional pre-OCR scan cleanup (CrispEmbed) ---

    def set_scan_cleanup(self, config: Optional[Dict]) -> None:
        """Configure pre-OCR scan cleanup. `config` is None (disabled) or a dict
        with 'mode' ('off'/'auto'/'on') and 'params' for CrispScanCleanup.process."""
        if config and config.get('mode', 'off') != 'off':
            self._scan_cleanup_config = config
        else:
            self._scan_cleanup_config = None

    def _get_scan_cleaner(self):
        """Lazily create and cache the CrispEmbed scan-cleanup engine.
        Returns None (once, quietly) if CrispEmbed isn't available."""
        if self._scan_cleaner is not None:
            return self._scan_cleaner
        if self._scan_cleaner_failed:
            return None
        try:
            import crispembed_adapter as ca
            if not ca.is_available():
                if self._debug:
                    logging.debug(f"Scan cleanup unavailable: {ca.unavailable_reason()}")
                self._scan_cleaner_failed = True
                return None
            self._scan_cleaner = ca.get_scan_cleanup()
            return self._scan_cleaner
        except Exception as e:
            if self._debug:
                logging.debug(f"Scan cleanup init failed: {e}")
            self._scan_cleaner_failed = True
            return None

    def _apply_scan_cleanup(self, pil_image):
        """Return a cleaned copy of `pil_image` when scan cleanup is enabled and
        CrispEmbed is available; otherwise return the original image unchanged.
        Never raises — OCR must proceed even if cleanup fails."""
        cfg = self._scan_cleanup_config
        if not cfg:
            return pil_image
        cleaner = self._get_scan_cleaner()
        if cleaner is None:
            return pil_image
        try:
            from PIL import Image
            out = cleaner.process(pil_image, **cfg.get('params', {}))  # numpy HxWx3 uint8
            cleaned = Image.fromarray(out)
            if not self._scan_cleanup_logged:
                logging.info(f"Pre-OCR scan cleanup active (mode={cfg.get('mode')}).")
                self._scan_cleanup_logged = True
            return cleaned
        except Exception as e:
            if self._debug:
                logging.debug(f"Scan cleanup skipped for a page (error): {e}")
            return pil_image

    def set_super_resolution(self, config: Optional[Dict]) -> None:
        """Configure optional pre-OCR super-resolution. `config` is None (disabled)
        or a dict with 'mode' ('off'/'auto'/'on'), 'engine', 'min_width' (auto
        trigger), 'max_input_width' (safety cap to avoid upscaling huge pages)."""
        if config and config.get('mode', 'off') != 'off':
            self._sr_config = config
        else:
            self._sr_config = None

    def _get_super_resolver(self):
        if self._super_resolver is not None:
            return self._super_resolver
        if self._sr_failed:
            return None
        try:
            import crispembed_adapter as ca
            if not ca.is_available():
                self._sr_failed = True
                return None
            engine = (self._sr_config or {}).get('engine', 'swinir')
            self._super_resolver = ca.get_super_resolver(engine)
            return self._super_resolver
        except Exception as e:
            if self._debug:
                logging.debug(f"Super-resolution init failed: {e}")
            self._sr_failed = True
            return None

    def _apply_super_resolution(self, pil_image):
        """Upscale low-resolution pages before OCR when enabled. Returns the
        original image when disabled, not triggered, or on any error."""
        cfg = self._sr_config
        if not cfg:
            return pil_image
        w = pil_image.size[0]
        mode = cfg.get('mode', 'off')
        min_width = cfg.get('min_width', 1500)
        max_input_width = cfg.get('max_input_width', 2500)
        # 'on' upscales any page below the safety cap; 'auto' only small pages.
        if mode == 'auto' and w >= min_width:
            return pil_image
        if w > max_input_width:
            # Page already large; upscaling would be slow/huge and pointless.
            return pil_image
        sr = self._get_super_resolver()
        if sr is None:
            return pil_image
        try:
            out = sr.upscale(pil_image)
            if not self._sr_logged:
                logging.info(f"Pre-OCR super-resolution active (engine={cfg.get('engine','swinir')}).")
                self._sr_logged = True
            return out
        except Exception as e:
            if self._debug:
                logging.debug(f"Super-resolution skipped for a page (error): {e}")
            return pil_image

    def _preprocess_ocr_image(self, pil_image):
        """Optional pre-OCR pipeline: super-resolution (low-res pages) then scan
        cleanup (deskew/crop/whiten). Both are no-ops unless configured + available."""
        pil_image = self._apply_super_resolution(pil_image)
        pil_image = self._apply_scan_cleanup(pil_image)
        return pil_image

    def set_crispembed_ocr(self, config: Optional[Dict]) -> None:
        """Configure the CrispEmbed OCR backend. `config` is None (disabled) or a
        dict with 'model' (registry name, default 'got-ocr2'), 'dpi' (render DPI,
        default 150), 'force_cpu' (avoid Metal unsupported-op aborts)."""
        self._crispembed_ocr_config = config or None
        # Force re-evaluation of method availability now that config changed.
        self._available_methods = None

    def extract_with_crispembed(self, pdf_path: str, progress_callback=None) -> str:
        """OCR each page with a single-pass CrispEmbed model (e.g. got-ocr2).
        Reuses the optional pre-OCR preprocessing pipeline."""
        if not self._init_ocr('crispembed') or self._crispembed_ocr is None:
            return ""
        pdf2image_module = self._safe_import('pdf2image')
        if not pdf2image_module:
            return ""
        cfg = self._crispembed_ocr_config or {}
        dpi = int(cfg.get('dpi', 150))
        text_parts = []
        images = None
        poppler_path_dir = self._get_poppler_path()
        try:
            conversion_args = {'dpi': dpi, 'thread_count': 1, 'userpw': self._password,
                               'poppler_path': poppler_path_dir}
            conversion_args = {k: v for k, v in conversion_args.items() if v is not None}
            images = pdf2image_module.convert_from_path(pdf_path, timeout=120, **conversion_args)
            for i, image in enumerate(images):
                if shutdown_flag.is_set(): break
                try:
                    image = self._preprocess_ocr_image(image)
                    page_text = self._crispembed_ocr.recognize(image)
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                except Exception as page_e:
                    if self._debug: logging.debug(f"CrispEmbed OCR page {i+1} failed: {page_e}")
                finally:
                    try: image.close()
                    except Exception: pass
                if progress_callback: progress_callback(1)
            return "\n\n".join(text_parts)
        except Exception as e:
            if self._debug: logging.error(f"CrispEmbed OCR extraction for {pdf_path} failed: {e}")
            self._ocr_failed_methods.add('crispembed')
            return ""
        finally:
            if images:
                for img in images:
                    try: img.close()
                    except Exception: pass

    def extract_with_tesseract(self, pdf_path: str, progress_callback=None) -> str:
        if not self._init_ocr('tesseract'): return "" # Ensures self._pytesseract and self._pdf2image are set
        
        pytesseract = self._pytesseract
        pdf2image = self._pdf2image
        text_parts = []
        images = None
        poppler_path_dir = self._get_poppler_path()
        conversion_timeout = 60 # Timeout in seconds for pdf2image conversion

        try:
            conversion_args = {'dpi': 300, 'thread_count': 1, 'grayscale': True, 
                               'userpw': self._password, 'poppler_path': poppler_path_dir}
            # Remove None values from args
            conversion_args = {k: v for k, v in conversion_args.items() if v is not None}

            images = pdf2image.convert_from_path(pdf_path, timeout=conversion_timeout, **conversion_args)
            
            for i, image in enumerate(images):
                if shutdown_flag.is_set(): break
                try:
                    # Optional pre-OCR scan cleanup (deskew/crop/whiten) via CrispEmbed.
                    image = self._preprocess_ocr_image(image)
                    text = pytesseract.image_to_string(image, lang='eng') # Add more langs if needed
                    if text.strip(): text_parts.append(text.strip())
                except Exception as page_e:
                    if self._debug: logging.debug(f"Tesseract OCR failed on page {i+1} of {pdf_path}: {page_e}")
                finally:
                    image.close() # Close image after processing
                if progress_callback: progress_callback(1)
            return "\n\n".join(text_parts)
        except pdf2image.exceptions.PDFInfoNotInstalledError:
            logging.error("Poppler not installed or not in PATH. Tesseract OCR via pdf2image needs it.")
            self._ocr_failed_methods.add('tesseract') # Mark as failed due to Poppler
            return ""
        except Exception as e:
            if self._debug: logging.error(f"Tesseract OCR extraction for {pdf_path} failed: {e}")
            self._ocr_failed_methods.add('tesseract') # General failure
            return ""
        finally:
            if images:
                for img in images:
                    try: img.close()
                    except: pass # Image might already be closed

    def _get_poppler_path(self) -> Optional[str]:
        # This helper should ideally be in utils or handled by pdf2image itself if possible
        # For now, assume it's specific to how this class uses pdf2image
        pdftoppm_path = self.get_binary_path('pdftoppm')
        if pdftoppm_path:
            return os.path.dirname(pdftoppm_path)
        # Check common Poppler paths if not found via get_binary_path (which relies on _binary_paths)
        # This part is more for robust fallback if _binary_paths wasn't comprehensive for Poppler.
        if platform.system() == "Windows":
            # Example common paths for Poppler's bin directory on Windows
            common_poppler_paths = [
                r'C:\Program Files\poppler-24.02.0\Library\bin', # Adjust version as needed
                r'C:\poppler\bin' 
            ]
            for p_path in common_poppler_paths:
                if os.path.isdir(p_path) and os.path.exists(os.path.join(p_path, "pdftoppm.exe")):
                    return p_path
        return None # Let pdf2image try to find it if None

    # wrapper to make it safer (ensure logging does not break e.g.)
    def extract_with_easyocr(self, pdf_path: str, progress_callback=None) -> str:
        return self._safe_library_call(self._extract_with_easyocr_wrapped, pdf_path, progress_callback)

    def _extract_with_easyocr_wrapped(self, pdf_path: str, progress_callback=None) -> str:
        if not self._init_ocr('easyocr'): return ""
        # Implementation similar to main script, using self._easyocr_reader
        # ... (ensure pdf2image, numpy are imported via self._safe_import)
        pdf2image_module = self._safe_import('pdf2image')
        np_module = self._safe_import('numpy')
        if not pdf2image_module or not np_module or not hasattr(self, '_easyocr_reader'):
            return ""
        
        text_parts = []
        images = None
        poppler_path_dir = self._get_poppler_path()
        conversion_timeout = 60 # Timeout in seconds for pdf2image conversion

        try:
            conversion_args = {'dpi': 300, 'thread_count': 1, 'userpw': self._password, 'poppler_path': poppler_path_dir}
            conversion_args = {k: v for k, v in conversion_args.items() if v is not None}
            images = pdf2image_module.convert_from_path(
                pdf_path,
                timeout=conversion_timeout, # <<< MODIFIED: Added timeout
                **conversion_args
            )

            for image in images:
                if shutdown_flag.is_set(): break
                try:
                    image = self._preprocess_ocr_image(image)
                    img_array = np_module.array(image)
                    results = self._easyocr_reader.readtext(img_array, detail=0, paragraph=True)
                    if results: text_parts.append("\n".join(results))
                except Exception as e_page:
                     if self._debug: logging.debug(f"EasyOCR page failed: {e_page}")
                finally:
                    image.close()
                if progress_callback: progress_callback(1)
            return "\n\n".join(text_parts)
        except Exception as e:
            if self._debug: logging.error(f"EasyOCR extraction for {pdf_path} failed: {e}")
            self._ocr_failed_methods.add('easyocr')
            return ""
        finally:
            if images: 
                for img in images:
                    try: img.close()
                    except: pass
            self._clear_gpu_memory() # Clear GPU after EasyOCR

    def extract_with_paddleocr(self, pdf_path: str, progress_callback=None) -> str:
        if not self._init_ocr('paddleocr') or not hasattr(self, '_paddleocr'):
             logging.warning("PaddleOCR not properly initialized.")
             return ""
        pdf2image_module = self._safe_import('pdf2image')
        np_module = self._safe_import('numpy')
        Image_module = self._safe_import('PIL.Image')
        io_module = self._safe_import('io')

        if not all([pdf2image_module, np_module, Image_module, io_module]): return ""

        text_parts = []
        images = None
        poppler_path_dir = self._get_poppler_path()
        conversion_timeout = 60 # Timeout in seconds

        try:
            conversion_args = {'dpi': 200, 'thread_count': 1, 'fmt': 'jpeg', 'userpw': self._password, 'poppler_path': poppler_path_dir}
            conversion_args = {k: v for k, v in conversion_args.items() if v is not None}
            images = pdf2image_module.convert_from_path(pdf_path, timeout=conversion_timeout, **conversion_args)

            for i, image in enumerate(images):
                if shutdown_flag.is_set(): break
                try:
                    image = self._preprocess_ocr_image(image)
                    img_buffer = io_module.BytesIO()
                    image.save(img_buffer, format="JPEG")
                    img_buffer.seek(0)
                    pil_img = Image_module.open(img_buffer)
                    
                    result = self._paddleocr.ocr(np_module.array(pil_img), cls=True) # Try np array
                    
                    # PaddleOCR result structure: [[box, (text, confidence)], ...] for each line
                    if result and result[0]: # result is a list of lists of results per image/page
                        page_result = result[0] # Assuming single image was passed
                        for line_info in page_result:
                            if isinstance(line_info, list) and len(line_info) == 2: # Should be [box, (text, conf)]
                                text_tuple = line_info[1]
                                if isinstance(text_tuple, tuple) and len(text_tuple) == 2:
                                    text, confidence = text_tuple
                                    if confidence > 0.5: # Confidence threshold
                                        text_parts.append(text)
                except Exception as page_e:
                    if self._debug: logging.debug(f"PaddleOCR page {i+1} failed: {page_e}")
                finally:
                    image.close()
                if progress_callback: progress_callback(1)
            return "\n".join(text_parts) # Usually PaddleOCR provides line-by-line text
        except Exception as e:
            if self._debug: logging.error(f"PaddleOCR extraction for {pdf_path} failed: {e}")
            self._ocr_failed_methods.add('paddleocr')
            return ""
        finally:
            if images: 
                for img in images:
                    try: img.close()
                    except: pass
            self._clear_gpu_memory()

    def extract_with_doctr(self, pdf_path: str, progress_callback=None) -> str:
        if not self._init_ocr('doctr') or not hasattr(self, '_doctr_predictor'):
            logging.warning("DocTR not properly initialized.")
            return ""
        pdf2image_module = self._safe_import('pdf2image')
        np_module = self._safe_import('numpy')
        if not all([pdf2image_module, np_module]): return ""

        text_parts = []
        images = None
        poppler_path_dir = self._get_poppler_path()
        conversion_timeout = 60 # Timeout in seconds

        try:
            conversion_args = {'dpi': 300, 'thread_count': 1, 'userpw': self._password, 'poppler_path': poppler_path_dir}
            conversion_args = {k: v for k, v in conversion_args.items() if v is not None}
            images = pdf2image_module.convert_from_path(pdf_path, timeout=conversion_timeout, **conversion_args)

            for i, image in enumerate(images):
                if shutdown_flag.is_set(): break
                try:
                    image = self._preprocess_ocr_image(image)
                    img_array = np_module.array(image)
                    result_doc = self._doctr_predictor([img_array]) # Pass as a list
                    page_text = result_doc.render() # render() gives a single string
                    if page_text.strip(): text_parts.append(page_text.strip())
                except Exception as page_e:
                    if self._debug: logging.debug(f"DocTR page {i+1} failed: {page_e}")
                finally:
                    image.close()
                if progress_callback: progress_callback(1)
            return "\n\n".join(text_parts)
        except Exception as e:
            if self._debug: logging.error(f"DocTR extraction for {pdf_path} failed: {e}")
            self._ocr_failed_methods.add('doctr')
            return ""
        finally:
            if images: 
                for img in images:
                    try: img.close()
                    except: pass
            self._clear_gpu_memory()
            
    def extract_with_kraken_cli(self, pdf_path: str, progress_callback=None) -> str:
        if not self._init_ocr('kraken_cli'): return ""
        pdf2image_module = self._safe_import('pdf2image')
        if not pdf2image_module: return ""

        kraken_bin = shutil.which('kraken') # Already confirmed by _init_ocr
        text_parts = []
        images = None
        poppler_path_dir = self._get_poppler_path()
        conversion_timeout = 60 
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                conversion_args = {'dpi': 300, 'thread_count': 1, 'grayscale': True, 
                                   'output_folder':temp_dir, 'fmt':'png', 'userpw': self._password,
                                   'poppler_path': poppler_path_dir}
                conversion_args = {k: v for k, v in conversion_args.items() if v is not None}
                # convert_from_path returns list of PIL images, but also saves them if output_folder is given
                pil_images = pdf2image_module.convert_from_path(pdf_path, timeout=conversion_timeout, **conversion_args)
                
                image_paths = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.png')])
                if not image_paths: return ""

                for i, img_path in enumerate(image_paths):
                    if shutdown_flag.is_set(): break
                    txt_output = os.path.join(temp_dir, f"page_{i}.txt")
                    ocr_cmd = [kraken_bin, "-i", img_path, txt_output, "segment", "-bl", "ocr"]
                    # Add model if found/downloaded (omitted for brevity, see full script)
                    # ...
                    result = run_process(ocr_cmd, timeout=120)
                    if result.returncode == 0 and os.path.exists(txt_output):
                        with open(txt_output, 'r', encoding='utf-8') as f:
                            page_text = f.read().strip()
                        if page_text: text_parts.append(page_text)
                    else:
                        if self._debug: logging.debug(f"Kraken CLI failed for {img_path}: {result.stderr}")
                    if progress_callback: progress_callback(1)
                
                for pil_img in pil_images: # Close PIL images
                    try: pil_img.close()
                    except: pass
                return "\n\n".join(text_parts)

            except Exception as e:
                if self._debug: logging.error(f"Kraken CLI extraction for {pdf_path} failed: {e}")
                self._ocr_failed_methods.add('kraken_cli')
                return ""
            finally:
                if images: # This was pil_images
                    for img in images: # if images was assigned pil_images
                        try: img.close()
                        except: pass

    def extract_with_kraken(self, pdf_path: str, progress_callback=None) -> str:
        if not self._init_ocr('kraken'):
            logging.warning("Kraken (API) initialization failed, skipping.")
            return ""
        
        pdf2image_module = self._safe_import('pdf2image')
        Image_module = self._safe_import('PIL.Image') # kraken needs PIL.Image
        
        if not pdf2image_module or not Image_module or not hasattr(self, '_kraken_models'):
            return ""

        kraken_binarization = getattr(self, '_kraken_binarization', None)
        kraken_pageseg = getattr(self, '_kraken_pageseg', None)
        kraken_rpred = getattr(self, '_kraken_rpred', None) # recognition predictor
        kraken_models = self._kraken_models

        if not all([kraken_binarization, kraken_pageseg, kraken_rpred]):
            logging.warning("Kraken API submodules (binarization, pageseg, rpred) not fully available.")
            return ""
            
        text_parts = []
        images = None
        model = None
        poppler_path_dir = self._get_poppler_path()
        conversion_timeout = 60

        try:
            # Load default model once
            try:
                # Ensure default model is available or downloaded
                default_model_path = kraken_rpred.get_default_model() # Might raise exception if not found
                model = kraken_models.load_any(default_model_path)
                if self._debug: logging.debug(f"Kraken default model loaded: {default_model_path}")
            except Exception as model_load_e:
                logging.error(f"Failed to load default Kraken model: {model_load_e}. Kraken OCR will likely fail.")
                # Attempt to download if it looks like a path issue
                if "No such file or directory" in str(model_load_e) and shutil.which('kraken'):
                    logging.info("Attempting to download default Kraken model using CLI...")
                    try:
                        run_process([shutil.which('kraken'), "get", "10.5281/zenodo.10592716"], timeout=120)
                        default_model_path = kraken_rpred.get_default_model() # try again
                        model = kraken_models.load_any(default_model_path)
                        logging.info(f"Successfully loaded Kraken model after download: {default_model_path}")
                    except Exception as download_e:
                        logging.error(f"Failed to download or load Kraken model after attempt: {download_e}")
                        self._ocr_failed_methods.add('kraken')
                        return ""
                else:
                    self._ocr_failed_methods.add('kraken')
                    return ""


            conversion_args = {'dpi': 300, 'thread_count': 1, 'grayscale': True, 'userpw': self._password, 'poppler_path': poppler_path_dir}
            conversion_args = {k: v for k, v in conversion_args.items() if v is not None}
            images = pdf2image_module.convert_from_path(pdf_path, timeout=conversion_timeout, **conversion_args)

            for i, pil_image in enumerate(images):
                if shutdown_flag.is_set(): break
                try:
                    pil_image = self._preprocess_ocr_image(pil_image)
                    bw_im = kraken_binarization.nlbin(pil_image)
                    seg = kraken_pageseg.segment(bw_im)
                    if seg and hasattr(seg, 'lines') and seg.lines:
                        # Pass PIL image for recognition as per some examples, bw_im is also PIL.Image
                        pred_it = kraken_rpred.rpred(model, pil_image, seg) 
                        page_text = "\n".join(record.prediction for record in pred_it if hasattr(record, 'prediction'))
                        if page_text.strip(): text_parts.append(page_text.strip())
                except Exception as page_e:
                    if self._debug: logging.debug(f"Kraken API page {i+1} failed: {page_e}")
                finally:
                    pil_image.close()
                if progress_callback: progress_callback(1)
            return "\n\n".join(text_parts)
        except Exception as e:
            if self._debug: logging.error(f"Kraken API extraction for {pdf_path} failed: {e}")
            self._ocr_failed_methods.add('kraken')
            return ""
        finally:
            if images: 
                for img in images:
                    try: img.close()
                    except: pass
            self._clear_gpu_memory() # Kraken can use GPU