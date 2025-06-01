# extractors/text_extractor.py

import os
import logging
import shutil
import tempfile
import subprocess # For Calibre if not using run_process
from typing import Optional, List, Dict, Any, Callable
from tqdm import tqdm
import re # For any regex operations if needed

# Direct import for utilities, assuming 'utils.py' is in PYTHONPATH
# (which it will be if running from the root directory)
try:
    from utils import ImportCache, run_process, shutdown_flag # Added shutdown_flag
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py in pdf_extractor.py: {e}")
    # Depending on how critical these are, you might re-raise or provide dummy objects
    # For now, let's assume they are critical.
    raise

class TextExtractor:
    """Plain text file extraction with encoding detection and Calibre for other text-like formats."""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods: Optional[Dict[str, bool]] = None
        self._binary_paths: Dict[str, str] = binary_paths if binary_paths is not None else {}

    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            self._available_methods = {
                'direct': True,  # Direct file reading (UTF-8 default)
                'charset_detection': self._import_cache.is_available('chardet'),
                'encoding_detection': self._import_cache.is_available('ftfy'), # ftfy for fixing mojibake
                'calibre': self._check_calibre_available() # For DOCX, RTF etc. via text conversion
            }
            if self._debug:
                logging.debug(f"TextExtractor available_methods: {self._available_methods}")
        return self._available_methods
    
    def _check_calibre_available(self) -> bool:
        calibre_bin = self._binary_paths.get('ebook-converter')
        if calibre_bin and os.path.exists(calibre_bin):
            return True
        for bin_name in ['ebook-converter', 'ebook-convert']:
            found_path = shutil.which(bin_name)
            if found_path:
                if 'ebook-converter' not in self._binary_paths or not self._binary_paths.get('ebook-converter'):
                    self._binary_paths['ebook-converter'] = found_path
                return True
        return False

    def extract_text(self, txt_path: str, preferred_method: Optional[str] = None,
                     progress_callback: Optional[Callable] = None, **kwargs) -> str:
        file_basename = os.path.basename(txt_path)
        file_ext = os.path.splitext(txt_path)[1].lower()
        
        # Prioritize Calibre for specific office/ebook formats handled as "Text"
        calibre_formats = ['.docx', '.doc', '.rtf', '.fb2', '.pdb', '.lit', 
                           '.odt', '.lrf', '.cbz', '.cbr', '.chm', '.snb', '.tcr']
        
        methods = ['charset_detection', 'encoding_detection', 'direct'] # Default for .txt, .md
        if file_ext in calibre_formats:
            methods.insert(0, 'calibre') # Try Calibre first for these

        if preferred_method and preferred_method in methods:
            methods.remove(preferred_method)
            methods.insert(0, preferred_method)
        
        logging.info(f"TextExtractor: Attempting methods {methods} for '{file_basename}'")
        extracted_text = ""

        with tqdm(total=len(methods), desc=f"Text Methods ({file_basename})", unit="mthd", leave=False, position=1) as method_pbar:
            for method_name in methods:
                if shutdown_flag.is_set(): break
                method_pbar.set_description(f"Text Method [{method_name}] ({file_basename})")

                if not self.available_methods.get(method_name):
                    method_pbar.update(1)
                    continue
                
                try:
                    if progress_callback: progress_callback(0, f"text_try_{method_name}")
                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    
                    internal_cb = None
                    if progress_callback:
                         internal_cb = lambda num_items=1, step_desc=method_name: progress_callback(num_items, f"text_{step_desc}_item")

                    current_method_text = extraction_func(txt_path, internal_cb)
                    
                    if current_method_text and current_method_text.strip():
                        extracted_text = current_method_text.strip()
                        logging.info(f"SUCCESS: Extracted {len(extracted_text)} chars from '{file_basename}' using: {method_name}")
                        if progress_callback: progress_callback(100, f"text_done_{method_name}")
                        method_pbar.update(1)
                        break
                    else:
                        if progress_callback: progress_callback(0, f"text_empty_{method_name}")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logging.warning(f"Error with text method '{method_name}' for '{file_basename}': {e}", exc_info=self._debug)
                    if progress_callback: progress_callback(0, f"text_err_{method_name}")
                method_pbar.update(1)
        
        if not extracted_text:
             logging.warning(f"TextExtractor: No text extracted from '{file_basename}'.")
        return extracted_text

    def extract_with_direct(self, txt_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Extract text directly with UTF-8 encoding, then common fallbacks."""
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings_to_try:
            try:
                with open(txt_path, 'r', encoding=encoding, errors='strict' if encoding == 'utf-8' else 'replace') as f:
                    text = f.read()
                if progress_callback: progress_callback(1)
                if self._debug: logging.debug(f"Successfully read {txt_path} with encoding {encoding}")
                return text
            except UnicodeDecodeError:
                if self._debug: logging.debug(f"Direct read with {encoding} failed for {txt_path}")
            except Exception as e:
                logging.debug(f"Direct text extraction with {encoding} failed for {txt_path}: {e}", exc_info=self._debug)
                return "" # General error, stop trying direct methods
        return "" # Failed all direct attempts
        
    def extract_with_charset_detection(self, txt_path: str, progress_callback: Optional[Callable] = None) -> str:
        chardet_module = self._import_cache.import_module('chardet')
        if not chardet_module: return ""
        
        try:
            with open(txt_path, 'rb') as f_raw:
                raw_data = f_raw.read()
            
            result = chardet_module.detect(raw_data)
            encoding = result.get('encoding')
            confidence = result.get('confidence', 0)
            
            if encoding and confidence > 0.7:
                if self._debug: logging.debug(f"Chardet detected encoding {encoding} with confidence {confidence:.2f} for {txt_path}")
                text = raw_data.decode(encoding, errors='replace')
                if progress_callback: progress_callback(1)
                return text
            else:
                if self._debug: logging.debug(f"Chardet low confidence ({confidence:.2f}) or no encoding for {txt_path}")
                return ""
        except Exception as e:
            logging.debug(f"Charset detection failed for {txt_path}: {e}", exc_info=self._debug)
            return ""

    def extract_with_encoding_detection(self, txt_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Extract text using ftfy for fixing encoding issues (mojibake)."""
        ftfy_module = self._import_cache.import_module('ftfy')
        if not ftfy_module: return ""
        
        try:
            # ftfy works best on text that's already "decoded" somehow, even if incorrectly.
            # Try reading with a common fallback first.
            raw_text = ""
            try:
                with open(txt_path, 'r', encoding='latin-1', errors='replace') as f: # latin-1 is a safe bet for many files
                    raw_text = f.read()
            except Exception: # If even latin-1 fails, try reading as bytes and decoding utf-8 with replace
                try:
                    with open(txt_path, 'rb') as f_bytes:
                        raw_text = f_bytes.read().decode('utf-8', errors='replace')
                except Exception as e_bytes:
                     logging.debug(f"Ftfy: Could not read file {txt_path} even as bytes: {e_bytes}")
                     return ""
            
            if not raw_text: return ""

            fixed_text = ftfy_module.fix_text(raw_text)
            if progress_callback: progress_callback(1)
            if self._debug: logging.debug(f"Ftfy processed {txt_path}")
            return fixed_text
        except Exception as e:
            logging.debug(f"Ftfy encoding detection failed for {txt_path}: {e}", exc_info=self._debug)
            return ""
            
    def extract_with_calibre(self, file_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Extract text from various formats (DOCX, RTF, etc.) using Calibre."""
        calibre_bin = self._binary_paths.get('ebook-converter') or shutil.which('ebook-converter') or shutil.which('ebook-convert')
        if not calibre_bin:
            logging.warning(f"Calibre (ebook-converter) not found for TextExtractor on {file_path}.")
            return ""
        
        temp_output_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                temp_output_file = tmp.name
            
            cmd = [calibre_bin, file_path, temp_output_file]
            if self._debug: logging.debug(f"Running Calibre for text conversion: {' '.join(map(escape_special_chars, cmd))}")
            
            result = run_process(cmd, timeout=180) # 3 min timeout

            if result.returncode != 0:
                logging.warning(f"Calibre text conversion for '{file_path}' failed (code {result.returncode}). Stderr: {result.stderr.strip()[:200]}")
                return ""
            
            extracted_text = ""
            if os.path.exists(temp_output_file):
                with open(temp_output_file, 'r', encoding='utf-8', errors='replace') as f:
                    extracted_text = f.read().strip()
            else:
                logging.warning(f"Calibre output text file not created: {temp_output_file}")

            if progress_callback: progress_callback(1)
            return extracted_text
        except Exception as e:
            logging.error(f"Exception during Calibre text conversion for {file_path}: {e}", exc_info=self._debug)
            return ""
        finally:
            if temp_output_file and os.path.exists(temp_output_file):
                try: os.unlink(temp_output_file)
                except: pass