# extractors/mobi_extractor.py:

import os
import logging
import shutil
import tempfile
import subprocess
import sys # For sys.executable
import re
from typing import Optional, List, Dict, Any, Callable, Tuple
from tqdm import tqdm

# Direct import for utilities, assuming 'utils.py' is in PYTHONPATH
# (which it will be if running from the root directory)
try:
    from utils import ImportCache, run_process, shutdown_flag # Added shutdown_flag
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py in pdf_extractor.py: {e}")
    # Depending on how critical these are, you might re-raise or provide dummy objects
    # For now, let's assume they are critical.
    raise

class MOBIExtractor:
    """MOBI text extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods: Optional[Dict[str, bool]] = None
        self._binary_paths: Dict[str, str] = binary_paths if binary_paths is not None else {}
        self._kindleunpack_script_path: Optional[str] = None # Store path if script is found
        self._kindleunpack_module: Optional[Any] = None # Store module if importable

        # Initialize kindleunpack details
        self._kindleunpack_type, self._kindleunpack_location = self._find_kindleunpack_source()
        if self._kindleunpack_type == 'script':
            self._kindleunpack_script_path = self._kindleunpack_location
        elif self._kindleunpack_type == 'module':
            # Try to import it to confirm, but full init in _init_kindleunpack
            if self._import_cache.is_available('kindleunpack'):
                 if self._debug: logging.debug("kindleunpack Python module seems available via ImportCache.")
            else: # If ImportCache says no, then module type is invalid
                 self._kindleunpack_type = None 
                 self._kindleunpack_location = None

        if self._debug and self._kindleunpack_type:
            logging.debug(f"MOBIExtractor: Found kindleunpack as {self._kindleunpack_type} at: {self._kindleunpack_location}")

    def _find_kindleunpack_source(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find kindleunpack module or script.
        Returns: (type: 'module' or 'script', location: path or module name)
        """
        # 1. Check for Python module 'kindleunpack'
        if self._import_cache.is_available('kindleunpack'):
            try:
                # Further check if it can actually be imported (find_spec is not enough for all cases)
                ku_test = self._import_cache.import_module('kindleunpack')
                if hasattr(ku_test, 'unpack'): # A common function name in such tools
                    return 'module', 'kindleunpack' 
            except Exception as e:
                if self._debug: logging.debug(f"KindleUnpack module import test failed: {e}")
        
        # 2. Check PATH for kindleunpack executable/script
        # This could be a Python script made executable or a compiled binary
        kindleunpack_cmd = shutil.which('kindleunpack')
        if kindleunpack_cmd:
            return 'script', kindleunpack_cmd
        
        # 3. Check common known script locations for kindleunpack.py
        # (This part was from the original script, might need adjustment for robustness)
        script_name = "kindleunpack.py" # Common name for the script
        potential_script_dirs = [
            # Locations relative to user's home or common dev paths
            os.path.expanduser("~/KindleUnpack/lib"),
            os.path.expanduser("~/code/KindleUnpack/lib"),
            # System-wide script locations (less common for non-packaged scripts)
            "/usr/local/bin",
            "/opt/kindleunpack", # Example custom install dir
        ]
        # Also check common Python script directories if site-packages wasn't it
        for path_prefix in sys.path:
            if os.path.isdir(path_prefix):
                potential_script_dirs.append(path_prefix) # Check in PYTHONPATH dirs
                # Check one level down if 'kindleunpack' is a subdir there
                potential_script_dirs.append(os.path.join(path_prefix, 'kindleunpack'))
                potential_script_dirs.append(os.path.join(path_prefix, 'KindleUnpack', 'lib'))


        for directory in potential_script_dirs:
            if os.path.isdir(directory): # Ensure directory exists
                script_path = os.path.join(directory, script_name)
                if os.path.exists(script_path) and os.access(script_path, os.X_OK if not script_path.endswith('.py') else os.R_OK):
                    if self._debug: logging.debug(f"Found potential kindleunpack.py script at: {script_path}")
                    return 'script', script_path
        
        return None, None


    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            self._available_methods = {
                'mobi': self._import_cache.is_available('mobi'),
                'kindleunpack': self._kindleunpack_type is not None,
                'calibre': self._check_calibre_available(),
                'zipfile': self._import_cache.is_available('zipfile') # Basic fallback always available
            }
            if self._debug:
                logging.debug(f"MOBIExtractor available_methods: {self._available_methods}")
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

    def extract_text(self, mobi_path: str, preferred_method: Optional[str] = None,
                     progress_callback: Optional[Callable] = None, **kwargs) -> str:
        methods = ['mobi', 'kindleunpack', 'calibre', 'zipfile']
        file_basename = os.path.basename(mobi_path)
        
        if preferred_method and preferred_method in methods:
            methods.remove(preferred_method)
            methods.insert(0, preferred_method)
        
        extracted_text = ""
        with tqdm(total=len(methods), desc=f"MOBI Methods ({file_basename})", unit="mthd", leave=False, position=1) as method_pbar:
            for method_name in methods:
                if shutdown_flag.is_set(): break
                method_pbar.set_description(f"MOBI Method [{method_name}] ({file_basename})")

                if not self.available_methods.get(method_name):
                    method_pbar.update(1)
                    continue
                
                try:
                    if progress_callback: progress_callback(0, f"mobi_try_{method_name}")
                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    
                    internal_cb = None
                    if progress_callback:
                         internal_cb = lambda num_items=1, step_desc=method_name: progress_callback(num_items, f"mobi_{step_desc}_item")

                    current_method_text = extraction_func(mobi_path, internal_cb)
                    
                    if current_method_text and current_method_text.strip():
                        extracted_text = current_method_text.strip()
                        logging.info(f"SUCCESS: Extracted {len(extracted_text)} chars from MOBI '{file_basename}' using: {method_name}")
                        if progress_callback: progress_callback(100, f"mobi_done_{method_name}")
                        method_pbar.update(1)
                        break
                    else:
                         if progress_callback: progress_callback(0, f"mobi_empty_{method_name}")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logging.warning(f"Error with MOBI method '{method_name}' for '{file_basename}': {e}", exc_info=self._debug)
                    if progress_callback: progress_callback(0, f"mobi_err_{method_name}")
                method_pbar.update(1)
        return extracted_text

    def _init_kindleunpack(self) -> bool:
        """Initializes kindleunpack module or confirms script path."""
        if self._kindleunpack_type == 'module':
            if self._kindleunpack_module is None: # Initialize only once
                self._kindleunpack_module = self._import_cache.import_module('kindleunpack')
            return self._kindleunpack_module is not None
        elif self._kindleunpack_type == 'script':
            return self._kindleunpack_script_path is not None and os.path.exists(self._kindleunpack_script_path)
        return False

    def extract_with_mobi(self, mobi_path: str, progress_callback: Optional[Callable] = None) -> str:
        mobi_module = self._import_cache.import_module('mobi') # Assumes available_methods check passed
        if not mobi_module: return ""
        
        text_parts = []
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                # mobi.Mobi requires filename as string
                book = mobi_module.Mobi(mobi_path)
                book.parse() # This can be time-consuming

                if progress_callback: progress_callback(50) # Rough progress after parsing

                # Extract HTML content (often the richest source)
                html_content = ""
                if hasattr(book, 'html') and book.html: # Some versions use 'html' attribute
                    html_content = book.html
                elif hasattr(book, 'markup') and book.markup: # Others might use 'markup'
                    html_content = book.markup
                # The original script used book.raw_html - check if this attribute exists
                elif hasattr(book, 'raw_html') and book.raw_html:
                    html_content = book.raw_html

                if html_content:
                    BeautifulSoup_class = self._import_cache.import_module('bs4', 'BeautifulSoup')
                    if BeautifulSoup_class:
                        soup = BeautifulSoup_class(html_content, 'html.parser')
                        for script_style in soup(["script", "style"]):
                            script_style.extract()
                        text_parts.append(soup.get_text(separator='\n', strip=True))
                    else: # Basic regex cleanup if bs4 not found
                        text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
                        text = re.sub(r'<[^>]+>', ' ', text)
                        text_parts.append(re.sub(r'\s+', ' ', text).strip())
                
                # Consider other metadata if available (title, author)
                # This isn't raw text but can be useful context if primary text fails.
                # For now, focusing on text extraction.

                if progress_callback: progress_callback(50) # Remaining progress
                return "\n\n".join(filter(None, text_parts))

            except Exception as e:
                logging.debug(f"Python 'mobi' library extraction failed for {mobi_path}: {e}", exc_info=self._debug)
                return ""

    def extract_with_kindleunpack(self, mobi_path: str, progress_callback: Optional[Callable] = None) -> str:
        if not self._init_kindleunpack():
            logging.warning("KindleUnpack not initialized or available.")
            return ""
        
        text_content = ""
        with tempfile.TemporaryDirectory() as tempdir_unpack:
            try:
                if self._kindleunpack_type == 'module' and self._kindleunpack_module:
                    # Example: self._kindleunpack_module.unpack(mobi_path, tempdir_unpack)
                    # The actual API of kindleunpack module might vary.
                    # This requires knowing the module's API.
                    # For now, let's assume it has a function similar to the script.
                    # If the module has a main function or class:
                    if hasattr(self._kindleunpack_module, 'main'): # Common pattern
                        # Need to simulate command-line args if 'main' expects them
                        # Or find a direct 'unpack' function if it exists
                        logging.warning("KindleUnpack module usage needs specific API knowledge. Trying script method as fallback if direct module call fails.")
                        # For now, assume script is the primary way if direct module call is complex
                        if not self._kindleunpack_script_path: return "" # No script fallback
                        # Fall through to script execution
                    else: # No obvious 'main' or 'unpack', try to use script
                         if not self._kindleunpack_script_path: return ""
                
                # Execute script if module way is not straightforward or failed
                if self._kindleunpack_type == 'script' or not self._kindleunpack_module : # Fallback to script
                    if not self._kindleunpack_script_path or not os.path.exists(self._kindleunpack_script_path):
                        logging.warning(f"KindleUnpack script not found at {self._kindleunpack_script_path}")
                        return ""

                    cmd = [sys.executable, self._kindleunpack_script_path, mobi_path, tempdir_unpack]
                    # If kindleunpack_script_path is already an executable (not .py)
                    if not self._kindleunpack_script_path.lower().endswith('.py') and os.access(self._kindleunpack_script_path, os.X_OK):
                        cmd = [self._kindleunpack_script_path, mobi_path, tempdir_unpack]
                    
                    logging.debug(f"Running KindleUnpack script: {' '.join(map(escape_special_chars, cmd))}")
                    result = run_process(cmd, timeout=180) # 3 min timeout

                    if result.returncode != 0:
                        logging.warning(f"KindleUnpack script execution failed (code {result.returncode}). Stderr: {result.stderr.strip()[:200]}")
                        return ""
                
                if progress_callback: progress_callback(50) # After unpacking

                # Process extracted files (HTML, TXT)
                html_files_found = []
                for root, _, files in os.walk(tempdir_unpack):
                    for file in files:
                        if file.lower().endswith(('.html', '.htm', '.xhtml', '.txt')):
                            html_files_found.append(os.path.join(root, file))
                
                if not html_files_found:
                    logging.warning("KindleUnpack extracted no HTML/TXT files.")
                    return ""

                temp_text_parts = []
                BeautifulSoup_class = self._import_cache.import_module('bs4', 'BeautifulSoup')

                for item_path in html_files_found:
                    try:
                        with open(item_path, 'r', encoding='utf-8', errors='replace') as f_item:
                            item_content = f_item.read()
                        if item_path.lower().endswith(('.html', '.htm', '.xhtml')):
                            if BeautifulSoup_class and self.h: # Need BS4 and html2text
                                soup = BeautifulSoup_class(item_content, 'html.parser')
                                for tag in soup(['script', 'style']): tag.extract()
                                temp_text_parts.append(self.h.handle(str(soup)).strip())
                            else: # Basic regex if no bs4/html2text
                                text = re.sub(r'<style[^>]*>.*?</style>', '', item_content, flags=re.DOTALL|re.IGNORECASE)
                                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL|re.IGNORECASE)
                                text = re.sub(r'<[^>]+>', ' ', text)
                                temp_text_parts.append(re.sub(r'\s+', ' ', text).strip())
                        elif item_path.lower().endswith('.txt'):
                            temp_text_parts.append(item_content.strip())
                    except Exception as e_file:
                         logging.debug(f"Error processing unpacked file {item_path}: {e_file}")
                
                text_content = "\n\n".join(filter(None, temp_text_parts))
                if progress_callback: progress_callback(50) # Remaining progress
                return text_content

            except Exception as e:
                logging.warning(f"KindleUnpack method failed for {mobi_path}: {e}", exc_info=self._debug)
                return ""

    def extract_with_calibre(self, mobi_path: str, progress_callback: Optional[Callable] = None) -> str:
        calibre_bin = self._binary_paths.get('ebook-converter') or shutil.which('ebook-converter') or shutil.which('ebook-convert')
        if not calibre_bin:
            logging.warning("Calibre (ebook-converter) not found for MOBI extraction.")
            return ""
        
        temp_output_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                temp_output_file = tmp.name
            
            cmd = [calibre_bin, mobi_path, temp_output_file]
            # Add MOBI specific options if any, e.g., --mobi-keep-original-images
            # For text extraction, default usually suffices.
            if self._debug: logging.debug(f"Running Calibre for MOBI: {' '.join(map(escape_special_chars, cmd))}")
            
            result = run_process(cmd, timeout=180) # 3 min timeout

            if result.returncode != 0:
                logging.warning(f"Calibre MOBI conversion failed (code {result.returncode}). Stderr: {result.stderr.strip()[:200]}")
                return ""
            
            extracted_text = ""
            if os.path.exists(temp_output_file):
                with open(temp_output_file, 'r', encoding='utf-8', errors='replace') as f:
                    extracted_text = f.read().strip()
            else:
                logging.warning(f"Calibre output file not created: {temp_output_file}")

            if progress_callback: progress_callback(1) # Simplified progress
            return extracted_text
        except Exception as e:
            logging.error(f"Exception during Calibre MOBI extraction for {mobi_path}: {e}", exc_info=self._debug)
            return ""
        finally:
            if temp_output_file and os.path.exists(temp_output_file):
                try: os.unlink(temp_output_file)
                except: pass
    
    def extract_with_zipfile(self, mobi_path: str, progress_callback: Optional[Callable] = None) -> str:
        # MOBI files are not typically ZIP archives in the standard sense.
        # This method is a very rough fallback, trying to find readable strings.
        # It might be more effective to use a MOBI-specific raw parsing or rely on other tools.
        # For now, a basic string extraction from binary.
        logging.debug(f"Attempting zipfile/raw string extraction for MOBI {mobi_path} (low success chance).")
        text_parts = []
        try:
            with open(mobi_path, 'rb') as f:
                content_bytes = f.read()
            
            # Try to decode with common encodings and extract printable sequences
            potential_text = ""
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    decoded_text = content_bytes.decode(encoding, errors='ignore')
                    # Look for sequences of at least N printable characters
                    # This is very heuristic.
                    printable_sequences = re.findall(r'[ -~\n\r\t]{50,}', decoded_text) # Min 50 printable chars
                    if printable_sequences:
                        potential_text = "\n".join(printable_sequences)
                        break # Found something
                except Exception:
                    continue
            
            if potential_text:
                # Further very basic HTML tag stripping if any obvious tags are found
                if '<' in potential_text and '>' in potential_text:
                    potential_text = re.sub(r'<[^>]+>', ' ', potential_text)
                text_parts.append(re.sub(r'\s+', ' ', potential_text).strip())

            if progress_callback: progress_callback(1)
            return "\n\n".join(filter(None, text_parts))
        except Exception as e:
            logging.debug(f"MOBI raw string extraction failed for {mobi_path}: {e}", exc_info=self._debug)
            return ""