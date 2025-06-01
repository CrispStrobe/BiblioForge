# extractors/html_extractor.py

import os
import logging
import shutil
import re
from typing import Optional, List, Dict, Any, Callable
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

class HTMLExtractor:
    """HTML file extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods: Optional[Dict[str, bool]] = None
        # binary_paths are kept for consistency, though not heavily used by current HTML methods
        self._binary_paths: Dict[str, str] = binary_paths if binary_paths is not None else {}

    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            self._available_methods = {
                # Calibre is not a primary method for raw HTML, but check can exist
                # 'calibre': self._check_calibre_available(), 
                'bs4': self._import_cache.is_available('bs4'),
                'html2text': self._import_cache.is_available('html2text'),
                'lxml': self._import_cache.is_available('lxml', submodules=['html']), # lxml.html
                'regex': True  # Basic regex is always available as a fallback
            }
            if self._debug:
                logging.debug(f"HTMLExtractor available_methods: {self._available_methods}")
        return self._available_methods
    
    def _check_calibre_available(self) -> bool: # Kept for consistency, but not used in HTML flow
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

    def extract_text(self, html_path: str, preferred_method: Optional[str] = None,
                     progress_callback: Optional[Callable] = None, **kwargs) -> str:
        # kwargs for consistency, not used by HTML specific methods for now
        methods = ['bs4', 'html2text', 'lxml', 'regex'] # Default order
        file_basename = os.path.basename(html_path)
        
        if preferred_method and preferred_method in methods:
            methods.remove(preferred_method)
            methods.insert(0, preferred_method)
        
        logging.info(f"HTMLExtractor: Attempting methods {methods} for '{file_basename}'")
        extracted_text = ""

        with tqdm(total=len(methods), desc=f"HTML Methods ({file_basename})", unit="mthd", leave=False, position=1) as method_pbar:
            for method_name in methods:
                if shutdown_flag.is_set(): break
                method_pbar.set_description(f"HTML Method [{method_name}] ({file_basename})")

                if not self.available_methods.get(method_name):
                    method_pbar.update(1)
                    continue
                
                try:
                    if progress_callback: progress_callback(0, f"html_try_{method_name}")
                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    
                    internal_cb = None # HTML methods are usually quick, per-item callback might be overkill
                    if progress_callback:
                         internal_cb = lambda num_items=1, step_desc=method_name: progress_callback(num_items, f"html_{step_desc}_item")

                    current_method_text = extraction_func(html_path, internal_cb)
                    
                    if current_method_text and current_method_text.strip():
                        extracted_text = current_method_text.strip()
                        logging.info(f"SUCCESS: Extracted {len(extracted_text)} chars from HTML '{file_basename}' using: {method_name}")
                        if progress_callback: progress_callback(100, f"html_done_{method_name}")
                        method_pbar.update(1)
                        break 
                    else:
                         if progress_callback: progress_callback(0, f"html_empty_{method_name}")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logging.warning(f"Error with HTML method '{method_name}' for '{file_basename}': {e}", exc_info=self._debug)
                    if progress_callback: progress_callback(0, f"html_err_{method_name}")
                method_pbar.update(1)
        
        if not extracted_text:
             logging.warning(f"HTMLExtractor: No text extracted from '{file_basename}'.")
        return extracted_text

    def _read_html_content(self, html_path: str) -> Optional[str]:
        """Reads HTML content from file, trying common encodings."""
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings_to_try:
            try:
                with open(html_path, 'r', encoding=encoding, errors='strict' if encoding=='utf-8' else 'replace') as f:
                    return f.read()
            except UnicodeDecodeError:
                if self._debug: logging.debug(f"HTML read with {encoding} failed for {html_path}")
            except Exception as e:
                logging.warning(f"Failed to read HTML file {html_path} with encoding {encoding}: {e}")
                return None # Stop if general read error
        logging.warning(f"Could not decode HTML file {html_path} with common encodings.")
        return None


    def extract_with_bs4(self, html_path: str, progress_callback: Optional[Callable] = None) -> str:
        BeautifulSoup_class = self._import_cache.import_module('bs4', 'BeautifulSoup')
        if not BeautifulSoup_class: return ""
        
        html_content = self._read_html_content(html_path)
        if not html_content: return ""
            
        try:
            soup = BeautifulSoup_class(html_content, 'html.parser')
            # Remove common non-content tags
            for tag in soup(["script", "style", "meta", "link", "noscript", "header", "footer", "nav", "aside", "form", "button", "input", "select", "textarea"]):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            # Further clean-up of excessive newlines that might result from get_text
            text = re.sub(r'\n\s*\n', '\n\n', text).strip() # Consolidate multiple newlines
            
            if progress_callback: progress_callback(1)
            return text
        except Exception as e:
            logging.debug(f"BeautifulSoup (bs4) HTML extraction failed for {html_path}: {e}", exc_info=self._debug)
            return ""

    def extract_with_html2text(self, html_path: str, progress_callback: Optional[Callable] = None) -> str:
        HTML2Text_class = self._import_cache.import_module('html2text', 'HTML2Text')
        if not HTML2Text_class: return ""

        html_content = self._read_html_content(html_path)
        if not html_content: return ""

        try:
            h = HTML2Text_class()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_tables = False # Keep table structure if possible
            h.body_width = 0  # No wrapping
            h.unicode_snob = True
            h.escape_snob = True
            
            text = h.handle(html_content)
            if progress_callback: progress_callback(1)
            return text.strip()
        except Exception as e:
            logging.debug(f"html2text extraction failed for {html_path}: {e}", exc_info=self._debug)
            return ""

    def extract_with_lxml(self, html_path: str, progress_callback: Optional[Callable] = None) -> str:
        lxml_html_module = self._import_cache.import_module('lxml.html')
        if not lxml_html_module: return ""

        html_content = self._read_html_content(html_path)
        if not html_content: return ""

        try:
            root = lxml_html_module.fromstring(html_content)
            # Remove script, style, and other common non-content elements
            for elem_xpath in ['//script', '//style', '//meta', '//link', '//noscript', '//header', '//footer', '//nav', '//aside', '//form', '//button', '//input', '//select', '//textarea']:
                for elem in root.xpath(elem_xpath):
                    elem.drop_tree() # More thorough removal
            
            # Extract text content from all remaining elements
            # Using text_content() method correctly joins text from children.
            text_parts = [elem.text_content() for elem in root.xpath('//body//*[not(self::script or self::style)] | //body/text()[normalize-space()]')]
            text = "\n".join(filter(None, (p.strip() for p in text_parts))).strip()
            text = re.sub(r'\n\s*\n', '\n\n', text).strip()


            if not text and not root.xpath('//body'): # If no body tag, try to get all text
                 text_parts = [elem.text_content() for elem in root.xpath('//*[not(self::script or self::style)] | //text()[normalize-space()]')]
                 text = "\n".join(filter(None, (p.strip() for p in text_parts))).strip()
                 text = re.sub(r'\n\s*\n', '\n\n', text).strip()


            if progress_callback: progress_callback(1)
            return text
        except Exception as e:
            logging.debug(f"lxml HTML extraction failed for {html_path}: {e}", exc_info=self._debug)
            return ""

    def extract_with_regex(self, html_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Extract text using basic regex patterns as a last resort."""
        html_content = self._read_html_content(html_path)
        if not html_content: return ""

        try:
            # Remove script and style sections
            text = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
            # Remove comments
            text = re.sub(r'', ' ', text, flags=re.DOTALL)
            # Remove all HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Decode common HTML entities (basic set)
            entities = {'&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'"}
            for entity, char_val in entities.items(): # Renamed char to char_val
                text = text.replace(entity, char_val)
            
            # Normalize whitespace and clean up
            text = re.sub(r'\s+', ' ', text).strip()
            # Attempt to form paragraphs from lines with significant content
            lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 10] # Only keep meaningful lines

            if progress_callback: progress_callback(1)
            return '\n\n'.join(lines) if lines else text # Fallback to raw text if no meaningful lines
        except Exception as e:
            logging.debug(f"Regex HTML extraction failed for {html_path}: {e}", exc_info=self._debug)
            return ""