# extractors/epub_extractor.py:

import os
import logging
import shutil
import tempfile
import subprocess # For Calibre if not using run_process
import sys # For kindleunpack script execution in MOBIExtractor (example, not EPUB)
from typing import Optional, List, Dict, Any, Callable
from tqdm import tqdm
import re # For some fallback methods or cleaning

# Direct import for utilities, assuming 'utils.py' is in PYTHONPATH
# (which it will be if running from the root directory)
try:
    from utils import ImportCache, run_process, shutdown_flag # Added shutdown_flag
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py in pdf_extractor.py: {e}")
    # Depending on how critical these are, you might re-raise or provide dummy objects
    # For now, let's assume they are critical.
    raise

class EPUBExtractor:
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods: Optional[Dict[str, bool]] = None
        self._binary_paths: Dict[str, str] = binary_paths if binary_paths is not None else {}

        self.h: Optional[Any] = None  # html2text instance
        if self._import_cache.is_available('html2text'):
            try:
                # html2text_module = self._import_cache.import_module('html2text')
                # self.h = html2text_module.HTML2Text()
                # A more direct way if HTML2Text is the class itself:
                HTML2Text_class = self._import_cache.import_module('html2text', 'HTML2Text')
                if HTML2Text_class:
                    self.h = HTML2Text_class()
                    self.h.ignore_links = True
                    self.h.ignore_images = True
                    self.h.ignore_tables = False
                    self.h.body_width = 0
                    self.h.unicode_snob = True
                    self.h.escape_snob = True
                    if self._debug:
                        logging.debug("EPUBExtractor: html2text initialized successfully.")
                else: # Should not happen if is_available is true and import_module works
                    if self._debug: logging.error("EPUBExtractor: HTML2Text class not found via import_module.")
            except Exception as e_html2text:
                if self._debug:
                    logging.error(f"EPUBExtractor: Failed to initialize html2text: {e_html2text}", exc_info=self._debug)
                self.h = None
        elif self._debug:
            logging.debug("EPUBExtractor: html2text module not available.")

    def _get_module_or_log(self, module_name: str, attribute_name: Optional[str] = None, method_context: str = "EPUB Method", file_basename: str = "epub_file") -> Optional[Any]:
        """Helper to import module/attribute and log if unavailable for a given method."""
        # Ensure submodule is handled correctly by import_module
        target_to_import = module_name
        if attribute_name:
            # ImportCache's import_module can handle attribute directly if submodule param is used
            # Or we can get the main module then getattr
            try:
                return self._import_cache.import_module(module_name, submodule=attribute_name)
            except ImportError: # Main module might be missing
                 logging.warning(f"{method_context} for '{file_basename}': Dependency '{module_name}' not available. Cannot use this aspect.")
                 return None
            except AttributeError: # Submodule/attribute might be missing
                 logging.warning(f"{method_context} for '{file_basename}': Attribute '{attribute_name}' not found in '{module_name}'. Cannot use this aspect.")
                 return None
        else: # Just import the module
            if not self._import_cache.is_available(target_to_import):
                logging.warning(f"{method_context} for '{file_basename}': Dependency '{target_to_import}' not available. Cannot use this method.")
                return None
            try:
                return self._import_cache.import_module(target_to_import)
            except Exception as e:
                logging.error(f"{method_context} for '{file_basename}': Failed to import '{target_to_import}': {e}", exc_info=self._debug)
                return None


    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            # Check for ebooklib and its dependency bs4
            can_ebooklib = self._import_cache.is_available('ebooklib') and \
                           self._import_cache.is_available('bs4')
            
            # Check for bs4 method (uses zipfile and html2text instance)
            can_bs4 = self._import_cache.is_available('bs4') and \
                      self._import_cache.is_available('zipfile') and \
                      (self.h is not None) # html2text instance must be ready
            
            can_epub2txt = self._import_cache.is_available('epub2txt')
            can_calibre = self._check_calibre_available()
            can_zipfile = self._import_cache.is_available('zipfile') # For basic zipfile + regex

            self._available_methods = {
                'ebooklib': can_ebooklib,
                'bs4': can_bs4,
                'epub2txt': can_epub2txt,
                'calibre': can_calibre,
                'zipfile': can_zipfile # Fallback, minimal dependencies
            }
            if self._debug:
                logging.debug(f"EPUBExtractor available_methods check results: {self._available_methods}")
        return self._available_methods

    def _check_calibre_available(self) -> bool:
        calibre_bin = self._binary_paths.get('ebook-converter')
        if calibre_bin and os.path.exists(calibre_bin):
            if self._debug: logging.debug(f"Calibre (ebook-converter) found via _binary_paths for EPUB: {calibre_bin}")
            return True
        
        for bin_name in ['ebook-converter', 'ebook-convert']:
            found_path = shutil.which(bin_name)
            if found_path:
                if self._debug: logging.debug(f"Calibre ({bin_name}) found via shutil.which for EPUB: {found_path}")
                if 'ebook-converter' not in self._binary_paths or not self._binary_paths.get('ebook-converter'):
                    self._binary_paths['ebook-converter'] = found_path
                return True
        if self._debug: logging.debug("Calibre binary not found for EPUBExtractor.")
        return False

    def extract_text(self, epub_path: str, preferred_method: Optional[str] = None,
                     progress_callback: Optional[Callable] = None, **kwargs) -> str:
        # kwargs is for consistency, EPUBExtractor doesn't use them specifically for now
        methods = ['ebooklib', 'bs4', 'epub2txt', 'calibre', 'zipfile']
        
        file_basename = os.path.basename(epub_path)
        logging.info(f"Starting EPUB extraction for: '{file_basename}'")

        if preferred_method:
            if preferred_method not in methods:
                logging.warning(f"Invalid preferred method for EPUB '{file_basename}': {preferred_method}. Using default order.")
            else:
                methods.remove(preferred_method)
                methods.insert(0, preferred_method)
                logging.info(f"Using preferred method '{preferred_method}'. New EPUB method order: {methods}")
        else:
            logging.info(f"Using default EPUB method order: {methods}")

        extracted_text_content = ""

        with tqdm(total=len(methods), desc=f"EPUB Methods ({file_basename})", unit="mthd", leave=False, position=1) as method_pbar:
            for method_name in methods:
                if shutdown_flag.is_set():
                    logging.info(f"Shutdown flag detected during EPUB method loop for '{file_basename}'. Aborting.")
                    break
                
                method_pbar.set_description(f"EPUB Method [{method_name}] ({file_basename})")

                if not self.available_methods.get(method_name):
                    logging.info(f"EPUB method '{method_name}' for '{file_basename}' is not available. Skipping.")
                    method_pbar.update(1)
                    continue

                logging.info(f"Attempting EPUB '{file_basename}' with method: {method_name}")
                current_method_text = ""
                try:
                    if progress_callback: progress_callback(0, f"epub_try_{method_name}") # Start of method attempt

                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    
                    internal_cb = None
                    if progress_callback:
                        # Pass a value like 1 for each item processed within the method
                        internal_cb = lambda num_items_processed=1, step_desc=method_name: progress_callback(num_items_processed, f"epub_{step_desc}_item")

                    current_method_text = extraction_func(epub_path, internal_cb)

                    if current_method_text and current_method_text.strip():
                        extracted_text_content = current_method_text.strip()
                        logging.info(f"SUCCESS: Extracted {len(extracted_text_content)} chars from EPUB '{file_basename}' using: {method_name}")
                        if progress_callback: progress_callback(100, f"epub_done_{method_name}") # End of successful method
                        method_pbar.update(1) # Update overall method progress
                        break 
                    else:
                        logging.warning(f"EPUB method '{method_name}' returned NO TEXT for '{file_basename}'.")
                        if progress_callback: progress_callback(0, f"epub_empty_{method_name}")


                except KeyboardInterrupt:
                    logging.info(f"EPUB extraction with '{method_name}' for '{file_basename}' interrupted by user.")
                    if progress_callback: progress_callback(0, f"epub_intr_{method_name}")
                    raise # Re-raise to stop processing
                except Exception as e:
                    logging.warning(f"Error during EPUB extraction with method '{method_name}' for '{file_basename}': {str(e)}")
                    if self._debug:
                        logging.debug(f"Traceback for EPUB method '{method_name}' failure on '{file_basename}':", exc_info=True)
                    if progress_callback: progress_callback(0, f"epub_err_{method_name}")
                
                method_pbar.update(1)
        
        if not extracted_text_content:
            logging.warning(f"FINAL WARNING: No text extracted from EPUB '{file_basename}' after all attempts.")
        return extracted_text_content

    def extract_with_ebooklib(self, epub_path: str, progress_callback: Optional[Callable] = None) -> str:
        file_basename = os.path.basename(epub_path)
        logging.info(f"Ebooklib: Starting extraction for '{file_basename}'.")

        ebooklib_module = self._get_module_or_log('ebooklib', method_context='Ebooklib', file_basename=file_basename)
        epub_module = self._get_module_or_log('ebooklib.epub', method_context='Ebooklib', file_basename=file_basename)
        BeautifulSoup_class = self._get_module_or_log('bs4', attribute_name='BeautifulSoup', method_context='Ebooklib', file_basename=file_basename)


        if not all([ebooklib_module, epub_module, BeautifulSoup_class]):
            if progress_callback: progress_callback(100, "ebooklib_deps_missing") # 100% of this method failed
            return ""
        
        text_parts: List[str] = []
        book = None
        processed_item_count = 0
        items_to_process = []
        
        try:
            book = epub_module.read_epub(epub_path)
            items_in_spine_order = []
            if book.spine: # TOC order
                item_map = {item.id: item for item in book.items}
                for item_id, _ in book.spine:
                    item_from_map = item_map.get(item_id)
                    if item_from_map and item_from_map.get_type() == ebooklib_module.ITEM_DOCUMENT:
                        items_in_spine_order.append(item_from_map)
            
            items_to_process = items_in_spine_order or list(book.get_items_of_type(ebooklib_module.ITEM_DOCUMENT))
            
            if not items_to_process:
                logging.warning(f"Ebooklib: No processable document items found in '{file_basename}'.")
                if progress_callback: progress_callback(1) # Still call it once to signify attempt
                return ""
            
            logging.debug(f"Ebooklib: Found {len(items_to_process)} document items to process in '{file_basename}'.")
            
            # Total progress for this method will be len(items_to_process) if callback is item-based
            # If progress_callback is for the whole method, call it once at the end or for major steps.

            with tqdm(total=len(items_to_process), desc=f"Ebooklib items ({file_basename})", unit="item", leave=False, position=2) as pbar_items:
                for item_idx, item in enumerate(items_to_process):
                    if shutdown_flag.is_set(): break
                    pbar_items.set_description(f"Ebooklib item {item_idx+1}/{len(items_to_process)} ({item.get_name()[:20]})")
                    item_text_content = ""
                    try:
                        content_bytes = item.get_content()
                        detected_encoding = 'utf-8' # Default
                        try: # Try to sniff encoding from XML prolog
                            preamble = content_bytes[:150].decode('ascii', errors='ignore').lower()
                            match = re.search(r'encoding="([^"]+)"', preamble)
                            if match: detected_encoding = match.group(1)
                        except Exception: pass

                        content = content_bytes.decode(detected_encoding, errors='replace')
                        soup = BeautifulSoup_class(content, 'html.parser')
                        
                        for tag_type in soup(['script', 'style', 'nav', 'meta', 'link', 'head', 'title', 
                                            'figure', 'figcaption', 'aside', 'footer', 'header', 
                                            'annotation', 'ann', '[fallback]', 'svg', 'img']):
                            tag_type.decompose()
                        
                        body_tag = soup.find('body')
                        target_node = body_tag if body_tag else soup
                        
                        item_text_segments = [s.strip() for s in target_node.get_text(separator='\n', strip=False).splitlines() if s.strip()]
                        item_text_content = '\n'.join(item_text_segments)

                        if item_text_content:
                            text_parts.append(item_text_content)
                        processed_item_count +=1
                    except Exception as e_item:
                        logging.debug(f"Ebooklib: Item extraction failed for '{item.get_name()}' in '{file_basename}': {str(e_item)[:100]}", exc_info=self._debug)
                    finally:
                        pbar_items.update(1)
                        if progress_callback: progress_callback(1) 
                            
        except Exception as e_main:
            logging.warning(f"Ebooklib: Main processing failed for EPUB '{file_basename}': {str(e_main)[:100]}", exc_info=self._debug)
            if progress_callback: progress_callback(len(items_to_process) if items_to_process else 1) # Mark as complete
            return ""
        finally:
            if book: del book # Ebooklib doesn't have an explicit close for read_epub

        final_text = '\n\n'.join(filter(None, text_parts))
        logging.info(f"Ebooklib: Finished for '{file_basename}', processed {processed_item_count}/{len(items_to_process)} items, extracted {len(final_text)} total chars.")
        return final_text

    def extract_with_bs4(self, epub_path: str, progress_callback: Optional[Callable] = None) -> str:
        file_basename = os.path.basename(epub_path)
        logging.info(f"BS4/Zip: Starting extraction for '{file_basename}'.")

        BeautifulSoup_class = self._get_module_or_log('bs4', attribute_name='BeautifulSoup', method_context='BS4/Zip', file_basename=file_basename)
        zipfile_module = self._get_module_or_log('zipfile', method_context='BS4/Zip', file_basename=file_basename)
        
        if not all([BeautifulSoup_class, zipfile_module, self.h]): # self.h is html2text instance
            if progress_callback: progress_callback(1) # Method attempted but failed dependencies
            return ""
        
        text_parts: List[str] = []
        processed_item_count = 0
        html_files = []
        try:
            with zipfile_module.ZipFile(epub_path) as zf:
                html_files = [f for f in zf.namelist() if f.lower().endswith(('.html', '.xhtml', '.htm')) and 
                                                        not f.lower().startswith(('meta-inf/', 'mimetype'))] # Exclude mimetype too
                if not html_files:
                    logging.warning(f"BS4/Zip: No HTML/XHTML files found in EPUB '{file_basename}'.")
                    if progress_callback: progress_callback(1)
                    return ""
                
                logging.debug(f"BS4/Zip: Found {len(html_files)} HTML/XHTML files in '{file_basename}'.")
                with tqdm(total=len(html_files), desc=f"BS4/Zip items ({file_basename})", unit="file", leave=False, position=2) as pbar_html:
                    for file_idx, html_file_name in enumerate(html_files):
                        if shutdown_flag.is_set(): break
                        pbar_html.set_description(f"BS4/Zip item {file_idx+1}/{len(html_files)}")
                        try:
                            content_bytes = zf.read(html_file_name)
                            content = content_bytes.decode('utf-8', errors='replace')
                            soup = BeautifulSoup_class(content, 'html.parser')
                            for tag_type in soup(['script', 'style', 'nav', 'meta', 'link', 'head', 'title', 
                                                'figure', 'figcaption', 'aside', 'footer', 'header', 'annotation', 'svg', 'img']):
                                tag_type.decompose()
                            
                            item_text_content = self.h.handle(str(soup)).strip()
                            if item_text_content:
                                text_parts.append(item_text_content)
                            processed_item_count += 1
                        except Exception as e_item:
                            logging.debug(f"BS4/Zip: Failed to process HTML file '{html_file_name}' in '{file_basename}': {str(e_item)[:100]}", exc_info=self._debug)
                        finally:
                            pbar_html.update(1)
                            if progress_callback: progress_callback(1)
        except zipfile_module.BadZipFile:
            logging.warning(f"BS4/Zip: '{file_basename}' is not a valid zip file or is corrupted.")
            if progress_callback: progress_callback(len(html_files) if html_files else 1)
            return ""
        except Exception as e_main:
            logging.warning(f"BS4/Zip: Main processing failed for EPUB '{file_basename}': {str(e_main)[:100]}", exc_info=self._debug)
            if progress_callback: progress_callback(len(html_files) if html_files else 1)
            return ""
            
        final_text = '\n\n'.join(filter(None, text_parts))
        logging.info(f"BS4/Zip: Finished for '{file_basename}', processed {processed_item_count}/{len(html_files)} items, extracted {len(final_text)} total chars.")
        return final_text

    def extract_with_epub2txt(self, epub_path: str, progress_callback: Optional[Callable] = None) -> str:
        file_basename = os.path.basename(epub_path)
        logging.info(f"Epub2txt: Starting extraction for '{file_basename}'.")
        epub2txt_module = self._get_module_or_log('epub2txt', method_context='Epub2txt', file_basename=file_basename)
        if not epub2txt_module:
            if progress_callback: progress_callback(1)
            return ""

        try:
            converter_func = None
            if hasattr(epub2txt_module, 'epub2txt') and callable(getattr(epub2txt_module, 'epub2txt')):
                converter_func = getattr(epub2txt_module, 'epub2txt')
            elif callable(epub2txt_module): # If the module itself is callable
                converter_func = epub2txt_module
            
            if not converter_func:
                logging.warning(f"Epub2txt: Could not find callable 'epub2txt' function in module for '{file_basename}'.")
                if progress_callback: progress_callback(1)
                return ""

            result = converter_func(epub_path, outputlist=False) 
            
            extracted_text = ""
            if isinstance(result, str):
                extracted_text = result.strip()
            elif isinstance(result, list): 
                extracted_text = "\n\n".join(str(item).strip() for item in result if str(item).strip())
            
            if extracted_text:
                logging.info(f"Epub2txt: Successfully extracted {len(extracted_text)} chars from '{file_basename}'.")
            else:
                logging.warning(f"Epub2txt: Extracted no text from '{file_basename}'.")
            
            if progress_callback: progress_callback(1)
            return extracted_text

        except Exception as e:
            logging.warning(f"Epub2txt: Extraction failed for '{file_basename}': {str(e)}", exc_info=self._debug)
            if progress_callback: progress_callback(1)
            return ""
            
    def extract_with_calibre(self, epub_path: str, progress_callback: Optional[Callable] = None) -> str:
        file_basename = os.path.basename(epub_path)
        if self._debug: logging.debug(f"EPUBExtractor: Attempting Calibre for '{file_basename}'")
        
        calibre_bin = self._binary_paths.get('ebook-converter')
        if not calibre_bin or not os.path.exists(calibre_bin):
            calibre_bin = shutil.which('ebook-converter') or shutil.which('ebook-convert')
        
        if not calibre_bin:
            logging.warning(f"Calibre ebook-converter not found for EPUB '{file_basename}'.")
            if progress_callback: progress_callback(1)
            return ""
            
        temp_output_file = None
        try:
            # Use mkstemp for a unique temporary file name
            fd, temp_output_file_path = tempfile.mkstemp(suffix='.txt')
            os.close(fd) # Close the file descriptor, we only need the name
            
            cmd = [calibre_bin, epub_path, temp_output_file_path, "--input-encoding=utf-8", "--output-encoding=utf-8"]
            if self._debug: logging.debug(f"Running Calibre command for EPUB '{file_basename}': {' '.join(map(escape_special_chars, cmd))}") # escape for logging
            
            # Use run_process utility
            result = run_process(cmd, timeout=180) # 3 min timeout for calibre
            
            if result.returncode != 0:
                logging.warning(f"Calibre conversion for EPUB '{file_basename}' failed (code {result.returncode}). Stderr: {result.stderr.strip()[:500]}")
                return "" 
            
            extracted_text = ""
            if os.path.exists(temp_output_file_path):
                with open(temp_output_file_path, 'r', encoding='utf-8', errors='replace') as f:
                    extracted_text = f.read().strip()
                if extracted_text:
                    logging.info(f"Calibre: Successfully extracted {len(extracted_text)} chars from '{file_basename}'")
                else:
                    logging.warning(f"Calibre: Extracted empty text from '{file_basename}'")
            else:
                logging.warning(f"Calibre output file was not created: {temp_output_file_path}")

            if progress_callback: progress_callback(1)
            return extracted_text

        except Exception as e:
            logging.error(f"Exception during Calibre EPUB extraction for '{file_basename}': {str(e)}", exc_info=self._debug)
            return ""
        finally:
            if temp_output_file_path and os.path.exists(temp_output_file_path):
                try: os.unlink(temp_output_file_path)
                except Exception as e_unlink: logging.debug(f"Error unlinking temp Calibre output {temp_output_file_path}: {e_unlink}")

    def extract_with_zipfile(self, epub_path: str, progress_callback: Optional[Callable] = None) -> str:
        file_basename = os.path.basename(epub_path)
        logging.info(f"Zipfile: Starting raw extraction for '{file_basename}'. This is a last resort.")
        zipfile_module = self._get_module_or_log('zipfile', method_context='Zipfile', file_basename=file_basename)
        if not zipfile_module:
            if progress_callback: progress_callback(1)
            return ""

        text_parts: List[str] = []
        script_style_pattern = re.compile(r'<(script|style)\b[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE)
        tag_pattern = re.compile(r'<[^>]+>')
        
        processed_item_count = 0
        content_files = []
        try:
            with zipfile_module.ZipFile(epub_path) as zf:
                try: # Try to get ordered content from OPF
                    opf_file_name = next(f for f in zf.namelist() if f.lower().endswith('.opf') and 'meta-inf' not in f.lower())
                    opf_content = zf.read(opf_file_name).decode('utf-8', errors='replace')
                    spine_item_ids = re.findall(r'<itemref\s+idref="([^"]+)"', opf_content, re.IGNORECASE)
                    manifest_items = {m.group(1): m.group(2) for m in re.finditer(r'<item\s+id="([^"]+)"[^>]+href="([^"]+)"', opf_content, re.IGNORECASE)}
                    opf_dir = os.path.dirname(opf_file_name)
                    for item_id in spine_item_ids:
                        href = manifest_items.get(item_id)
                        if href:
                            full_href_path = os.path.normpath(os.path.join(opf_dir, href))
                            if full_href_path in zf.namelist(): # Ensure it's actually in the zip
                                content_files.append(full_href_path)
                    if self._debug and content_files: logging.debug(f"Zipfile: Prioritizing {len(content_files)} files from OPF spine.")
                except StopIteration:
                    logging.debug(f"Zipfile: No OPF file found in '{file_basename}'.")
                except Exception as e_opf:
                    logging.debug(f"Zipfile: Error parsing OPF for '{file_basename}': {e_opf}")

                if not content_files: # Fallback if OPF parsing fails or no spine
                    content_files = [f for f in zf.namelist() if f.lower().endswith(('.html', '.xhtml', '.htm', '.txt')) and 
                                                              not f.lower().startswith(('meta-inf/', 'mimetype'))]

                if not content_files:
                    logging.warning(f"Zipfile: No suitable content files found in EPUB '{file_basename}'.")
                    if progress_callback: progress_callback(1)
                    return ""
                
                with tqdm(total=len(content_files), desc=f"Zip items ({file_basename})", unit="item", leave=False, position=2) as pbar_zip:
                    for item_idx, item_name in enumerate(content_files):
                        if shutdown_flag.is_set(): break
                        pbar_zip.set_description(f"Zip item {item_idx+1}/{len(content_files)}")
                        try:
                            content_bytes = zf.read(item_name)
                            content = content_bytes.decode('utf-8', errors='replace')
                            
                            if item_name.lower().endswith(('.html', '.xhtml', '.htm')):
                                cleaned_content = script_style_pattern.sub(' ', content) # Replace with space
                                cleaned_content = tag_pattern.sub(' ', cleaned_content) # Replace tags with space
                            else: # For .txt files
                                cleaned_content = content
                            
                            cleaned_content = cleaned_content.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                            item_text = re.sub(r'\s+', ' ', cleaned_content).strip()
                            
                            if item_text:
                                text_parts.append(item_text)
                            processed_item_count += 1
                        except Exception as e_item:
                            logging.debug(f"Zipfile: Error processing item '{item_name}' in '{file_basename}': {str(e_item)[:100]}", exc_info=self._debug)
                        finally:
                            pbar_zip.update(1)
                            if progress_callback: progress_callback(1)
        except zipfile_module.BadZipFile:
            logging.warning(f"Zipfile: '{file_basename}' is not a valid zip file or is corrupted.")
            if progress_callback: progress_callback(len(content_files) if content_files else 1)
            return ""
        except Exception as e_main:
            logging.warning(f"Zipfile: Main error processing '{file_basename}': {str(e_main)[:100]}", exc_info=self._debug)
            if progress_callback: progress_callback(len(content_files) if content_files else 1)
            return ""

        final_text = '\n\n'.join(filter(None, text_parts))
        logging.info(f"Zipfile: Finished for '{file_basename}', processed {processed_item_count}/{len(content_files)} items, extracted {len(final_text)} total chars.")
        return final_text