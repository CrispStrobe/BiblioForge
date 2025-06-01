# extractors/djvu_extractor.py:

import os
import logging
import shutil
import tempfile
import subprocess
import glob # For finding image files in OCR method
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

class DJVUExtractor:
    """DJVU text extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods: Optional[Dict[str, bool]] = None
        self._binary_paths: Dict[str, str] = binary_paths if binary_paths is not None else {}
        
        # Check for djvu library specifically during init
        djvu_type, djvu_path = self.find_djvu_lib()
        self._djvu_type = djvu_type # 'module', 'command', or None
        self._djvu_path = djvu_path # Path to module or command
        
        if self._debug and djvu_type:
            logging.debug(f"DJVUExtractor: Found djvu as {djvu_type} at: {djvu_path}")

    def find_djvu_lib(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find djvu Python bindings or command-line tools in various locations.
        Returns: tuple (type, path)
        """
        # First check if python-djvulibre Python bindings are available
        if self._import_cache.is_available('djvu'):
            try:
                djvu_module = self._import_cache.import_module('djvu')
                return 'module', getattr(djvu_module, '__file__', 'djvu_python_bindings')
            except Exception as e:
                if self._debug: logging.debug(f"Error importing djvu module though available: {e}")

        # Check for djvulibre command line tools (djvutxt for text, ddjvu for conversion)
        for binary_key in ['djvutxt', 'ddjvu']:
            path_from_binaries = self._binary_paths.get(binary_key)
            if path_from_binaries and os.path.exists(path_from_binaries):
                return 'command', path_from_binaries
            
            path_from_shutil = shutil.which(binary_key)
            if path_from_shutil:
                # Store it if found via shutil.which for future use within this instance
                if binary_key not in self._binary_paths or not self._binary_paths.get(binary_key):
                    self._binary_paths[binary_key] = path_from_shutil
                return 'command', path_from_shutil
        
        return None, None

    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            self._available_methods = {
                'djvulibre': self._check_djvulibre_method_available(), # Python bindings or djvutxt CLI
                'pdf_conversion': self._check_pdf_conversion_available(), # ddjvu CLI
                'ocr': self._check_ocr_dependencies_available() # ddjvu + tesseract + pdf2image
            }
            if self._debug:
                logging.debug(f"DJVUExtractor available_methods: {self._available_methods}")
        return self._available_methods

    def _check_djvulibre_method_available(self) -> bool:
        # This method is available if we found either the Python module or the 'djvutxt' command
        return self._djvu_type is not None and (self._djvu_type == 'module' or self._binary_paths.get('djvutxt') is not None)

    def _check_pdf_conversion_available(self) -> bool:
        # Requires 'ddjvu' command and PDFExtractor capabilities
        ddjvu_path = self._binary_paths.get('ddjvu') or shutil.which('ddjvu')
        if not ddjvu_path: return False
        # PDFExtractor itself will determine if it can process a PDF
        return True 

    def _check_ocr_dependencies_available(self) -> bool:
        # Requires 'ddjvu' (for image conversion), and Tesseract components
        ddjvu_path = self._binary_paths.get('ddjvu') or shutil.which('ddjvu')
        tesseract_path = self._binary_paths.get('tesseract') or shutil.which('tesseract')
        
        return bool(ddjvu_path and \
                    tesseract_path and \
                    self._import_cache.is_available('pdf2image') and # Though we use ddjvu, pdf2image implies PIL is there
                    self._import_cache.is_available('PIL', submodules=['Image']) and \
                    self._import_cache.is_available('pytesseract'))


    def extract_text(self, djvu_path: str, preferred_method: Optional[str] = None,
                     progress_callback: Optional[Callable] = None, **kwargs) -> str:
        # kwargs is for consistency, DJVUExtractor doesn't use them specifically for now
        methods = ['djvulibre', 'pdf_conversion', 'ocr'] # Order of preference
        file_basename = os.path.basename(djvu_path)

        if preferred_method and preferred_method in methods:
            methods.remove(preferred_method)
            methods.insert(0, preferred_method)
            logging.info(f"DJVU: Using preferred method '{preferred_method}'. Order: {methods}")
        else:
            logging.info(f"DJVU: Using default method order: {methods}")

        extracted_text = ""
        with tqdm(total=len(methods), desc=f"DJVU Methods ({file_basename})", unit="mthd", leave=False, position=1) as method_pbar:
            for method_name in methods:
                if shutdown_flag.is_set():
                    logging.info(f"Shutdown during DJVU methods for '{file_basename}'.")
                    break
                
                method_pbar.set_description(f"DJVU Method [{method_name}] ({file_basename})")
                if not self.available_methods.get(method_name):
                    logging.info(f"DJVU method '{method_name}' for '{file_basename}' not available. Skipping.")
                    method_pbar.update(1)
                    continue

                logging.info(f"Attempting DJVU '{file_basename}' with method: {method_name}")
                try:
                    if progress_callback: progress_callback(0, f"djvu_try_{method_name}")
                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    
                    internal_cb = None
                    if progress_callback:
                        internal_cb = lambda num_items=1, step_desc=method_name: progress_callback(num_items, f"djvu_{step_desc}_item")
                    
                    current_method_text = extraction_func(djvu_path, internal_cb)
                    
                    if current_method_text and current_method_text.strip():
                        extracted_text = current_method_text.strip()
                        logging.info(f"SUCCESS: Extracted {len(extracted_text)} chars from DJVU '{file_basename}' using: {method_name}")
                        if progress_callback: progress_callback(100, f"djvu_done_{method_name}")
                        method_pbar.update(1)
                        break
                    else:
                        logging.warning(f"DJVU method '{method_name}' returned NO TEXT for '{file_basename}'.")
                        if progress_callback: progress_callback(0, f"djvu_empty_{method_name}")
                
                except KeyboardInterrupt:
                    logging.info(f"DJVU extraction with '{method_name}' for '{file_basename}' interrupted.")
                    raise
                except Exception as e:
                    logging.warning(f"Error during DJVU extraction with method '{method_name}' for '{file_basename}': {e}", exc_info=self._debug)
                    if progress_callback: progress_callback(0, f"djvu_err_{method_name}")
                
                method_pbar.update(1)
        
        if not extracted_text:
            logging.warning(f"FINAL WARNING: No text extracted from DJVU '{file_basename}' after all attempts.")
        return extracted_text

    def extract_with_djvulibre(self, djvu_path: str, progress_callback: Optional[Callable] = None) -> str:
        # Try Python bindings first if available
        if self._djvu_type == 'module' and self._import_cache.is_available('djvu'):
            djvu_module = self._import_cache.import_module('djvu')
            if hasattr(djvu_module, 'DjVuDocument'):
                text_parts = []
                doc = None
                try:
                    doc = djvu_module.DjVuDocument.create_by_filename(djvu_path.encode('utf-8')) # Filename needs to be bytes
                    total_pages = doc.pages_number if hasattr(doc, 'pages_number') else len(doc.pages)

                    for i in range(total_pages):
                        if shutdown_flag.is_set(): break
                        page = doc.pages[i]
                        page_text_bytes = page.text.encode('utf-8', 'replace') # Access text, then encode
                        page_text = page_text_bytes.decode('utf-8', 'replace')
                        if page_text.strip(): text_parts.append(page_text.strip())
                        if progress_callback: progress_callback(1)
                    return "\n\n".join(text_parts)
                except Exception as e:
                    logging.debug(f"Python-djvulibre bindings extraction failed: {e}", exc_info=self._debug)
                    # Fall through to command-line if bindings fail
                finally:
                    if doc: del doc # Or doc.close() if available

        # Fallback to djvutxt command line tool
        djvutxt_bin = self._binary_paths.get('djvutxt') or shutil.which('djvutxt')
        if not djvutxt_bin:
            logging.warning("djvutxt binary not found.")
            return ""

        temp_output_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                temp_output_file = tmp.name
            
            cmd = [djvutxt_bin, djvu_path, temp_output_file]
            if self._debug: logging.debug(f"Running djvutxt: {' '.join(map(escape_special_chars, cmd))}")
            
            result = run_process(cmd, timeout=120) # 2 min timeout
            
            if result.returncode != 0:
                # djvutxt might output to stderr on success for some info, check stdout for actual error
                err_msg = result.stderr.strip()
                # Some versions of djvutxt print "Empty URL passed" for valid files but still work
                if "Empty URL passed" not in err_msg or not os.path.exists(temp_output_file) or os.path.getsize(temp_output_file) == 0:
                    raise RuntimeError(f"djvutxt failed (code {result.returncode}). Stderr: {err_msg[:200]}")
                elif self._debug:
                    logging.debug(f"djvutxt stderr (potentially ignorable): {err_msg[:200]}")

            if os.path.exists(temp_output_file):
                with open(temp_output_file, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read().strip()
                if progress_callback: progress_callback(1) # Simplified progress for CLI tool
                return text
            return ""
        except Exception as e:
            logging.debug(f"DjVuLibre command-line (djvutxt) extraction failed: {e}", exc_info=self._debug)
            return ""
        finally:
            if temp_output_file and os.path.exists(temp_output_file):
                try: os.unlink(temp_output_file)
                except: pass

    def extract_with_pdf_conversion(self, djvu_path: str, progress_callback: Optional[Callable] = None) -> str:
        ddjvu_bin = self._binary_paths.get('ddjvu') or shutil.which('ddjvu')
        if not ddjvu_bin:
            logging.warning("ddjvu binary not found for PDF conversion method.")
            return ""

        temp_pdf_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                temp_pdf_file = tmp.name
            
            cmd = [ddjvu_bin, '-format=pdf', djvu_path, temp_pdf_file]
            if self._debug: logging.debug(f"Running ddjvu: {' '.join(map(escape_special_chars, cmd))}")
            
            result = run_process(cmd, timeout=300) # 5 min timeout for conversion
            
            if result.returncode != 0:
                raise RuntimeError(f"DJVU to PDF conversion (ddjvu) failed (code {result.returncode}). Stderr: {result.stderr.strip()[:200]}")
            
            if not os.path.exists(temp_pdf_file) or os.path.getsize(temp_pdf_file) == 0:
                raise RuntimeError(f"ddjvu created an empty or missing PDF: {temp_pdf_file}")

            logging.info(f"Successfully converted DJVU to temporary PDF: {temp_pdf_file}")
            
            # Now extract text from the temporary PDF
            # Pass along self._binary_paths which PDFExtractor will use
            pdf_extractor = PDFExtractor(debug=self._debug, binary_paths=self._binary_paths)
            text = pdf_extractor.extract_text(temp_pdf_file, progress_callback=progress_callback) # Pass callback
            return text
            
        except Exception as e:
            logging.warning(f"DJVU to PDF conversion and extraction failed: {e}", exc_info=self._debug)
            return ""
        finally:
            if temp_pdf_file and os.path.exists(temp_pdf_file):
                try: os.unlink(temp_pdf_file)
                except: pass

    def extract_with_ocr(self, djvu_path: str, progress_callback: Optional[Callable] = None) -> str:
        ddjvu_bin = self._binary_paths.get('ddjvu') or shutil.which('ddjvu')
        tesseract_bin = self._binary_paths.get('tesseract') or shutil.which('tesseract')
        pytesseract_module = self._import_cache.is_available('pytesseract')
        pil_image_module = self._import_cache.is_available('PIL', submodules=['Image'])


        if not (ddjvu_bin and tesseract_bin and pytesseract_module and pil_image_module):
            logging.warning("OCR dependencies (ddjvu, tesseract, pytesseract, Pillow) not fully met for DJVU OCR.")
            return ""
        
        pytesseract = self._import_cache.import_module('pytesseract')
        Image = self._import_cache.import_module('PIL', 'Image')

        text_parts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Convert DJVU to images (e.g., TIFF or PNG)
                # ddjvu -format=tiff <input.djvu> <output_pattern.tif>
                # Example: ddjvu -format=tiff mydoc.djvu page-%d.tif
                output_pattern = os.path.join(temp_dir, 'page-%d.png') # Using png
                cmd_convert = [ddjvu_bin, '-format=png', djvu_path, output_pattern]
                
                if self._debug: logging.debug(f"Running ddjvu for image conversion: {' '.join(map(escape_special_chars, cmd_convert))}")
                result_convert = run_process(cmd_convert, timeout=300) # 5 min timeout
                
                if result_convert.returncode != 0:
                    raise RuntimeError(f"DJVU to image conversion failed: {result_convert.stderr.strip()[:200]}")

                image_files = sorted(glob.glob(os.path.join(temp_dir, 'page-*.png')))
                if not image_files:
                    raise RuntimeError("No images generated from DJVU for OCR.")

                with tqdm(total=len(image_files), desc="DJVU OCR (Tesseract)", unit="page", leave=False, position=2) as pbar_ocr:
                    for img_path in image_files:
                        if shutdown_flag.is_set(): break
                        try:
                            pil_img = Image.open(img_path)
                            page_text = pytesseract.image_to_string(pil_img) # Add lang if needed
                            if page_text.strip():
                                text_parts.append(page_text.strip())
                            pil_img.close()
                        except Exception as e_page_ocr:
                            logging.debug(f"Tesseract OCR failed for image {img_path}: {e_page_ocr}")
                        finally:
                            pbar_ocr.update(1)
                            if progress_callback: progress_callback(1)
                return "\n\n".join(text_parts)
            except Exception as e:
                logging.warning(f"DJVU OCR extraction process failed: {e}", exc_info=self._debug)
                return ""