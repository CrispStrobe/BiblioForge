# extractors/pptx_extractor.py

import os
import logging
import shutil
import tempfile
from typing import Optional, List, Dict, Any, Callable

from tqdm import tqdm

# Direct import for utilities, assuming 'utils.py' is in PYTHONPATH
try:
    from utils import ImportCache, run_process, shutdown_flag, escape_special_chars
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py in pptx_extractor.py: {e}")
    raise

class PPTXExtractor:
    """PowerPoint (.pptx) text extraction with multiple fallback methods."""

    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods: Optional[Dict[str, bool]] = None
        self._binary_paths: Dict[str, str] = binary_paths if binary_paths is not None else {}

    def _check_calibre_available(self) -> bool:
        """Checks if Calibre's ebook-converter is available."""
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

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Check for available extraction methods for PPTX files."""
        if self._available_methods is None:
            self._available_methods = {
                'pptx': self._import_cache.is_available('pptx'),
                'calibre': self._check_calibre_available(),
            }
            if self._debug:
                logging.debug(f"PPTXExtractor available_methods: {self._available_methods}")
        return self._available_methods

    def extract_text(self, pptx_path: str, preferred_method: Optional[str] = None,
                       progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """
        Extracts text from a .pptx file using the best available method.
        """
        methods = ['pptx', 'calibre']
        file_basename = os.path.basename(pptx_path)

        if preferred_method and preferred_method in methods:
            methods.remove(preferred_method)
            methods.insert(0, preferred_method)

        extracted_text = ""
        with tqdm(total=len(methods), desc=f"PPTX Methods ({file_basename})", unit="mthd", leave=False, position=1) as method_pbar:
            for method_name in methods:
                if shutdown_flag.is_set():
                    break
                method_pbar.set_description(f"PPTX Method [{method_name}] ({file_basename})")

                if not self.available_methods.get(method_name):
                    method_pbar.update(1)
                    continue

                try:
                    if progress_callback: progress_callback(0, f"pptx_try_{method_name}")
                    extraction_func = getattr(self, f'extract_with_{method_name}')
                    
                    internal_cb = None
                    if progress_callback:
                        internal_cb = lambda num_items=1, step_desc=method_name: progress_callback(num_items, f"pptx_{step_desc}_item")

                    current_method_text = extraction_func(pptx_path, internal_cb)

                    if current_method_text and current_method_text.strip():
                        extracted_text = current_method_text.strip()
                        logging.info(f"SUCCESS: Extracted {len(extracted_text)} chars from '{file_basename}' using: {method_name}")
                        if progress_callback: progress_callback(100, f"pptx_done_{method_name}")
                        method_pbar.update(1)
                        break
                    else:
                        if progress_callback: progress_callback(0, f"pptx_empty_{method_name}")

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logging.warning(f"Error with PPTX method '{method_name}' for '{file_basename}': {e}", exc_info=self._debug)
                    if progress_callback: progress_callback(0, f"pptx_err_{method_name}")

                method_pbar.update(1)

        if not extracted_text:
            logging.warning(f"PPTXExtractor: No text extracted from '{file_basename}'.")
        return extracted_text

    def extract_with_pptx(self, pptx_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Extract text using the python-pptx library."""
        Presentation = self._import_cache.import_module('pptx', 'Presentation')
        if not Presentation:
            return ""

        text_parts = []
        try:
            prs = Presentation(pptx_path)
            for i, slide in enumerate(prs.slides):
                if shutdown_flag.is_set(): break
                
                # Sort shapes by top position to get a logical reading order
                shapes = sorted([s for s in slide.shapes if s.has_text_frame], key=lambda s: s.top)
                
                for shape in shapes:
                    if shape.has_text_frame:
                        # Extract text paragraph by paragraph to preserve internal line breaks
                        paragraph_texts = [p.text for p in shape.text_frame.paragraphs if p.text]
                        if paragraph_texts:
                            text_parts.append("\n".join(paragraph_texts))

                # Extract speaker notes
                if slide.has_notes_slide:
                    notes_text = slide.notes_slide.notes_text_frame.text
                    if notes_text and notes_text.strip():
                        text_parts.append("\n--- Speaker Notes ---\n" + notes_text)

                if progress_callback: progress_callback(1)
            
            return "\n\n".join(filter(None, text_parts))
        except Exception as e:
            logging.debug(f"python-pptx extraction failed for {pptx_path}: {e}", exc_info=self._debug)
            return ""

    def extract_with_calibre(self, pptx_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Extract text from .pptx using Calibre's ebook-converter."""
        calibre_bin = self._binary_paths.get('ebook-converter') or shutil.which('ebook-converter') or shutil.which('ebook-convert')
        if not calibre_bin:
            logging.warning(f"Calibre (ebook-converter) not found for PPTXExtractor on {pptx_path}.")
            return ""
        
        temp_output_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                temp_output_file = tmp.name
            
            cmd = [calibre_bin, pptx_path, temp_output_file]
            if self._debug: logging.debug(f"Running Calibre for PPTX conversion: {' '.join(map(escape_special_chars, cmd))}")
            
            result = run_process(cmd, timeout=180) # 3 min timeout

            if result.returncode != 0:
                logging.warning(f"Calibre PPTX conversion failed (code {result.returncode}). Stderr: {result.stderr.strip()[:200]}")
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
            logging.error(f"Exception during Calibre PPTX extraction for {pptx_path}: {e}", exc_info=self._debug)
            return ""
        finally:
            if temp_output_file and os.path.exists(temp_output_file):
                try: os.unlink(temp_output_file)
                except: pass