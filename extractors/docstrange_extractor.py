# extractors/docstrange_extractor.py
import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple
from .pdf_extractor import PDFExtractor
import json
import re

try:
    from utils import ImportCache, shutdown_flag, validate_and_fix_year
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py: {e}")
    raise


# --- MONKEYPATCH for docstrange GPU detection on Apple Silicon ---
try:
    import torch
    import docstrange.utils.gpu_utils
    
    patch_logger = logging.getLogger(__name__)

    def _patched_is_gpu_available() -> bool:
        """Patched version to check for both CUDA and Apple MPS."""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                patch_logger.info(f"Patched Check: GPU detected (CUDA): {gpu_name}")
                return True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                patch_logger.info("Patched Check: Apple MPS GPU detected")
                return True
            else:
                patch_logger.info("Patched Check: No CUDA or MPS GPU available via torch")
                return False
        except Exception as e:
            patch_logger.warning(f"Error during patched GPU check: {e}")
            return False

    docstrange.utils.gpu_utils.is_gpu_available = _patched_is_gpu_available
    logging.info("Applied monkeypatch to docstrange for Apple Silicon (MPS) GPU detection.")

except (ImportError, AttributeError) as e:
    logging.debug(f"Could not apply docstrange monkeypatch: {e}")
# --- END MONKEYPATCH ---


class DocStrangeExtractor:
    """
    DocStrange/Nanonets extractor with support for:
    - Cloud API (10k docs/month with authentication)
    - Local GPU processing (100% private)
    - Local CPU processing (100% private)
    
    Features:
    - Advanced OCR and layout detection
    - Table extraction to Markdown/HTML/CSV
    - Structured data extraction with schemas
    - LaTeX equation recognition
    - Multi-format support (PDF, images, DOCX, etc.)
    """
    
    OUTPUT_FORMATS = ['markdown', 'json', 'html', 'csv', 'text', 'markdown-financial-docs']
    
    def __init__(self, import_cache: ImportCache, debug: bool = False,
                 mode: str = 'cloud',
                 api_key: Optional[str] = None,
                 binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._mode = mode
        self._api_key = api_key
        self._binary_paths = binary_paths or {}
        
        self._docstrange_module = None
        self._extractor_instance = None
        self._extractor_actual_mode = None  # Track what mode the instance is actually in
        
        self._available_methods: Optional[Dict[str, bool]] = None
        self._is_authenticated = False
        
        if self._import_cache.is_available('docstrange'):
            try:
                self._docstrange_module = self._import_cache.import_module('docstrange')
                if self._debug:
                    logging.debug(f"DocStrange module loaded successfully (mode: {self._mode})")
                
                if self._mode == 'cloud':
                    self._check_authentication()
            except Exception as e:
                logging.warning(f"Failed to load DocStrange module: {e}")
        else:
            logging.error("DocStrange not installed. Install with: pip install docstrange")

    def _extract_text_with_fallback(self, file_path: str) -> str:
        """
        Fallback to PDFExtractor when DocStrange returns empty content.
        This mirrors the ebook-converter fallback in the simple script.
        """
        try:
            logging.info("Attempting PDFExtractor fallback for empty DocStrange result...")
            
            # Create a PDFExtractor instance
            pdf_extractor = PDFExtractor(
                import_cache=self._import_cache,
                debug=self._debug,
                binary_paths=self._binary_paths
            )
            
            # Try core PDF extraction methods (no OCR initially)
            extracted_text = pdf_extractor.extract_text(
                pdf_path=file_path,
                preferred_method='pymupdf',  # Start with fastest method
                ocr_method=None,
                force_ocr=False,
                extract_tables=False
            )
            
            if extracted_text and len(extracted_text.strip()) > 50:
                return extracted_text
            
            # If core methods failed, try with OCR
            logging.info("Core PDF extraction returned little/no text, trying OCR...")
            extracted_text = pdf_extractor.extract_text(
                pdf_path=file_path,
                preferred_method=None,
                ocr_method='auto',
                force_ocr=True,
                extract_tables=False
            )
            
            return extracted_text
            
        except Exception as e:
            logging.error(f"PDFExtractor fallback failed: {e}", exc_info=self._debug)
            return ""
    
    @property
    def available_methods(self) -> Dict[str, bool]:
        if self._available_methods is None:
            self._available_methods = {
                'docstrange_cloud': self._check_cloud_available(),
                'docstrange_local_gpu': self._check_local_gpu_available(),
                'docstrange_local_cpu': self._check_local_cpu_available(),
            }
            if self._debug:
                logging.debug(f"DocStrange available methods: {self._available_methods}")
        return self._available_methods
    
    def _check_authentication(self):
        """Check if user is authenticated (after 'docstrange login')"""
        try:
            creds_path = Path.home() / ".docstrange" / "credentials"
            self._is_authenticated = creds_path.exists()
            
            if self._is_authenticated:
                logging.info("DocStrange: Using authenticated session (10k docs/month)")
            elif self._api_key:
                logging.info("DocStrange: Using API key (10k docs/month)")
            else:
                logging.info("DocStrange: Using rate-limited free tier")
                logging.info("  Run 'docstrange login' or provide API key for 10k docs/month")
        except Exception as e:
            if self._debug:
                logging.debug(f"Could not check authentication: {e}")
    
    def _check_cloud_available(self) -> bool:
        """Check if cloud API is available"""
        return self._docstrange_module is not None
    
    def _check_local_gpu_available(self) -> bool:
        """Check if local GPU processing is available, including Apple MPS."""
        if not self._docstrange_module:
            return False
        
        try:
            torch_module = self._import_cache.import_module('torch')
            if torch_module:
                if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                    if self._debug:
                        logging.debug("NVIDIA CUDA GPU detected.")
                    return True
                if hasattr(torch_module.backends, 'mps') and torch_module.backends.mps.is_available():
                    if self._debug:
                        logging.debug("Apple MPS GPU detected.")
                    return True
        except Exception as e:
            if self._debug:
                logging.debug(f"GPU check failed: {e}")
        return False
    
    def _check_local_cpu_available(self) -> bool:
        """Check if local CPU processing is available"""
        return self._docstrange_module is not None
    
    def _init_extractor(self) -> bool:
        """Initialize DocStrange extractor with automatic fallback on errors."""
        # Avoid re-initialization if already done correctly
        if self._extractor_instance is not None and self._extractor_actual_mode == self._mode:
            return True
        
        if not self._docstrange_module:
            logging.error("DocStrange module not available")
            return False
            
        try:
            from docstrange import DocumentExtractor
            
            # Attempt to initialize in the requested mode, falling back to cloud on any error
            if self._mode == 'local_gpu':
                try:
                    self._extractor_instance = DocumentExtractor(gpu=True)
                    self._extractor_actual_mode = 'local_gpu'
                    logging.info("🔒 DocStrange: LOCAL GPU mode (100% Private)")
                    return True
                except Exception as e:
                    logging.warning(f"⚠️  Local GPU mode failed: {e}. Falling back to CLOUD mode.")
                    self._mode = 'cloud' # Force fallback to the next step

            if self._mode == 'local_cpu':
                 try:
                    self._extractor_instance = DocumentExtractor()
                    self._extractor_actual_mode = 'local_cpu'
                    logging.info("🔒 DocStrange: LOCAL CPU mode (100% Private)")
                    return True
                 except Exception as e:
                    logging.warning(f"⚠️  Local CPU mode failed: {e}. Falling back to CLOUD mode.")
                    self._mode = 'cloud' # Force fallback to the next step
            
            # Default or fallback to cloud mode
            self._extractor_instance = DocumentExtractor(api_key=self._api_key)
            self._extractor_actual_mode = 'cloud'
            logging.info("☁️  DocStrange: CLOUD mode")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize any DocStrange extractor: {e}", exc_info=self._debug)
            self._extractor_instance = None
            return False
    
    def extract_text(self, file_path: str, **kwargs) -> str:
        """Extracts plain text, with robust fallbacks."""
        if not self._init_extractor():
            return ""

        extracted_text = ""
        try:
            # First, try the primary docstrange method which handles its own local->cloud fallback
            result = self._extractor_instance.extract(file_path)
            extracted_text = result.extract_markdown()
        except Exception as e:
            if self._debug:
                logging.warning(f"Initial DocStrange text extraction failed: {e}")
        
        # If the primary method returned nothing, use the reliable local fallback
        if not extracted_text or not extracted_text.strip():
            if self._debug:
                logging.debug("DocStrange returned no text, attempting local text extraction fallback.")
            return self._get_fallback_text(file_path)
        
        return extracted_text

    # --- HELPER METHOD ---
    def _extract_text_with_fallback(self, file_path: str) -> str:
        """
        Fallback to PDFExtractor when DocStrange returns empty content for a PDF.
        """
        try:
            from .pdf_extractor import PDFExtractor
            logging.info("Attempting PDFExtractor fallback for empty DocStrange result...")
            
            pdf_extractor = PDFExtractor(
                import_cache=self._import_cache,
                debug=self._debug,
                binary_paths=self._binary_paths
            )
            
            # Use a robust method like PyMuPDF first
            extracted_text = pdf_extractor.extract_text(
                pdf_path=file_path,
                preferred_method='pymupdf',
                force_ocr=False
            )
            
            return extracted_text
        except Exception as e:
            logging.error(f"PDFExtractor fallback failed: {e}", exc_info=self._debug)
            return ""
    
    def _get_formatted_output(self, result, output_format: str, **kwargs) -> str:
        """
        Get output in the requested format - FIXED to match working simple script pattern
        With added debugging for empty responses
        """
        try:
            if output_format == 'markdown' or output_format == 'markdown-financial-docs':
                # SIMPLE: Just call extract_markdown() like the working script
                markdown = result.extract_markdown()
                
                # DEBUG: Check if we got empty response
                if not markdown or len(markdown.strip()) == 0:
                    if self._debug:
                        logging.debug("extract_markdown() returned empty content")
                        logging.debug(f"Result type: {type(result)}")
                        logging.debug(f"Result class: {result.__class__.__name__}")
                        
                        # Check if result has any attributes we can inspect
                        if hasattr(result, '__dict__'):
                            logging.debug(f"Result attributes: {list(result.__dict__.keys())}")
                        
                        # For CloudConversionResult, check the raw response
                        if hasattr(result, '_api_response'):
                            logging.debug(f"Raw API response: {result._api_response}")
                        
                        # Check cloud_processor for more details
                        if hasattr(result, 'cloud_processor'):
                            cp = result.cloud_processor
                            if hasattr(cp, '_last_response'):
                                resp = cp._last_response
                                logging.debug(f"Last HTTP response status: {resp.status_code if hasattr(resp, 'status_code') else 'unknown'}")
                                logging.debug(f"Last HTTP response text: {resp.text if hasattr(resp, 'text') else 'N/A'}")
                            if hasattr(cp, 'response_data'):
                                logging.debug(f"Response data: {cp.response_data}")
                            if hasattr(cp, '_response_json'):
                                logging.debug(f"Response JSON: {cp._response_json}")
                        
                        # Check _cached_outputs
                        if hasattr(result, '_cached_outputs'):
                            logging.debug(f"Cached outputs keys: {list(result._cached_outputs.keys())}")
                            for key, val in result._cached_outputs.items():
                                logging.debug(f"  {key}: {len(str(val)) if val else 0} chars")
                        
                        # Check if there's a content attribute
                        if hasattr(result, 'content'):
                            logging.debug(f"Result.content: {result.content}")
                        
                        # Check extraction status
                        if hasattr(result, 'status'):
                            logging.debug(f"Result.status: {result.status}")
                    
                    logging.warning("DocStrange returned empty content - possible causes:")
                    logging.warning("  1. File might be scanned/image-only PDF requiring OCR")
                    logging.warning("  2. API rate limit reached")
                    logging.warning("  3. File format not fully supported")
                    logging.warning("  4. Authentication issue")
                    logging.warning("")
                    logging.warning("💡 Suggestion: Try a different extractor:")
                    logging.warning("   --extractor-method pymupdf4llm")
                    logging.warning("   --extractor-method docling")
                
                return markdown
            
            elif output_format == 'json':
                # SIMPLE: Just call extract_data() with the right params
                specified_fields = kwargs.get('specified_fields')
                json_schema = kwargs.get('json_schema')
                
                raw_result = result.extract_data(
                    specified_fields=specified_fields,
                    json_schema=json_schema
                )
                
                # Parse response format like the working script
                extracted_fields = self._parse_api_response(raw_result)
                return json.dumps(extracted_fields, indent=2, ensure_ascii=False)
            
            elif output_format == 'html':
                if hasattr(result, 'extract_html'):
                    return result.extract_html()
                # Fallback to markdown
                return result.extract_markdown()
            
            elif output_format == 'csv':
                if hasattr(result, 'extract_csv'):
                    return result.extract_csv()
                # Fallback to markdown
                return result.extract_markdown()
            
            elif output_format == 'text':
                if hasattr(result, 'extract_text'):
                    return result.extract_text()
                # Fallback to markdown
                return result.extract_markdown()
            
            # Default fallback
            return result.extract_markdown()
            
        except Exception as e:
            logging.error(f"Format conversion failed: {e}", exc_info=self._debug)
            return ""
    
    
    
    def extract_metadata(self, file_path: str,
                         schema: Optional[Dict] = None,
                         fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extracts structured metadata using DocStrange, with a robust fallback loop
        that mirrors the successful test script's logic.
        """
        if not self._init_extractor():
            return {}

        file_basename = os.path.basename(file_path)
        
        if not fields and not schema:
            fields = ['title', 'author', 'authors', 'year', 'publication_date', 'publisher', 'language']

        # --- Attempt 1: Process the original file directly ---
        try:
            if self._debug:
                logging.debug(f"DocStrange metadata: Attempting direct extraction on '{file_basename}'")
            
            result_obj = self._extractor_instance.extract(file_path)
            raw_metadata = result_obj.extract_data(specified_fields=fields, json_schema=schema)
            parsed_data = self._parse_api_response(raw_metadata)

            if parsed_data and parsed_data.get('title') and (parsed_data.get('author') or parsed_data.get('authors')):
                logging.info(f"✓ DocStrange successfully extracted metadata directly from '{file_basename}'")
                return self._format_metadata_for_biblioforge(parsed_data)

            if self._debug:
                logging.debug(f"Direct DocStrange metadata extraction yielded incomplete data for '{file_basename}'")

        except Exception as e:
            if self._debug:
                logging.warning(f"Direct DocStrange metadata extraction failed for '{file_basename}': {e}")

        # --- Attempt 2: Fallback to text extraction, then send text to DocStrange ---
        logging.warning(f"⚠️  Direct metadata extraction failed for '{file_basename}'. Attempting fallback...")

        temp_file_path = None
        try:
            text_content = self._get_fallback_text(file_path)
            if not text_content or not text_content.strip():
                logging.error(f"Fallback text extraction also failed for '{file_basename}'. Cannot proceed.")
                return {}

            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(text_content)
                temp_file_path = tmp.name

            if self._debug:
                logging.debug(f"Sending temporary text file to DocStrange for metadata analysis...")
            
            # Ensure we have a cloud extractor for text file processing
            if self._extractor_actual_mode != 'cloud':
                logging.info("Switching to cloud extractor for fallback text processing.")
                self._extractor_instance = self._docstrange_module.DocumentExtractor(api_key=self._api_key)

            result_obj = self._extractor_instance.extract(temp_file_path)
            raw_metadata = result_obj.extract_data(specified_fields=fields, json_schema=schema)
            parsed_data = self._parse_api_response(raw_metadata)

            if parsed_data and parsed_data.get('title') and (parsed_data.get('author') or parsed_data.get('authors')):
                logging.info(f"✓ Fallback successful: DocStrange extracted metadata from text for '{file_basename}'")
                return self._format_metadata_for_biblioforge(parsed_data)
            else:
                logging.error(f"DocStrange failed to extract metadata even from fallback text for '{file_basename}'.")

        except Exception as e:
            logging.error(f"The DocStrange fallback process failed for '{file_basename}': {e}", exc_info=self._debug)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        return {}
    
    
    
    def _get_fallback_text(self, file_path: str) -> str:
        """Uses a reliable local extractor to get plain text as a fallback."""
        try:
            # Use specific, reliable extractors for fallback
            if file_path.lower().endswith('.pdf'):
                from .pdf_extractor import PDFExtractor
                extractor = PDFExtractor(self._import_cache, self._debug, self._binary_paths)
                # PyMuPDF is generally the most reliable for text layer extraction
                return extractor.extract_text(file_path, preferred_method='pymupdf')
            else: # For .epub, .mobi, .docx etc., Calibre is the best bet
                from .text_extractor import TextExtractor
                extractor = TextExtractor(self._import_cache, self._debug, self._binary_paths)
                return extractor.extract_text(file_path, preferred_method='calibre')
        except Exception as e:
            if self._debug:
                logging.error(f"Helper _get_fallback_text failed for '{file_path}': {e}", exc_info=True)
            return ""

    def _parse_api_response(self, raw_result: Dict) -> Dict:
        """
        Robustly parse various API response formats, including messy raw_content,
        inspired by the successful test script.
        """
        extracted_fields = {}
        
        # Case 1: Clean, direct response
        if 'extracted_fields' in raw_result and raw_result['extracted_fields']:
            return raw_result['extracted_fields']
        
        # Case 2: Messy response with JSON embedded in a raw string
        if 'document' in raw_result and 'raw_content' in raw_result['document']:
            raw = raw_result['document']['raw_content']
            # Regex to find all JSON-like objects in the messy string
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, raw)
            
            for match in matches:
                try:
                    data = json.loads(match)
                    # Update fields, but don't overwrite with empty values from later matches
                    for key, value in data.items():
                        if value and (key not in extracted_fields or not extracted_fields[key]):
                            extracted_fields[key] = value
                except json.JSONDecodeError:
                    continue # Ignore parts that look like JSON but aren't valid
        
        return extracted_fields

    def _format_metadata_for_biblioforge(self, fields_data: Dict) -> Dict:
        """Takes raw parsed data and converts it to the standard BiblioForge format."""
        title = str(fields_data.get('title', '')).strip()
        
        author = str(fields_data.get('author', '')).strip()
        if not author and fields_data.get('authors'):
            authors_data = fields_data['authors']
            if isinstance(authors_data, list) and authors_data:
                author = str(authors_data[0]).strip()
            elif isinstance(authors_data, str):
                author = re.split(r'\s*[,;]\s*', authors_data)[0].strip()

        year = fields_data.get('year', '')
        if not year:
            pub_date = fields_data.get('publication_date', '')
            year_match = re.search(r'\b(19|20)\d{2}\b', str(pub_date))
            if year_match:
                year = year_match.group(0)
        
        year = validate_and_fix_year(str(year))

        language = str(fields_data.get('language', 'ul')).strip().lower()
        if len(language) > 2:
            lang_map = {'english': 'en', 'german': 'de', 'french': 'fr', 'spanish': 'es', 'italian': 'it'}
            language = lang_map.get(language, language[:2])
        language = language[:2] if len(language) >= 2 else 'ul'

        if not title or not author:
            return {} # Return empty if core fields are missing

        return {
            'title': title,
            'author': author,
            'year': year,
            'language': language
        }
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current processing mode"""
        privacy_mode = 'LOCAL (100% Private)' if self._mode in ['local_gpu', 'local_cpu'] else 'CLOUD (API)'
        
        return {
            'mode': self._mode,
            'is_local': self._mode in ['local_gpu', 'local_cpu'],
            'is_cloud': self._mode == 'cloud',
            'has_api_key': bool(self._api_key),
            'is_authenticated': self._is_authenticated,
            'gpu_available': self.available_methods.get('docstrange_local_gpu', False),
            'privacy_mode': privacy_mode,
            'rate_limit': '10k docs/month' if (self._api_key or self._is_authenticated) else 'Limited (free tier)'
        }
    
    @staticmethod
    def run_login() -> bool:
        """Run DocStrange authentication flow (docstrange login)"""
        try:
            logging.info("Starting DocStrange authentication...")
            logging.info("This will open a browser window for Google authentication")
            
            result = subprocess.run(
                ['docstrange', 'login'],
                check=True,
                capture_output=True,
                text=True
            )
            
            logging.info("✓ DocStrange authentication complete")
            logging.info("  You now have 10k docs/month free")
            return True
            
        except FileNotFoundError:
            logging.error("DocStrange CLI not found. Install with: pip install docstrange")
            return False
        except subprocess.CalledProcessError as e:
            logging.error(f"DocStrange login failed: {e.stderr}")
            return False
        except Exception as e:
            logging.error(f"DocStrange login error: {e}")
            return False
    
    @staticmethod
    def run_logout() -> bool:
        """Run DocStrange logout to clear cached credentials"""
        try:
            creds_path = Path.home() / ".docstrange" / "credentials"
            if creds_path.exists():
                creds_path.unlink()
            
            logging.info("✓ DocStrange logout complete")
            return True
            
        except Exception as e:
            logging.error(f"DocStrange logout error: {e}")
            return False