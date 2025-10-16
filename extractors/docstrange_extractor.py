# extractors/docstrange_extractor.py
import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Union
from tqdm import tqdm

try:
    from utils import ImportCache, shutdown_flag
except ImportError as e:
    logging.error(f"CRITICAL: Failed to import from utils.py: {e}")
    raise


# --- MONKEYPATCH for docstrange GPU detection on Apple Silicon ---
try:
    import torch
    import docstrange.utils.gpu_utils
    
    # Get a logger instance for our patch
    patch_logger = logging.getLogger(__name__)

    # 1. Define our corrected GPU check function
    def _patched_is_gpu_available() -> bool:
        """Patched version to check for both CUDA and Apple MPS."""
        try:
            # Check for NVIDIA CUDA
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                patch_logger.info(f"Patched Check: GPU detected (CUDA): {gpu_name}")
                return True
            # Check for Apple Silicon MPS
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                patch_logger.info("Patched Check: Apple MPS GPU detected")
                return True
            else:
                patch_logger.info("Patched Check: No CUDA or MPS GPU available via torch")
                return False
        except Exception as e:
            patch_logger.warning(f"Error during patched GPU check: {e}")
            return False

    # 2. Apply the patch by replacing the original function with our version
    docstrange.utils.gpu_utils.is_gpu_available = _patched_is_gpu_available
    logging.info("Applied monkeypatch to docstrange for Apple Silicon (MPS) GPU detection.")

except (ImportError, AttributeError) as e:
    logging.debug(f"Could not apply docstrange monkeypatch (docstrange or torch likely not installed): {e}")
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
    
    # Output formats supported
    OUTPUT_FORMATS = ['markdown', 'json', 'html', 'csv', 'text', 'markdown-financial-docs']
    
    def __init__(self, import_cache: ImportCache, debug: bool = False,
                 mode: str = 'cloud',
                 api_key: Optional[str] = None,
                 binary_paths: Optional[Dict[str, str]] = None):
        self._import_cache = import_cache
        self._debug = debug
        self._mode = mode  # 'cloud', 'local_gpu', 'local_cpu'
        self._api_key = api_key
        self._binary_paths = binary_paths or {}
        
        self._docstrange_module = None
        self._extractor_instance = None
        self._available_methods: Optional[Dict[str, bool]] = None
        self._is_authenticated = False
        
        # Try to import docstrange
        if self._import_cache.is_available('docstrange'):
            try:
                self._docstrange_module = self._import_cache.import_module('docstrange')
                if self._debug:
                    logging.debug(f"DocStrange module loaded successfully (mode: {self._mode})")
                
                # Check authentication status for cloud mode
                if self._mode == 'cloud':
                    self._check_authentication()
            except Exception as e:
                logging.warning(f"Failed to load DocStrange module: {e}")
        else:
            logging.error("DocStrange not installed. Install with: pip install docstrange")
    
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
            # DocStrange stores credentials after login
            # Check for credential file (implementation depends on DocStrange's auth mechanism)
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
        if not self._docstrange_module:
            return False
        
        # Cloud mode is always available if module is installed
        return True
    
    def _check_local_gpu_available(self) -> bool:
        """Check if local GPU processing is available, including Apple MPS."""
        if not self._docstrange_module:
            return False
        
        # Check for CUDA/GPU support
        try:
            torch_module = self._import_cache.import_module('torch')
            if torch_module:
                # Check for NVIDIA CUDA
                if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                    if self._debug: logging.debug("NVIDIA CUDA GPU detected.")
                    return True
                # Check for Apple Silicon MPS
                if hasattr(torch_module.backends, 'mps') and torch_module.backends.mps.is_available():
                    if self._debug: logging.debug("Apple MPS GPU detected.")
                    return True
        except Exception as e:
            if self._debug:
                logging.debug(f"GPU check failed: {e}")
        return False
    
    def _check_local_cpu_available(self) -> bool:
        """Check if local CPU processing is available"""
        return self._docstrange_module is not None
    
    def _init_extractor(self) -> bool:
        """Initialize DocStrange extractor."""
        if self._extractor_instance is not None:
            return True
        
        if not self._docstrange_module:
            logging.error("DocStrange module not available")
            return False
        
        try:
            from docstrange import DocumentExtractor
            
            if self._mode == 'cloud':
                if self._api_key:
                    self._extractor_instance = DocumentExtractor(api_key=self._api_key)
                else:
                    self._extractor_instance = DocumentExtractor()

            elif self._mode == 'local_gpu':
                if not self.available_methods['docstrange_local_gpu']:
                    logging.error("GPU not available for DocStrange local processing")
                    return False
                # This call now works because of our monkeypatch
                self._extractor_instance = DocumentExtractor(gpu=True)
                logging.info("🔒 DocStrange: LOCAL GPU mode (100% Private)")

            elif self._mode == 'local_cpu':
                # The library doesn't support a `cpu=True` flag.
                # To run on CPU, you'd typically initialize without `gpu=True`
                # and ensure torch doesn't find a GPU.
                # However, since you specified local_gpu, we focus on that path.
                logging.error("local_cpu mode is not directly supported by the DocumentExtractor constructor flags.")
                return False

            else:
                logging.error(f"Unknown DocStrange mode: {self._mode}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize DocStrange extractor: {e}", 
                          exc_info=self._debug)
            return False
    
    def extract_text(self, file_path: str,
                    preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None,
                    output_format: str = 'markdown',
                    **kwargs) -> str:
        """Extract text using DocStrange."""
        
        file_basename = os.path.basename(file_path)
        
        # Validate output format
        if output_format not in self.OUTPUT_FORMATS:
            logging.warning(f"Unknown output format: {output_format}, using markdown")
            output_format = 'markdown'
        
        if not self._init_extractor():
            logging.error(f"DocStrange extractor initialization failed for {file_basename}")
            return ""
        
        try:
            if self._debug:
                logging.debug(f"DocStrange processing {file_basename} in {self._mode} mode")
                logging.debug(f"  Output format: {output_format}")
            
            if progress_callback:
                progress_callback(1, f"docstrange_{self._mode}_extracting")
            
            # Extract document using DocStrange API
            result = self._extractor_instance.extract(file_path)
            
            # DEBUG: Check what we got back
            if self._debug:
                logging.debug(f"DocStrange result type: {type(result)}")
                logging.debug(f"DocStrange result has extract_markdown: {hasattr(result, 'extract_markdown')}")
                logging.debug(f"DocStrange result has extract_data: {hasattr(result, 'extract_data')}")
                logging.debug(f"DocStrange result dir: {[m for m in dir(result) if not m.startswith('_')]}")
                
                # Try to get ANY content to verify API worked
                if hasattr(result, 'extract_markdown'):
                    test_md = result.extract_markdown()
                    logging.debug(f"Test extract_markdown() returned {len(test_md)} chars")
                    # After the line: test_md = result.extract_markdown()
                    if len(test_md) == 0:
                        logging.error("DocStrange extract_markdown() returned empty string!")
                        
                        # ADD THIS: Inspect the raw response
                        if hasattr(result, 'content'):
                            logging.debug(f"Raw result.content type: {type(result.content)}")
                            logging.debug(f"Raw result.content: {result.content}")
                        
                        if hasattr(result, 'metadata'):
                            logging.debug(f"Result metadata: {result.metadata}")
                        
                        if hasattr(result, 'cloud_processor'):
                            proc = result.cloud_processor
                            if hasattr(proc, '_last_response'):
                                logging.debug(f"Last API response: {proc._last_response}")
                            if hasattr(proc, '_api_response'):
                                logging.debug(f"API response: {proc._api_response}")
            
            # Get output in requested format
            extracted_text = self._get_formatted_output(
                result, output_format, **kwargs
            )
            
            if progress_callback:
                progress_callback(1, f"docstrange_{self._mode}_complete")
            
            if self._debug:
                logging.debug(f"DocStrange extracted {len(extracted_text)} chars from {file_basename}")
            
            return extracted_text
            
        except Exception as e:
            logging.error(f"DocStrange extraction failed for {file_basename}: {e}", 
                        exc_info=self._debug)
            return ""
    
    def _get_formatted_output(self, result, output_format: str, **kwargs) -> str:
        """Get output in the requested format - FIXED for actual DocStrange API"""
        try:
            # For CloudConversionResult, we need to handle potential API failures
            if hasattr(result, '__class__') and result.__class__.__name__ == 'CloudConversionResult':
                # Check if this is a lazy cloud result that hasn't made API calls yet
                if hasattr(result, '_cached_outputs'):
                    if self._debug:
                        logging.debug(f"CloudConversionResult cached outputs: {list(result._cached_outputs.keys())}")
            
            if output_format == 'markdown' or output_format == 'markdown-financial-docs':
                # Try extract_markdown() first
                if hasattr(result, 'extract_markdown'):
                    try:
                        content = result.extract_markdown()
                        if content:  # Only return if we got actual content
                            return content
                        else:
                            # If empty, this might be an API failure
                            if self._debug:
                                logging.warning("extract_markdown() returned empty content")
                                # Check for API errors
                                if hasattr(result, 'cloud_processor'):
                                    proc = result.cloud_processor
                                    if hasattr(proc, 'api_key'):
                                        logging.debug(f"API key present: {bool(proc.api_key)}")
                                    if hasattr(proc, 'api_url'):
                                        logging.debug(f"API URL: {proc.api_url}")
                    except Exception as e:
                        logging.error(f"extract_markdown() failed: {e}", exc_info=self._debug)
                
                # Fallback approaches if extract_markdown fails
                if hasattr(result, 'content') and result.content:
                    if self._debug:
                        logging.debug("Using direct result.content")
                    return result.content
                elif isinstance(result, dict) and 'content' in result:
                    return result['content']
                elif isinstance(result, str):
                    return result
                
                # Try to force content extraction for cloud results
                if hasattr(result, '_get_cloud_output'):
                    try:
                        if self._debug:
                            logging.debug("Attempting direct _get_cloud_output call")
                        content = result._get_cloud_output('markdown')
                        if content:
                            return content
                    except Exception as e:
                        logging.error(f"Direct _get_cloud_output failed: {e}")
            
            elif output_format == 'json':
                # Your existing JSON handling code is correct
                specified_fields = kwargs.get('specified_fields')
                json_schema = kwargs.get('json_schema')
                
                if hasattr(result, 'extract_data'):
                    try:
                        extracted_data = result.extract_data(
                            specified_fields=specified_fields,
                            json_schema=json_schema
                        )
                        # Parse response format
                        if isinstance(extracted_data, dict):
                            if 'extracted_fields' in extracted_data:
                                data_to_serialize = extracted_data['extracted_fields']
                            elif 'structured_data' in extracted_data:
                                data_to_serialize = extracted_data['structured_data']
                            else:
                                data_to_serialize = extracted_data
                        else:
                            data_to_serialize = extracted_data
                        
                        import json
                        return json.dumps(data_to_serialize, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logging.error(f"extract_data() failed: {e}", exc_info=self._debug)
            
                
                # Parse response format per docs:
                # Returns: {"extracted_fields": {...}, "format": "specified_fields"}
                # Or: {"structured_data": {...}, "schema": {...}, "format": "structured_json"}
                if isinstance(extracted_data, dict):
                    if 'extracted_fields' in extracted_data:
                        data_to_serialize = extracted_data['extracted_fields']
                    elif 'structured_data' in extracted_data:
                        data_to_serialize = extracted_data['structured_data']
                    else:
                        data_to_serialize = extracted_data
                else:
                    data_to_serialize = extracted_data
                
                import json
                return json.dumps(data_to_serialize, indent=2, ensure_ascii=False)
            
            elif output_format == 'html':
                if hasattr(result, 'extract_html'):
                    return result.extract_html()
                elif isinstance(result, dict) and 'content' in result:
                    return result['content']
            
            elif output_format == 'csv':
                if hasattr(result, 'extract_csv'):
                    return result.extract_csv()
                elif isinstance(result, dict) and 'content' in result:
                    return result['content']
            
            elif output_format == 'text':
                if hasattr(result, 'extract_text'):
                    return result.extract_text()
                elif isinstance(result, dict) and 'content' in result:
                    return result['content']
            
            # Emergency fallback: check if result has ANY extraction method
            if hasattr(result, 'extract_markdown'):
                if self._debug:
                    logging.debug("Fallback: using extract_markdown()")
                return result.extract_markdown()
            
            # If result is a dict, log its structure for debugging
            if self._debug and isinstance(result, dict):
                logging.debug(f"DocStrange result keys: {list(result.keys())}")
                logging.debug(f"DocStrange result type: {type(result)}")
            
            # Final fallback: if we have a CloudConversionResult with no content, 
            # it might be an authentication or rate limit issue
            if hasattr(result, '__class__') and result.__class__.__name__ == 'CloudConversionResult':
                logging.error("CloudConversionResult returned no content - possible causes:")
                logging.error("1. Rate limit exceeded (use 'docstrange login' or API key)")
                logging.error("2. Network/API connectivity issue")
                logging.error("3. File format not supported by cloud API")
                
            return ""

        except Exception as e:
            logging.error(f"Format conversion failed: {e}", exc_info=self._debug)
            
            # Last resort: try extract_markdown
            try:
                if hasattr(result, 'extract_markdown'):
                    return result.extract_markdown()
            except:
                pass
            
            return ""
    
    def extract_metadata(self, file_path: str,
                        schema: Optional[Dict] = None,
                        fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract structured metadata using DocStrange.
        This method is CORRECT - no changes needed.
        
        Args:
            file_path: Path to document
            schema: JSON schema for structured extraction
            fields: List of specific fields to extract
        
        Returns:
            Dictionary with extracted metadata in BiblioForge format
        """
        if not self._init_extractor():
            return {}
        
        try:
            # Extract document
            result = self._extractor_instance.extract(file_path)
            
            # Extract structured data
            if schema:
                raw_metadata = result.extract_data(json_schema=schema)
            elif fields:
                raw_metadata = result.extract_data(specified_fields=fields)
            else:
                # Default metadata fields for bibliographic data
                default_fields = [
                    'title', 'author', 'authors', 'year', 
                    'publication_date', 'publisher', 'language',
                    'doi', 'isbn', 'abstract'
                ]
                raw_metadata = result.extract_data(specified_fields=default_fields)
            
            # FIXED: Parse DocStrange response format
            if isinstance(raw_metadata, dict):
                if 'extracted_fields' in raw_metadata:
                    # Extract from wrapped format
                    fields_data = raw_metadata['extracted_fields']
                elif 'structured_data' in raw_metadata:
                    # Extract from schema-based format
                    fields_data = raw_metadata['structured_data']
                else:
                    # Direct format
                    fields_data = raw_metadata
            else:
                fields_data = raw_metadata
            
            # Convert to BiblioForge metadata format
            # Parse and validate according to BiblioForge needs
            title = fields_data.get('title', '').strip()
            
            # Get author - prefer 'author' field, fallback to first in 'authors'
            author = fields_data.get('author', '').strip()
            if not author and fields_data.get('authors'):
                authors_list = fields_data['authors']
                if isinstance(authors_list, list) and authors_list:
                    author = authors_list[0]
                elif isinstance(authors_list, str):
                    # Parse author list string
                    if ',' in authors_list:
                        author = authors_list.split(',')[0].strip()
                    elif ';' in authors_list:
                        author = authors_list.split(';')[0].strip()
                    else:
                        author = authors_list.strip()
            
            # Get and validate year
            import re
            year = fields_data.get('year', '').strip()
            if not year:
                pub_date = fields_data.get('publication_date', '')
                year_match = re.search(r'\b(19|20)\d{2}\b', str(pub_date))
                if year_match:
                    year = year_match.group(0)
            
            # Import validate function from utils
            from utils import validate_and_fix_year
            year = validate_and_fix_year(year)
            
            # Get language
            language = fields_data.get('language', 'ul').strip().lower()
            if len(language) > 2:
                # Convert full language name to code
                lang_map = {
                    'english': 'en', 'german': 'de', 'french': 'fr',
                    'spanish': 'es', 'italian': 'it', 'portuguese': 'pt',
                    'chinese': 'zh', 'japanese': 'ja', 'korean': 'ko',
                    'russian': 'ru', 'arabic': 'ar'
                }
                language = lang_map.get(language.lower(), language[:2])
            language = language[:2] if len(language) >= 2 else 'ul'
            
            if not title or not author:
                if self._debug:
                    logging.warning(f"DocStrange metadata incomplete for {os.path.basename(file_path)}")
                return {}
            
            return {
                'title': title,
                'author': author,
                'year': year,
                'language': language
            }
            
        except Exception as e:
            logging.error(f"DocStrange metadata extraction failed: {e}", 
                         exc_info=self._debug)
            return {}
    
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
        """
        Run DocStrange authentication flow (docstrange login)
        
        Returns:
            True if successful, False otherwise
        """
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
        """
        Run DocStrange logout to clear cached credentials
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear credentials file
            creds_path = Path.home() / ".docstrange" / "credentials"
            if creds_path.exists():
                creds_path.unlink()
            
            logging.info("✓ DocStrange logout complete")
            return True
            
        except Exception as e:
            logging.error(f"DocStrange logout error: {e}")
            return False