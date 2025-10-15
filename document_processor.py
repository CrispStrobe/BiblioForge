# document_processor.py
import datetime
import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import platform
import traceback # For more detailed error logging if needed
from concurrent.futures import TimeoutError as FutureTimeoutError
import tempfile

# --- Corrected Imports for sibling modules ---
from extraction_manager import ExtractionManager
from utils import (
    file_lock, parse_metadata, sanitize_filename, 
    validate_and_fix_year, add_rename_command, 
    shutdown_flag, valid_author_name, initialize_rename_scripts, safe_initialize_rename_scripts,
    setup_logging, _restore_logging_if_corrupted, should_process_file, purge_rename_script
)
from llm_providers import (
    extract_metadata as llm_extract_metadata, 
    sort_author_names,
    get_llm_provider, 
    LLMProvider,
    OllamaProvider, # For type checking example
    LlamaCPPProvider # For type checking example
    # Import other specific provider classes if directly type-hinted or checked with isinstance
)

# For _extract_pdf_metadata and _extract_epub_metadata
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None 
try:
    from ebooklib import epub
except ImportError:
    epub = None 


class DocumentProcessor:
    """Main document processing coordinator"""
    
    def __init__(self, debug: bool = False):
        self.manager = ExtractionManager(debug=debug)
        self._debug = debug

    def _ensure_logging_intact(self):
        """Ensure logging configuration is still intact"""
        root_logger = logging.getLogger()
        if self._debug and (not root_logger.handlers or root_logger.level > logging.DEBUG):
            # Re-setup logging if it was corrupted
            from utils import setup_logging
            setup_logging(2)  # Debug level
            logging.warning("Logging configuration was corrupted and has been restored.")

    def sort_author_with_retries(
        self,
        author_name: str,
        llm_provider_instance,
        input_file: str,
        **kwargs_from_main
    ) -> str:
        """
        Sorts author names, prioritizing direct comma parsing and using the LLM
        only when necessary. Returns a valid name or "UnknownAuthor".
        """
        if not valid_author_name(author_name):
            return "UnknownAuthor"

        # Pre-process to handle "Lastname, Firstname" format directly
        cleaned_name = self._preprocess_author_name(author_name)
        if ',' in cleaned_name:
            parts = [p.strip() for p in cleaned_name.split(',', 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                # It's already in a parsable format, just re-order it.
                formatted = f"{parts[0]} {parts[1]}"
                return formatted if valid_author_name(formatted) else "UnknownAuthor"

        # If it looks like "Firstname Lastname", use LLM to sort
        if len(cleaned_name.split()) >= 2:
            try:
                # We extract only the parameters that sort_author_names expects
                sorted_name = sort_author_names(
                    author_names_input=cleaned_name,
                    provider_arg=llm_provider_instance,
                    verbose=self._debug,
                    filename_for_logging=input_file,
                    prompt_template_index=0  # Start with first template
                    # Don't pass **kwargs_from_main - it contains incompatible params
                )
                if valid_author_name(sorted_name):
                    return sorted_name
            except Exception as e:
                if self._debug:
                    logging.warning(f"LLM sort failed for '{cleaned_name}': {e}")
        
        # If all else fails, return the cleaned name if it's valid, otherwise UnknownAuthor
        return cleaned_name if valid_author_name(cleaned_name) else "UnknownAuthor"
    
    def sort_author_with_retries_old(
        self, 
        author_name: str, 
        llm_provider_instance, 
        input_file: str, 
        max_template_retries: int = 3,
        sort_timeout: int = 60,  # NEW: 1 minute timeout per sort attempt
        **kwargs_from_main
    ) -> str:
        """
        Sort author name with timeout protection.
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        if not valid_author_name(author_name):
            return "UnknownAuthor"
        
        # Pre-process comma-separated names
        if ',' in author_name:
            parts = [p.strip() for p in author_name.split(',')]
            if len(parts) == 2 and all(valid_author_name(p) for p in parts):
                formatted_name = f"{parts[0]} {parts[1]}"
                if self._debug:
                    logging.debug(f"sort_author_with_retries: Pre-sorted comma-separated name '{author_name}' to '{formatted_name}'")
                return formatted_name
        
        def is_likely_correct_format(name: str) -> bool:
            parts = name.split()
            if len(parts) >= 2:
                lastname = parts[0]
                firstname_parts = " ".join(parts[1:])
                
                is_lastname_valid = lastname.replace('-', '').replace("'", '').isalpha()
                is_firstname_valid = firstname_parts.replace('-', '').replace("'", '').isalpha()

                if (lastname[0].isupper() and firstname_parts[0].isupper() and
                    2 <= len(lastname) <= 30 and 2 <= len(firstname_parts) <= 30 and
                    is_lastname_valid and is_firstname_valid):
                    return True
            return False
        
        if is_likely_correct_format(author_name):
            if self._debug:
                logging.debug(f"sort_author_with_retries: '{author_name}' already in correct format")
            return author_name
        
        all_results = []
        
        for template_idx in range(max_template_retries):
            if shutdown_flag.is_set():
                break
                    
            if self._debug:
                logging.debug(f"sort_author_with_retries: Trying template {template_idx} for '{author_name}'")
            
            try:
                # Use ThreadPoolExecutor with timeout for sort call
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        sort_author_names,
                        author_names_input=author_name,
                        provider_arg=llm_provider_instance,
                        verbose=self._debug,
                        filename_for_logging=input_file,
                        prompt_template_index=template_idx,
                        **kwargs_from_main
                    )
                    
                    try:
                        sorted_name = future.result(timeout=sort_timeout)
                    except FutureTimeoutError:
                        logging.warning(f"sort_author_with_retries: Sort timed out after {sort_timeout}s for '{author_name}' (template {template_idx})")
                        continue
                
                if sorted_name and valid_author_name(sorted_name):
                    all_results.append(sorted_name)
                    if self._debug:
                        logging.debug(f"sort_author_with_retries: Template {template_idx} result: '{sorted_name}'")
                else:
                    if self._debug:
                        logging.warning(f"sort_author_with_retries: Template {template_idx} returned invalid: '{sorted_name}'")
                            
            except Exception as e:
                if self._debug:
                    logging.warning(f"sort_author_with_retries: Template {template_idx} failed: {e}")
                continue
        
        if not all_results:
            if self._debug:
                logging.warning(f"sort_author_with_retries: All attempts failed for '{author_name}'")
            return author_name if valid_author_name(author_name) else "UnknownAuthor"
        
        # Analyze results for consistency
        if len(set(all_results)) == 1:
            return all_results[0]
        else:
            from collections import Counter
            result_counts = Counter(all_results)
            most_common = result_counts.most_common(1)[0]
            
            if most_common[1] > 1:
                return most_common[0]
            else:
                best_result = all_results[0]
                for result in all_results:
                    if is_likely_correct_format(result):
                        best_result = result
                        break
                return best_result
            
    def _get_poppler_path(self) -> Optional[str]:
        """Get poppler path from binary_paths if available"""
        if hasattr(self.manager, '_binary_paths') and 'pdftoppm' in self.manager._binary_paths:
            return os.path.dirname(self.manager._binary_paths['pdftoppm'])
        return None
            
    def extract_metadata_with_nanonets_ocr2(self, file_path: str,
                                            nanonets_extractor) -> Optional[Dict[str, str]]:
        """Extract metadata using Nanonets-OCR2 VQA"""
        try:
            # For PDF, convert first page to image
            if file_path.lower().endswith('.pdf'):
                from pdf2image import convert_from_path
                poppler_path = self._get_poppler_path()
                
                images = convert_from_path(
                    file_path, dpi=300, first_page=1, last_page=1,
                    poppler_path=poppler_path
                )
                
                if not images:
                    return None
                
                # Save first page as temp image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    temp_image_path = tmp.name
                    images[0].save(temp_image_path, 'PNG')
                
                try:
                    metadata = nanonets_extractor.extract_metadata_fields(
                        temp_image_path,
                        fields=['title', 'author', 'authors', 'year', 'language']
                    )
                finally:
                    try:
                        os.unlink(temp_image_path)
                    except:
                        pass
            else:
                # For image files
                metadata = nanonets_extractor.extract_metadata_fields(
                    file_path,
                    fields=['title', 'author', 'authors', 'year', 'language']
                )
            
            if not metadata:
                return None
            
            # Parse and validate
            title = metadata.get('title', '').strip()
            author = metadata.get('author', '').strip()
            
            # Handle authors list
            if not author and metadata.get('authors'):
                authors_str = metadata['authors']
                if ',' in authors_str:
                    author = authors_str.split(',')[0].strip()
                elif ';' in authors_str:
                    author = authors_str.split(';')[0].strip()
                else:
                    author = authors_str.strip()
            
            # Validate and fix year
            year = validate_and_fix_year(metadata.get('year', ''))
            
            # Get language
            language = metadata.get('language', 'ul').strip().lower()
            if len(language) > 2:
                lang_map = {
                    'english': 'en', 'german': 'de', 'french': 'fr',
                    'spanish': 'es', 'italian': 'it', 'portuguese': 'pt'
                }
                language = lang_map.get(language.lower(), 'ul')[:2]
            
            if not title or not author:
                return None
            
            return {
                'title': title,
                'author': author,
                'year': year,
                'language': language
            }
            
        except Exception as e:
            logging.error(f"Nanonets-OCR2 metadata extraction failed: {e}", 
                        exc_info=self._debug)
            return None
            
    def extract_metadata_with_docstrange(self, file_path: str,
                                        docstrange_extractor,
                                        metadata_fields: Optional[List[str]] = None,
                                        metadata_schema: Optional[Dict] = None) -> Optional[Dict[str, str]]:
        """
        Extract metadata using DocStrange structured extraction.
        
        Returns metadata in BiblioForge format: {title, author, year, language}
        """
        try:
            # Use provided fields/schema or default
            if not metadata_fields and not metadata_schema:
                metadata_fields = [
                    'title', 'author', 'authors', 'year',
                    'publication_date', 'publisher', 'language'
                ]
            
            # FIXED: Use correct DocStrange API
            # First extract the document
            result = docstrange_extractor._extractor_instance.extract(file_path)
            
            # Then get structured data using extract_data()
            if metadata_schema:
                raw_metadata = result.extract_data(json_schema=metadata_schema)
            elif metadata_fields:
                raw_metadata = result.extract_data(specified_fields=metadata_fields)
            else:
                raw_metadata = result.extract_data()
            
            if not raw_metadata:
                return None
            
            # Handle different response formats
            if 'extracted_fields' in raw_metadata:
                # Format: {"extracted_fields": {...}, "format": "..."}
                fields_data = raw_metadata['extracted_fields']
            elif 'structured_data' in raw_metadata:
                # Format: {"structured_data": {...}, "schema": {...}}
                fields_data = raw_metadata['structured_data']
            else:
                # Direct format
                fields_data = raw_metadata
            
            # Parse and validate
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
            year = fields_data.get('year', '').strip()
            if not year:
                pub_date = fields_data.get('publication_date', '')
                year_match = re.search(r'\b(19|20)\d{2}\b', str(pub_date))
                if year_match:
                    year = year_match.group(0)
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
                logging.warning(f"DocStrange metadata incomplete for {os.path.basename(file_path)}")
                return None
            
            return {
                'title': title,
                'author': author,
                'year': year,
                'language': language
            }
            
        except Exception as e:
            logging.error(f"DocStrange metadata extraction failed: {e}",
                        exc_info=self._debug)
            return None

    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = {}
        if PdfReader is None:
            if self._debug: logging.debug(f"_extract_pdf_metadata: pypdf not available for {file_path}.")
            return metadata
        
        # Check file size before attempting to read
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                if self._debug: 
                    logging.debug(f"_extract_pdf_metadata: Skipping 0-byte file: {file_path}")
                return metadata
        except OSError as e:
            if self._debug: 
                logging.debug(f"_extract_pdf_metadata: Cannot access file {file_path}: {e}")
            return metadata
        
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                doc_info = reader.metadata
                if doc_info:
                    for key, value in doc_info.items():
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            metadata[key.lstrip('/')] = value 
                        else:
                            metadata[key.lstrip('/')] = str(value)
            if self._debug: logging.debug(f"_extract_pdf_metadata: Extracted for {file_path}: {str(metadata)[:200]}...")
        except Exception as e:
            if self._debug: logging.warning(f"_extract_pdf_metadata: Failed for {file_path}: {e}", exc_info=self._debug)
        return metadata

    def _extract_epub_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = {}
        if epub is None:
            if self._debug: logging.debug(f"_extract_epub_metadata: ebooklib not available for {file_path}.")
            return metadata
        
        # Check file size before attempting to read
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                if self._debug: 
                    logging.debug(f"_extract_epub_metadata: Skipping 0-byte file: {file_path}")
                return metadata
        except OSError as e:
            if self._debug: 
                logging.debug(f"_extract_epub_metadata: Cannot access file {file_path}: {e}")
            return metadata
        
        try:
            book = epub.read_epub(file_path)
            dc_fields = ['title', 'creator', 'subject', 'description', 'publisher', 
                        'contributor', 'date', 'type', 'format', 'identifier', 
                        'source', 'language', 'relation', 'coverage', 'rights']
            for field in dc_fields:
                meta_values = book.get_metadata('DC', field)
                if meta_values:
                    processed_values = [item[0] for item in meta_values if item and item[0]]
                    if len(processed_values) == 1:
                        metadata[field] = processed_values[0]
                    elif len(processed_values) > 1:
                        metadata[field] = processed_values
            if 'creator' in metadata:
                metadata['authors'] = metadata['creator']
                if not isinstance(metadata['authors'], list):
                    metadata['authors'] = [metadata['authors']]
            if self._debug: logging.debug(f"_extract_epub_metadata: Extracted for {file_path}: {str(metadata)[:200]}...")
        except Exception as e:
            if self._debug: logging.warning(f"_extract_epub_metadata: Failed for {file_path}: {e}", exc_info=self._debug)
        return metadata

    def _extract_document_inherent_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = {}
        try:
            file_info = Path(file_path)
            
            # Get basic file stats
            try:
                file_stat = file_info.stat()
                file_size = file_stat.st_size
                
                metadata.update({
                    'filename': file_info.name,
                    'size': file_size,
                    'modified_datetime': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
                
                # Skip format-specific metadata extraction for 0-byte files
                if file_size == 0:
                    if self._debug: 
                        logging.debug(f"_extract_document_inherent_metadata: Skipping format-specific metadata for 0-byte file: {file_path}")
                    return metadata
                    
            except OSError as e:
                if self._debug: 
                    logging.debug(f"_extract_document_inherent_metadata: Cannot stat file {file_path}: {e}")
                return metadata
            
            # Extract format-specific metadata only if file has content
            file_suffix_lower = file_info.suffix.lower()
            if file_suffix_lower == '.pdf':
                metadata.update(self._extract_pdf_metadata(file_path))
            elif file_suffix_lower == '.epub':
                metadata.update(self._extract_epub_metadata(file_path))
            
            if self._debug: 
                logging.debug(f"_extract_document_inherent_metadata: Extracted for {file_path}: {str(metadata)[:200]}...")
        except Exception as e:
            if self._debug: 
                logging.error(f"_extract_document_inherent_metadata: Failed for {file_path}: {e}", exc_info=self._debug)
        return metadata
        
    def _get_unique_output_path(self, input_file: str, output_dir_base_str: Optional[str] = None, noskip: bool = False) -> str:
        input_basename = os.path.basename(input_file)
        input_stem = Path(input_basename).stem
        
        eff_output_dir = Path(output_dir_base_str or '.').resolve()
        eff_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_name = f"{input_stem}.txt"
        output_path = eff_output_dir / output_name
        
        # This condition needs to check the resolved output_path specifically
        if noskip and output_path.exists():
            counter = 1
            # Start with the base path and check if it needs a suffix
            current_check_path = output_path
            while current_check_path.exists():
                output_name = f"{input_stem}_{counter}.txt"
                current_check_path = eff_output_dir / output_name
                counter += 1
            # current_check_path is now the first unique path
            final_output_path = current_check_path
            if self._debug: logging.debug(f"_get_unique_output_path: Using unique output path: {str(final_output_path)} for original stem '{input_stem}'")
            return str(final_output_path)

        elif not noskip and output_path.exists():
            if self._debug: logging.debug(f"_get_unique_output_path: Output file {str(output_path)} exists and noskip=False. Caller will decide to skip/reuse.")
        elif self._debug: # Path does not exist, or noskip=False and path doesn't exist
            logging.debug(f"_get_unique_output_path: Standard output path: {str(output_path)} (noskip={noskip}, exists={output_path.exists()})")
            
        return str(output_path)
    
    def has_invalid_author_keywords(self, author_name: str) -> bool:
        """
        Check if author name contains invalid placeholder keywords that indicate LLM failure.
        Uses word boundary matching to avoid false positives.
        """
        if not author_name:
            return True
        
        author_lower = author_name.lower().strip()
        
        # Remove potential prefixes that LLMs sometimes add
        author_to_check = re.sub(r"^(>?(Main Author|Author|Editor|By):?\s*)", "", author_lower, flags=re.IGNORECASE).strip()
        
        # Check minimum length
        if len(author_to_check) < 2:
            return True
        
        # Keywords that indicate the LLM failed to extract a real author name
        # Use word boundaries to avoid false positives with legitimate names
        invalid_patterns = [
            r'\bunknownauthor\b', r'\bunknown\s+author\b', r'\bunknown\b',
            r'\blastname\s+firstname\b', r'\bfirstname\s+lastname\b', r'\bsurname\s+name\b',
            r'\bauthor\s+name\b', r'\bname\s+author\b', r'\bmain\s+author\b', r'\bprimary\s+author\b',
            r'\bcontributor\b', r'\bwriter\b', r'\bcreator\b', r'\beditor\b', r'\beditors\b',
            r'\bextracted\b', r'\bpublication\b', r'\bdocument\b', r'\bvarious\b',
            r'\bmultiple\b', r'\bet\s+al\b', r'\band\s+others\b', r'\banonymous\b',
            r'\bn/?a\b', r'\bnicht\s+verfügbar\b', r'\bunbekannt\b', r'\bautor\b', r'\bverfasser\b'
        ]
        
        # Check for word boundary matches with invalid patterns
        for pattern in invalid_patterns:
            if re.search(pattern, author_to_check):
                return True
        
        # Check for exact matches with problematic placeholders
        exact_invalid = ['lastname firstname', 'firstname lastname', 'surname name']
        if author_to_check in exact_invalid:
            return True
        
        # Check if it's too short or just punctuation
        meaningful_chars = ''.join(c for c in author_to_check if c.isalnum())
        if len(meaningful_chars) < 3:
            return True
        
        return False
    
    def _preprocess_author_name(self, author_name: str) -> str:
        """
        Cleans and pre-formats author names before more complex validation or LLM sorting.
        This function handles the most common "dirty" formats.
        """
        if not author_name or not isinstance(author_name, str):
            return "" # Return empty string to be caught by validation
        
        # Strip XML tags and common LLM boilerplate text
        cleaned_name = re.sub(r"^\s*<AUTHOR>\s*|\s*</AUTHOR>\s*$", "", author_name, flags=re.IGNORECASE).strip()
        cleaned_name = re.sub(r"^(By|Author|Editor|Written by)[s]?:\s*", "", cleaned_name, flags=re.IGNORECASE).strip()

        # If multiple authors are listed, take only the first one for simplicity in sorting and filing.
        # This splits by common delimiters like ';', '&', ' and ', or a newline.
        cleaned_name = re.split(r'\s*;\s*|\s*&\s*|\s+and\s+|\s*\n\s*', cleaned_name, flags=re.IGNORECASE)[0].strip()

        # Directly handle "Lastname, Firstname" format, which is a very common, reliable pattern.
        if ',' in cleaned_name:
            parts = [p.strip() for p in cleaned_name.split(',', 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                # Reconstruct as "Lastname Firstname" and return, as it's likely already correct.
                return f"{parts[0]} {parts[1]}"
        
        return cleaned_name

    def _is_valid_title(self, title: str) -> bool:
        """
        Checks if the extracted title is a valid, meaningful title and not a placeholder.
        """
        if not title or not isinstance(title, str) or len(title.strip()) < 3:
            return False
        
        lower_title = title.lower().strip()
        
        # List of common invalid or placeholder patterns returned by LLMs
        invalid_patterns = [
            r'^untitled$', r'^unknown$', r'^no title$', r'^title$',
            r'^document$', r'^untitled document$', r'extracted publication title',
            r'^publication title$', r'the exact title of the publication',
            r'^[.,\-_/\s]+$' # Title is only punctuation or whitespace
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, lower_title):
                return False
        
        return True

    def process_llm_metadata_with_retries(
        self,
        text: str,
        input_file: str,
        llm_provider_instance,
        **kwargs_from_main # Catches temperature, max_tokens, etc.
    ) -> Optional[Dict[str, str]]:
        """
        Runs multiple prompt templates and intelligently consolidates the best results
        from all attempts for a more robust metadata extraction.
        """
        max_llm_template_attempts = 3
        all_results = []

        for template_attempt in range(max_llm_template_attempts):
            if shutdown_flag.is_set(): break
            try:
                llm_response_str = llm_extract_metadata(
                    text=text, filename=input_file,
                    llm_provider_arg=llm_provider_instance, verbose=self._debug,
                    prompt_template_index_to_try=template_attempt, **kwargs_from_main
                )
                if llm_response_str:
                    parsed_meta = parse_metadata(llm_response_str, os.path.basename(input_file))
                    if parsed_meta:
                        all_results.append(parsed_meta)
            except Exception as e:
                logging.error(f"DS_Proc: Error on template {template_attempt}: {e}", exc_info=self._debug)

        if not all_results:
            logging.error(f"DS_Proc: All LLM attempts failed for '{os.path.basename(input_file)}'.")
            return None

        # --- Intelligent Consolidation ---
        best_title = ""
        best_author = ""
        best_year = "UnknownYear"
        
        # Find the best non-placeholder title (prefer longest)
        valid_titles = [r['title'] for r in all_results if self._is_valid_title(r.get('title'))]
        if valid_titles:
            best_title = max(valid_titles, key=len)

        # Find the best non-placeholder author (prefer longest)
        valid_authors = [r['author'] for r in all_results if not self.has_invalid_author_keywords(r.get('author'))]
        if valid_authors:
            best_author = max(valid_authors, key=len)

        # Find the best valid year
        valid_years = [validate_and_fix_year(r.get('year')) for r in all_results]
        numeric_years = [y for y in valid_years if y != "UnknownYear"]
        if numeric_years:
            best_year = max(numeric_years)

        # --- Final Processing and Validation ---
        if not best_title or not best_author:
            logging.error(f"DS_Proc: Failed to consolidate a valid title and author. Best attempt: Title='{best_title}', Author='{best_author}'")
            return None

        sorted_author = self.sort_author_with_retries(
            best_author, llm_provider_instance, input_file, **kwargs_from_main
        )

        final_metadata = {
            'title': best_title,
            'year': best_year,
            'author': "UnknownAuthor", # Default to UnknownAuthor
            'language': all_results[0].get('language', 'ul')
        }

        # Assign the sorted author name only if it's valid
        if valid_author_name(sorted_author):
            final_metadata['author'] = sorted_author
        elif valid_author_name(best_author): # Fallback to the best unsorted name
             final_metadata['author'] = best_author

        logging.info(f"DS_Proc: Successfully consolidated metadata for '{os.path.basename(input_file)}'")
        return final_metadata

    def _process_single_file(self, input_file: str,
                        current_output_dir_base_str: str,
                        method: Optional[str],
                        ocr_method: Optional[str],
                        password: Optional[str],
                        extract_tables: bool,
                        force_ocr: bool,
                        noskip: bool,
                        effective_sort_flag: bool,
                        rename_script_base_path_for_unparseables: Optional[str],
                        initialized_rename_script_paths: Optional[Dict[str, Optional[str]]],
                        llm_provider_instance: Optional[LLMProvider],
                        temperature_for_metadata: float,
                        max_tokens_for_metadata: int,
                        **kwargs_from_main) -> Dict[str, Any]:

        debug = self._debug
        
        if self._debug:
            logging.debug(f"DS_Proc: Starting _process_single_file for: '{input_file}'")
            logging.debug(f"Processing: {os.path.basename(input_file)}")

        # INITIALIZE result FIRST
        result: Dict[str, Any] = {
            'success': False, 'text': '', 'output_path': None, 'skipped': False,
            'error': None, 'tables': [], 'metadata_llm': None, 'renamed_info': None,
            'input_file': input_file, 'inherent_metadata': {}
        }
        
        actual_text_content_path_for_llm_and_rename: Optional[str] = None

        # --- EARLY FILE VALIDATION ---
        try:
            file_path = Path(input_file)
            if not file_path.exists():
                error_message = f"File '{input_file}' does not exist."
                logging.warning(f"DS_Proc: {error_message}")
                result['error'] = error_message
                result['success'] = False
                return result
                    
            file_size = file_path.stat().st_size
            if file_size == 0:
                error_message = f"File '{input_file}' is 0 bytes and cannot be processed."
                logging.warning(f"DS_Proc: {error_message}")
                result['error'] = error_message
                result['success'] = False
                try: 
                    result['inherent_metadata'] = self._extract_document_inherent_metadata(input_file)
                except: 
                    pass
                return result
                    
            # Check for very small files that might be corrupted
            if file_size < 100 and not input_file.lower().endswith(('.txt', '.md')):
                error_message = f"File '{input_file}' is suspiciously small ({file_size} bytes) and might be corrupted."
                logging.warning(f"DS_Proc: {error_message}")
                result['error'] = error_message
                result['success'] = False
                return result
                    
        except OSError as e_size:
            error_message = f"Could not access file '{input_file}': {e_size}"
            logging.warning(f"DS_Proc: {error_message}")
            result['error'] = error_message
            result['success'] = False
            return result
        
        # --- NEW: Determine skip logic BEFORE any processing ---
        if shutdown_flag.is_set():
            result['error'] = "Processing aborted due to shutdown signal"
            return result
        
        try:
            is_source_direct_text_type = input_file.lower().endswith(('.txt', '.md'))
            text_loaded_from_file = False
            
            # Determine output path
            if is_source_direct_text_type:
                determined_extraction_output_path = input_file
                actual_text_content_path_for_llm_and_rename = input_file
            else:
                determined_extraction_output_path = self._get_unique_output_path(
                    input_file, current_output_dir_base_str, noskip
                )
                actual_text_content_path_for_llm_and_rename = determined_extraction_output_path
            
            result['output_path'] = determined_extraction_output_path
            
            # **NEW: Check what processing should be done**
            rename_script_to_check = None
            if initialized_rename_script_paths:
                # Use the first available script path for checking
                rename_script_to_check = (
                    initialized_rename_script_paths.get('bash_script') or 
                    initialized_rename_script_paths.get('batch_script')
                )
            
            skip_info = should_process_file(
                input_file=input_file,
                output_txt_path=determined_extraction_output_path,
                rename_script_path=rename_script_to_check,
                sort_enabled=effective_sort_flag,
                noskip=noskip
            )
            
            if self._debug:
                logging.debug(f"DS_Proc: Skip info for '{input_file}': {skip_info}")
            
            # **Case 1: Skip entirely** (txt exists AND in rename script)
            if skip_info['skip_entirely']:
                logging.info(f"Skipping '{input_file}' - already processed and in rename script.")
                result['skipped'] = True
                result['success'] = True
                try:
                    result['inherent_metadata'] = self._extract_document_inherent_metadata(input_file)
                except:
                    pass
                return result
            
            # **Case 2: Skip extraction only** (txt exists but NOT in rename script)
            if skip_info['skip_extraction'] and not is_source_direct_text_type:
                if self._debug:
                    logging.debug(f"DS_Proc: Loading existing text from '{determined_extraction_output_path}' for '{input_file}'")
                try:
                    with open(determined_extraction_output_path, 'r', encoding='utf-8') as f:
                        result['text'] = f.read()
                    if result['text'].strip():
                        result['success'] = True
                        text_loaded_from_file = True
                        if self._debug:
                            logging.debug(f"DS_Proc: Successfully loaded {len(result['text'])} chars from existing file")
                    else:
                        if self._debug:
                            logging.warning(f"DS_Proc: Existing text file is empty, will re-extract")
                        result['success'] = False
                except Exception as e_load:
                    if self._debug:
                        logging.warning(f"DS_Proc: Failed to load existing text file: {e_load}, will re-extract")
                    result['success'] = False
            
            # **Case 3: Full extraction** (txt doesn't exist or failed to load)
            if is_source_direct_text_type and not text_loaded_from_file:
                try:
                    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
                        result['text'] = f.read()
                    if result['text'].strip():
                        result['success'] = True
                        text_loaded_from_file = True
                    else:
                        result['error'] = f"Source text file '{input_file}' is empty."
                except Exception as e_read_txt:
                    result['error'] = f"Failed to read source text file '{input_file}': {e_read_txt}"
            
            if not text_loaded_from_file and not is_source_direct_text_type:
                if self._debug:
                    logging.debug(f"DS_Proc: Performing extraction for '{input_file}'")
                
                extraction_result = self.manager.extract(
                    input_path=input_file, 
                    output_path=actual_text_content_path_for_llm_and_rename,
                    method=method, ocr_method=ocr_method, password=password,
                    extract_tables=extract_tables, force_ocr=force_ocr,
                    temperature=temperature_for_metadata, max_tokens=max_tokens_for_metadata,
                    **kwargs_from_main
                )
                result['text'] = extraction_result.get('text', '')
                result['success'] = extraction_result.get('success', False)
                result['tables'] = extraction_result.get('tables', [])
                if extraction_result.get('error'): 
                    result['error'] = extraction_result.get('error')
                if result['success']: 
                    result['output_path'] = actual_text_content_path_for_llm_and_rename
            
            # --- LLM Metadata Extraction and Sorting ---
            if result['success'] and result['text'] and actual_text_content_path_for_llm_and_rename and effective_sort_flag:
                # **NEW: Check if we should skip LLM processing**
                if skip_info.get('skip_llm_and_rename', False):
                    if self._debug:
                        logging.debug(f"DS_Proc: Skipping LLM and rename for '{input_file}' - already in script")
                else:
                    if llm_provider_instance:
                        if not initialized_rename_script_paths and self._debug:
                            logging.warning(f"DS_Proc: Sort is True for '{input_file}', but rename scripts not initialized. Cannot generate rename commands.")

                        # NEW: Try specialized extraction first if available
                        final_parsed_llm_meta = None
                        
                        # Check if Nanonets-OCR2 metadata extraction is enabled
                        nanonets_config = kwargs_from_main.get('nanonets_config')
                        if nanonets_config and nanonets_config.get('use_for_metadata'):
                            nanonets_extractor = self.manager._get_extractor(
                                input_file,
                                use_nanonets_ocr2=True,
                                nanonets_config=nanonets_config
                            )
                            if nanonets_extractor:
                                if self._debug:
                                    logging.debug(f"DS_Proc: Trying Nanonets-OCR2 metadata extraction for '{input_file}'")
                                final_parsed_llm_meta = self.extract_metadata_with_nanonets_ocr2(
                                    input_file, nanonets_extractor
                                )
                        
                        # Check if DocStrange metadata extraction is enabled
                        if not final_parsed_llm_meta:
                            docstrange_config = kwargs_from_main.get('docstrange_config')
                            if docstrange_config and docstrange_config.get('use_for_metadata'):
                                docstrange_extractor = self.manager._get_extractor(
                                    input_file,
                                    use_docstrange=True,
                                    docstrange_config=docstrange_config
                                )
                                if docstrange_extractor:
                                    if self._debug:
                                        logging.debug(f"DS_Proc: Trying DocStrange metadata extraction for '{input_file}'")
                                    final_parsed_llm_meta = self.extract_metadata_with_docstrange(
                                        input_file,
                                        docstrange_extractor,
                                        metadata_fields=docstrange_config.get('metadata_fields'),
                                        metadata_schema=docstrange_config.get('metadata_schema')
                                    )
                        
                        # Fallback to standard LLM text-based extraction
                        if not final_parsed_llm_meta:
                            final_parsed_llm_meta = self.process_llm_metadata_with_retries(
                                text=result["text"],
                                input_file=input_file,
                                llm_provider_instance=llm_provider_instance,
                                temperature=temperature_for_metadata,
                                max_tokens=max_tokens_for_metadata,
                                **kwargs_from_main
                            )

                        # Continue with rename command generation...
                        if final_parsed_llm_meta:
                            result['metadata_llm'] = final_parsed_llm_meta
                            if initialized_rename_script_paths:
                                if self._debug:
                                    logging.debug(f"DS_Proc: Generating rename command for '{input_file}' with Author='{final_parsed_llm_meta['author']}', Year='{final_parsed_llm_meta['year']}', Title='{final_parsed_llm_meta['title']}'")
                                
                                rename_info_dict = add_rename_command(
                                    rename_script_paths=initialized_rename_script_paths,
                                    source_path_original_file=input_file,
                                    actual_text_content_file_path=actual_text_content_path_for_llm_and_rename,
                                    target_dir_name=final_parsed_llm_meta['author'],
                                    new_filename_base=f"{final_parsed_llm_meta['year']} {final_parsed_llm_meta['title']}",
                                    output_dir_for_sorted_files=str(current_output_dir_base_str),
                                    debug=self._debug
                                )
                                result["renamed_info"] = rename_info_dict
                                if self._debug and rename_info_dict:
                                    logging.debug(f"DS_Proc: Rename command generated for {input_file}. Info: {rename_info_dict}")
                                elif self._debug and not rename_info_dict:
                                    logging.debug(f"DS_Proc: add_rename_command returned None (likely duplicate) for {input_file}")
                        else:
                            logging.warning(f"DS_Proc: All enhanced LLM attempts failed for '{input_file}'")
                            if rename_script_base_path_for_unparseables:
                                unparseables_path = Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst"
                                with file_lock:
                                    with open(unparseables_path, "a", encoding='utf-8') as f_unp:
                                        f_unp.write(f"{input_file} - All LLM attempts failed to yield sortable metadata.\n")
            
            elif not result['success'] and not result.get('skipped'):
                if not result.get("error"): 
                    result['error'] = "Text extraction or direct read failed/produced no text."
            elif not result['text'] and not result.get('skipped'):
                result['error'] = "Text acquisition successful but produced empty text."
                result['success'] = False
            
            if not result.get('skipped'):
                try:
                    result['inherent_metadata'] = self._extract_document_inherent_metadata(input_file)
                except Exception as e_meta_final:
                    if self._debug: 
                        logging.debug(f"DS_Proc: Could not get inherent metadata for '{input_file}' at the end: {e_meta_final}")

        except Exception as e:
            result['error'] = f"Overall processing of '{input_file}' failed in _process_single_file: {str(e)}"
            logging.error(result['error'], exc_info=self._debug)
            result['success'] = False
            if effective_sort_flag and rename_script_base_path_for_unparseables:
                try:
                    with file_lock:
                        unparseables_path = Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst"
                        with open(unparseables_path, "a", encoding='utf-8') as f_unp:
                            f_unp.write(f"{input_file} - Overall Error in _process_single_file: {str(e)}\n")
                except Exception as e_unp:
                    logging.error(f"Failed to write to unparseables.lst for {input_file}: {e_unp}")
        
        if self._debug:
            logging.debug(f"DS_Proc: Finished _process_single_file for: '{input_file}', "
                        f"Success: {result['success']}, Error: {result.get('error')}, "
                        f"Renamed: {bool(result.get('renamed_info'))}, "
                        f"Skipped: {result.get('skipped')}")
        return result

    def process_files(self, input_files: List[str], 
                  output_dir: Optional[str] = None,
                  method: Optional[str] = None,
                  ocr_method: Optional[str] = None,
                  password: Optional[str] = None,
                  extract_tables: bool = False,  
                  force_ocr: bool = False,
                  max_workers: Optional[int] = None,
                  noskip: bool = False,
                  sort: bool = False,          
                  rename_script_path: Optional[str] = None,
                  reset_rename_script: bool = False,  # NEW
                  noremove_from_script: bool = False,  # NEW
                  llm_provider_arg: Optional[Any] = None, 
                  temperature: float = 0.7,       
                  max_tokens: int = 300,
                  **kwargs) -> Dict[str, Any]:
        
        if self._debug:
            logging.debug(f"process_files: Starting. Total input_files: {len(input_files)}")
            logging.debug(f"process_files: output_dir='{output_dir}', sort (original intent)='{sort}', rename_script_path='{rename_script_path}'")

        final_results: Dict[str, Any] = {'processed': {}, 'failed': [], 'skipped': [], 'counters': {}}
        counters = {'total': len(input_files), 'processed_ok': 0, 'skipped': 0, 
                    'failed_extraction': 0, 'sorted_commands_generated': 0, 'sort_failed_metadata': 0}

        eff_output_dir = Path(output_dir or '.').resolve()
        eff_output_dir.mkdir(parents=True, exist_ok=True)
        if self._debug: 
            logging.debug(f"process_files: Effective output directory: {eff_output_dir}")

        llm_provider_instance: Optional[LLMProvider] = None
        effective_sort_flag = sort

        if effective_sort_flag:
            if isinstance(llm_provider_arg, LLMProvider):
                llm_provider_instance = llm_provider_arg
                if self._debug: 
                    logging.debug("process_files: Using pre-passed LLMProvider instance.")
            elif isinstance(llm_provider_arg, str) or llm_provider_arg is None:
                provider_name_str = llm_provider_arg if isinstance(llm_provider_arg, str) else "ollama"
                try:
                    model_name_for_provider = kwargs.get('llm_model') 
                    api_key_for_provider = kwargs.get('api_key')    
                    
                    provider_specific_constructor_args = {
                        "ollama_host": kwargs.get('ollama_host'),
                        "local_openai_base_url": kwargs.get('local_openai_base_url'),
                        "llamacpp_repo_id": kwargs.get('llamacpp_repo_id'),
                        "llamacpp_gguf_filename": kwargs.get('llamacpp_gguf_filename'),
                        "llamacpp_n_ctx": kwargs.get('llamacpp_n_ctx'),
                        "llamacpp_n_gpu_layers": kwargs.get('llamacpp_n_gpu_layers'),
                        "llamacpp_chat_format": kwargs.get('llamacpp_chat_format')
                    }
                    provider_specific_constructor_args = {k: v for k, v in provider_specific_constructor_args.items() if v is not None}

                    if self._debug: 
                        logging.debug(f"process_files: Attempting to initialize LLM provider '{provider_name_str}'")
                    
                    llm_provider_instance = get_llm_provider(
                        provider_type=provider_name_str,
                        model_name=model_name_for_provider,
                        api_key=api_key_for_provider,
                        **provider_specific_constructor_args
                    )
                    if llm_provider_instance and self._debug:
                        logging.debug(f"process_files: LLM provider initialized successfully.")

                except Exception as e:
                    logging.error(f"Failed to initialize LLM provider '{provider_name_str}' for sorting: {e}. Sorting will be disabled.", exc_info=self._debug)
                    effective_sort_flag = False 
            else: 
                logging.warning(f"Invalid llm_provider_arg type '{type(llm_provider_arg)}' for process_files. Disabling sorting.")
                effective_sort_flag = False
        
        actual_max_workers = max_workers
        if actual_max_workers is None:
            is_local_llm_active = effective_sort_flag and llm_provider_instance and \
                                (isinstance(llm_provider_instance, OllamaProvider) or \
                                isinstance(llm_provider_instance, LlamaCPPProvider) or \
                                'local_openai' in llm_provider_instance.__class__.__name__.lower())
            cpu_count = os.cpu_count() or 1
            actual_max_workers = min(4, cpu_count) if is_local_llm_active else min(8, cpu_count)
        
        logging.info(f"Using up to {actual_max_workers} worker threads. Effective sort enabled: {effective_sort_flag}")

        actual_rename_script_base_path_str: Optional[str] = None
        initialized_script_paths_map: Optional[Dict[str, Optional[str]]] = None

        if effective_sort_flag and rename_script_path: 
            if os.path.isabs(rename_script_path) or os.path.dirname(rename_script_path):
                actual_rename_script_base_path_str = str(Path(rename_script_path).resolve())
            else: 
                temp_path_obj = eff_output_dir / rename_script_path
                actual_rename_script_base_path_str = str(temp_path_obj.resolve())
            
            if self._debug: 
                logging.debug(f"process_files: Determined rename script base path string: {actual_rename_script_base_path_str}")
            
            # **NEW: Purge non-existent files from script (unless --noremove)**
            if not noremove_from_script and not reset_rename_script:
                if os.path.exists(actual_rename_script_base_path_str):
                    logging.info(f"Cleaning rename script: removing entries for non-existent files...")
                    purge_stats = purge_rename_script(actual_rename_script_base_path_str, debug=self._debug)
                    if purge_stats['removed_blocks'] > 0:
                        logging.info(f"Removed {purge_stats['removed_blocks']} command blocks for deleted files")
                
                # Also check for .bat version on Windows
                if platform.system() == 'Windows':
                    batch_version = os.path.splitext(actual_rename_script_base_path_str)[0] + '.bat'
                    if os.path.exists(batch_version):
                        purge_stats_batch = purge_rename_script(batch_version, debug=self._debug)
                        if purge_stats_batch['removed_blocks'] > 0:
                            logging.info(f"Removed {purge_stats_batch['removed_blocks']} command blocks from batch script")
            
            # **CHANGED: Pass reset_mode to safe_initialize_rename_scripts**
            initialized_script_paths_map = safe_initialize_rename_scripts(
                actual_rename_script_base_path_str,
                append_mode=True,
                reset_mode=reset_rename_script  # NEW: respect --reset flag
            )
            if self._debug: 
                logging.debug(f"process_files: Rename scripts initialized. Paths map: {initialized_script_paths_map}")
        elif effective_sort_flag and not rename_script_path:
            logging.warning("Sorting is enabled but no rename script path was effectively determined. Rename command generation will be skipped.")


        # --- ROBUST PRE-INITIALIZATION BLOCK ---
        # This block ensures that any heavy, non-thread-safe models are loaded
        # once in the main thread before any parallel workers are created.
        nanonets_config = kwargs.get('nanonets_config')
        if nanonets_config and nanonets_config.get('use_gguf'):
            logging.info("Pre-initializing Nanonets GGUF model to prevent race conditions...")
            try:
                # Get an instance of the extractor. The filename is a placeholder; the config is what matters.
                extractor = self.manager._get_extractor(
                    "pre_init.pdf", 
                    use_nanonets_ocr2=True, 
                    nanonets_config=nanonets_config
                )
                if extractor and hasattr(extractor, '_init_llama_cpp_model'):
                    # Explicitly call the internal lazy-loading method.
                    # This triggers the expensive model download and GPU initialization safely.
                    init_success = extractor._init_llama_cpp_model()
                    if init_success:
                        logging.info("Nanonets GGUF model pre-initialized successfully.")
                    else:
                        logging.error("Failed to pre-initialize the Nanonets GGUF model. Processing may fail.")
                else:
                    logging.warning("Could not retrieve a valid Nanonets extractor for pre-initialization.")
            except Exception as e:
                logging.error(f"An error occurred during GGUF model pre-initialization: {e}", exc_info=self._debug)
        # --- END OF PRE-INITIALIZATION BLOCK ---

        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            
            

            futures = {}
            for input_file_item in input_files:
                if shutdown_flag.is_set():
                    logging.info(f"Shutdown initiated, no more files will be submitted. {len(futures)} already submitted.")
                    break
                
                if self._debug: 
                    logging.debug(f"process_files: Submitting job for file: {input_file_item}")
                
                futures[executor.submit(
                    self._process_single_file,
                    input_file_item, 
                    str(eff_output_dir), 
                    method, ocr_method, password,
                    extract_tables, force_ocr, noskip, 
                    effective_sort_flag, 
                    actual_rename_script_base_path_str,
                    initialized_script_paths_map,          
                    llm_provider_instance, 
                    temperature,
                    max_tokens,
                    **kwargs
                )] = input_file_item
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Overall File Processing", unit="file"):
                if shutdown_flag.is_set() and not future.done(): 
                    if self._debug: 
                        logging.debug(f"process_files: Cancelling future for {futures[future]} due to shutdown.")
                    future.cancel()
                
                input_file_path_completed = futures[future]
                try:
                    # **NEW: Add timeout to future.result() to prevent indefinite hangs**
                    # Timeout: 30 minutes per file (adjust as needed)
                    single_file_result = future.result(timeout=1800)  
                    
                    if self._debug: 
                        logging.debug(f"process_files: Result for '{input_file_path_completed}': "
                                    f"Success={single_file_result.get('success')}, "
                                    f"Skipped={single_file_result.get('skipped')}")
                    
                    final_results['processed'][input_file_path_completed] = single_file_result
                    if single_file_result.get('skipped'):
                        counters['skipped'] += 1
                        final_results['skipped'].append(input_file_path_completed)
                    elif single_file_result.get('success'):
                        counters['processed_ok'] += 1
                        if single_file_result.get('renamed_info'): 
                            counters['sorted_commands_generated'] +=1
                        elif effective_sort_flag:
                            counters['sort_failed_metadata'] +=1
                    else:
                        counters['failed_extraction'] += 1
                        final_results['failed'].append({'file': input_file_path_completed, 'error': single_file_result.get('error', 'Unknown processing error')})
                
                except FutureTimeoutError:
                    # File processing took too long
                    error_msg = f"Processing timed out after 30 minutes"
                    logging.error(f"{input_file_path_completed}: {error_msg}")
                    final_results['failed'].append({'file': input_file_path_completed, 'error': error_msg})
                    counters['failed_extraction'] += 1
                except Exception as e_future:
                    logging.error(f"Exception processing future for {input_file_path_completed}: {e_future}", exc_info=self._debug)
                    final_results['failed'].append({'file': input_file_path_completed, 'error': str(e_future)})
                    counters['failed_extraction'] += 1
            
            final_results['counters'] = counters 
            logging.info("\n--- Processing Summary ---")
            logging.info(f"Total files attempted: {counters['total']}")
            logging.info(f"Successfully processed (text extracted/loaded): {counters['processed_ok']}")
            if kwargs.get('sort_arg_from_main', False):
                logging.info(f"Rename commands generated for: {counters['sorted_commands_generated']}")
                logging.info(f"Sorting failed (metadata issues/LLM errors): {counters['sort_failed_metadata']}")
            logging.info(f"Skipped (already processed): {counters['skipped']}")
            logging.info(f"Failed extraction/critical processing errors: {counters['failed_extraction']}")
            if final_results['failed']:
                logging.warning("Details for files that failed critical processing:")
                for fail_info in final_results['failed']:
                    logging.warning(f"  - {fail_info['file']}: {fail_info['error']}")
            logging.info("------------------------")

            return final_results