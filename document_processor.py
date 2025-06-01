# document_processor.py
import datetime
import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Imports from our custom modules
from extraction_manager import ExtractionManager # Direct import
from utils import (                             # Direct import
    file_lock, parse_metadata, sanitize_filename, 
    validate_and_fix_year, add_rename_command, 
    shutdown_flag, valid_author_name, initialize_rename_scripts
)
from llm_providers import ( # Ensure this is direct import if document_processor.py is at root
    extract_metadata as llm_extract_metadata, 
    sort_author_names,
    get_llm_provider, 
    LLMProvider,
    OllamaProvider, OpenAIProvider, GroqProvider, CohereProvider
)

# For _extract_pdf_metadata and _extract_epub_metadata (can also be moved to utils if general enough)
# For now, keeping them as private helpers if their usage is tied to DocumentProcessor's workflow.
try:
    from pypdf import PdfReader
    # from PyPDF2 import PdfReader # Older PyPDF2
except ImportError:
    PdfReader = None # Will be checked before use
try:
    from ebooklib import epub
except ImportError:
    epub = None # Will be checked before use


class DocumentProcessor:
    """Main document processing coordinator"""
    
    def __init__(self, debug: bool = False):
        self.manager = ExtractionManager(debug=debug)
        self._debug = debug
        # TableExtractor is now instantiated within PDFExtractor, so not needed here.
        # self._table_extractor = TableExtractor(ImportCache()) 

    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata using PyPDF"""
        metadata = {}
        if PdfReader is None:
            logging.warning("pypdf not available, cannot extract PDF metadata.")
            return metadata
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                doc_info = reader.metadata
                if doc_info:
                    for key, value in doc_info.items():
                        # Ensure values are serializable if metadata is dumped to JSON
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            metadata[key.lstrip('/')] = value 
                        else:
                            metadata[key.lstrip('/')] = str(value) 
        except Exception as e:
            logging.warning(f"Failed to extract PDF metadata for {file_path}: {e}")
        return metadata

    def _extract_epub_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract EPUB metadata using ebooklib"""
        metadata = {}
        if epub is None:
            logging.warning("ebooklib not available, cannot extract EPUB metadata.")
            return metadata
        try:
            book = epub.read_epub(file_path)
            # Standard DC metadata fields
            dc_fields = ['title', 'creator', 'subject', 'description', 'publisher', 
                         'contributor', 'date', 'type', 'format', 'identifier', 
                         'source', 'language', 'relation', 'coverage', 'rights']
            for field in dc_fields:
                meta_values = book.get_metadata('DC', field)
                if meta_values:
                    # Store as list if multiple, single if one, else None
                    processed_values = [item[0] for item in meta_values if item and item[0]]
                    if len(processed_values) == 1:
                        metadata[field] = processed_values[0]
                    elif len(processed_values) > 1:
                        metadata[field] = processed_values
            # Specifically map creator to authors for consistency with PDF
            if 'creator' in metadata:
                metadata['authors'] = metadata['creator']
                if not isinstance(metadata['authors'], list):
                    metadata['authors'] = [metadata['authors']]
        except Exception as e:
            logging.warning(f"Failed to extract EPUB metadata for {file_path}: {e}")
        return metadata

    def _extract_document_inherent_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document's inherent metadata (not from LLM)."""
        metadata = {}
        try:
            file_info = Path(file_path)
            metadata.update({
                'filename': file_info.name,
                'size': file_info.stat().st_size,
                'modified_timestamp': file_info.stat().st_mtime, # Keep timestamp for potential use
                'modified_datetime': datetime.datetime.fromtimestamp(file_info.stat().st_mtime).isoformat()
            })
            
            file_suffix_lower = file_info.suffix.lower()
            if file_suffix_lower == '.pdf':
                metadata.update(self._extract_pdf_metadata(file_path))
            elif file_suffix_lower == '.epub':
                metadata.update(self._extract_epub_metadata(file_path))
            # Add more for other types if specific library-based metadata extraction is available
            # e.g., for MOBI using mobi-python library's header info.
                
        except Exception as e:
            if self._debug:
                logging.error(f"Inherent metadata extraction failed for {file_path}: {e}")
        return metadata
        
    def _get_unique_output_path(self, input_file: str, output_dir_base: Optional[str] = None, noskip: bool = False) -> str:
        """
        Generate output path for a given input file's text output.
        Ensures output is directly in output_dir_base, not preserving subdirs from input.
        """
        input_basename = os.path.basename(input_file)
        input_stem = Path(input_basename).stem
        
        eff_output_dir = Path(output_dir_base or '.').resolve() # output_dir_base is now always a string or '.'
        eff_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_name = f"{input_stem}.txt"
        output_path = eff_output_dir / output_name
        
        if noskip and output_path.exists():
            counter = 1
            while True:
                output_name = f"{input_stem}_{counter}.txt"
                output_path = eff_output_dir / output_name
                if not output_path.exists():
                    if self._debug: logging.debug(f"_get_unique_output_path: Using unique output path for text file: {output_path}")
                    break
                counter += 1
        elif not noskip and output_path.exists():
            if self._debug: logging.debug(f"_get_unique_output_path: Output text file {output_path} exists and noskip is False.")
        elif self._debug:
            logging.debug(f"_get_unique_output_path: Standard output path for text file: {output_path} (noskip={noskip}, exists={output_path.exists()})")
            
        return str(output_path)


    def _process_single_file(self, input_file: str,
                             output_dir_base: Optional[str], # This is a resolved string path
                             method: Optional[str],
                             ocr_method: Optional[str],
                             password: Optional[str],
                             extract_tables: bool,
                             force_ocr: bool,
                             noskip: bool,
                             sort: bool, # This reflects whether sorting should be attempted
                             rename_script_base_path_for_unparseables: Optional[str],
                             initialized_rename_script_paths: Optional[Dict[str, Optional[str]]],
                             llm_provider_instance: Optional[LLMProvider], 
                             temperature: float, 
                             max_tokens: int,     
                             **kwargs) -> Dict[str, Any]:
        
        # Ensure output_dir_base is a string, as it's used with os.path.join and Path
        # process_files now passes it as str(eff_output_dir)
        current_output_dir_base = str(output_dir_base) if output_dir_base else '.'

        if self._debug: 
            logging.debug(f"DS_Proc: Starting _process_single_file for: '{input_file}'")
            logging.debug(f"DS_Proc: Effective output_dir_base='{current_output_dir_base}'")
            logging.debug(f"DS_Proc: noskip={noskip}, sort={sort}")
            logging.debug(f"DS_Proc: rename_script_base_path_for_unparseables='{rename_script_base_path_for_unparseables}'")
            logging.debug(f"DS_Proc: initialized_rename_script_paths='{initialized_rename_script_paths}'")

        result: Dict[str, Any] = {
            'success': False, 'text': '', 'output_path': None, 'skipped': False, 
            'error': None, 'tables': [], 'metadata_llm': None, 'renamed_info': None,
            'input_file': input_file, 'inherent_metadata': {}
        }
        
        if shutdown_flag.is_set():
            result['error'] = "Processing aborted due to shutdown signal"
            if self._debug: logging.debug(f"DS_Proc: Shutdown signal detected for '{input_file}'. Aborting.")
            return result

        try:
            is_source_txt_file = input_file.lower().endswith(('.txt', '.md')) # Treat .md like .txt for direct reading
            text_loaded_from_file = False
            actual_text_content_path_for_rename: Optional[str] = None # Path of the text content used for LLM

            if is_source_txt_file:
                if self._debug: logging.debug(f"DS_Proc: Input '{input_file}' is a direct text type file.")
                # For a source .txt/.md, its "output path" (conceptually, before sorting) is its place in the output_dir.
                # No _1 suffix should be generated for the source file itself due to noskip for THIS initial step.
                # _get_unique_output_path would be used if saving an *extraction of* this, but we're reading directly.
                # However, for consistency, result['output_path'] should still point to where this text *content* is considered to be.
                # If output_dir_base is '.', then Path(input_file).resolve() is appropriate.
                # If output_dir_base is different, it's where it *would* be if copied/processed to that dir.
                # Let's set result['output_path'] to the original file's path if it's directly read.
                # This path will then be passed to add_rename_command as `actual_text_content_file_path`.
                
                result['output_path'] = str(Path(current_output_dir_base) / os.path.basename(input_file))
                actual_text_content_path_for_rename = input_file # The source IS the content

                if not noskip and os.path.exists(result['output_path']) and Path(input_file).resolve() != Path(result['output_path']).resolve() and not sort:
                    # This case is tricky: if output_path (e.g. ./file.txt) exists, and input is elsewhere (../file.txt)
                    # and we are not sorting, we might skip. For simplicity, if it's a source.txt, and --noskip is false,
                    # and we are not sorting, we usually just use it.
                    # The main `noskip` check below handles the general case.
                    # If it's a source .txt, and `sort` is false, and `noskip` is false,
                    # we shouldn't really "skip" it, we just don't do much with it.
                    # The outer `process_files` loop's skip logic is more for "output text already exists".
                    # For now, if it's a source.txt, we always "process" it (read content).
                    pass


                try:
                    if self._debug: logging.debug(f"DS_Proc: Directly reading content from source text file: '{input_file}'")
                    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
                        result['text'] = f.read()
                    if result['text'].strip():
                        result['success'] = True
                        text_loaded_from_file = True 
                        if self._debug: logging.debug(f"DS_Proc: Successfully read content from source file '{input_file}'. Length: {len(result['text'])}")
                    else:
                        result['error'] = f"Source text file '{input_file}' is empty."
                        result['success'] = False
                        if self._debug: logging.warning(result['error'])
                except Exception as e_read_txt:
                    result['error'] = f"Failed to read source text file '{input_file}': {e_read_txt}"
                    result['success'] = False
                    logging.error(result['error'], exc_info=self._debug)
            
            # If not a source text file, or if reading source text file failed somehow:
            if not text_loaded_from_file:
                if self._debug and is_source_txt_file : logging.debug(f"DS_Proc: Direct read of source text file '{input_file}' failed or was empty, proceeding to manager.extract as fallback.")
                
                # This is the path where the extracted .txt will be saved by the manager.
                # _get_unique_output_path handles --noskip correctly by potentially adding _1, _2, etc.
                # if the base target (e.g. file.txt) already exists in current_output_dir_base
                determined_extraction_output_path = self._get_unique_output_path(input_file, current_output_dir_base, noskip)
                result['output_path'] = determined_extraction_output_path # Update result['output_path']
                actual_text_content_path_for_rename = determined_extraction_output_path # This is where the text lives

                if not noskip and os.path.exists(determined_extraction_output_path):
                    if self._debug: logging.debug(f"DS_Proc: Extraction output '{determined_extraction_output_path}' exists and noskip=False for '{input_file}'.")
                    if not sort:
                        logging.info(f"Skipping '{input_file}' - text output '{determined_extraction_output_path}' exists, noskip=False, and sort=False.")
                        result['skipped'] = True
                        result['success'] = True 
                        result['text'] = ""
                        # Try to get inherent metadata even if skipped
                        try: result['inherent_metadata'] = self._extract_document_inherent_metadata(input_file)
                        except Exception as e_meta:
                            if self._debug: logging.debug(f"Could not get inherent metadata for skipped file '{input_file}': {e_meta}")
                        return result # Early exit for skip
                    else: # noskip=False, output exists, but sort=True, so load text
                        if self._debug: logging.debug(f"DS_Proc: noskip=False, sort=True. Loading text from existing '{determined_extraction_output_path}' for '{input_file}'.")
                        try:
                            with open(determined_extraction_output_path, 'r', encoding='utf-8') as f:
                                result['text'] = f.read()
                            if result['text'].strip():
                                result['success'] = True
                                text_loaded_from_file = True
                                if self._debug: logging.debug(f"DS_Proc: Reused text from '{determined_extraction_output_path}' for sorting '{input_file}'. Length: {len(result['text'])}")
                            else:
                                logging.warning(f"Existing text file '{determined_extraction_output_path}' for '{input_file}' is empty. Re-extracting.")
                                result['success'] = False # Force re-extraction
                        except Exception as e_read:
                            logging.warning(f"Could not read '{determined_extraction_output_path}' for sorting '{input_file}': {e_read}. Re-extracting.")
                            result['success'] = False # Force re-extraction

            # Call manager.extract only if text wasn't loaded directly or from a pre-existing file
            if not text_loaded_from_file:
                if self._debug: logging.debug(f"DS_Proc: Calling manager.extract for '{input_file}' to output at '{actual_text_content_path_for_rename}'.")
                extraction_result = self.manager.extract(
                    input_path=input_file,
                    output_path=actual_text_content_path_for_rename, # Manager writes here
                    method=method, ocr_method=ocr_method, password=password,
                    extract_tables=extract_tables, force_ocr=force_ocr, 
                    temperature=temperature, max_tokens=max_tokens, 
                    **kwargs
                )
                result['text'] = extraction_result.get('text', '')
                result['success'] = extraction_result.get('success', False)
                result['tables'] = extraction_result.get('tables', [])
                if extraction_result.get('error'):
                    result['error'] = extraction_result.get('error')
                
                if result['success']:
                    result['output_path'] = actual_text_content_path_for_rename # Confirm output path
                    if self._debug: logging.debug(f"DS_Proc: manager.extract successful for '{input_file}'. Text length: {len(result['text'])}.")
                elif self._debug:
                     logging.debug(f"DS_Proc: manager.extract failed for '{input_file}'. Error: {result.get('error')}")


            # --- Sorting Logic ---
            if result['success'] and result['text']:
                if self._debug: logging.debug(f"DS_Proc: Text available for '{input_file}'. Sort flag: {sort}.")
                if sort:
                    if llm_provider_instance and initialized_rename_script_paths:
                        if self._debug: logging.debug(f"DS_Proc: Processing LLM metadata for sorting: '{input_file}' using text from '{actual_text_content_path_for_rename}'")
                        try:
                            metadata_llm_str = llm_extract_metadata(
                                text=result["text"], filename=input_file,
                                llm_provider_arg=llm_provider_instance,
                                verbose=self._debug 
                            )
                            if metadata_llm_str:
                                # (The rest of the metadata parsing, author sorting, and add_rename_command call logic remains largely the same as previous corrected version)
                                # Ensure `add_rename_command` is called with `actual_text_content_path_for_rename`
                                # ... (LLM metadata parsing and author sorting) ...
                                if self._debug: logging.debug(f"DS_Proc: LLM returned for metadata: '{metadata_llm_str[:150]}...' for '{input_file}'")
                                parsed_llm_meta = parse_metadata(metadata_llm_str, filename=os.path.basename(input_file))
                                if parsed_llm_meta:
                                    if self._debug: logging.debug(f"DS_Proc: Parsed LLM metadata: {parsed_llm_meta} for '{input_file}'")
                                    result['metadata_llm'] = parsed_llm_meta.copy()
                                    
                                    year_str_from_llm = parsed_llm_meta.get('year')
                                    parsed_llm_meta['year'] = validate_and_fix_year(year_str_from_llm)
                                    if self._debug: logging.debug(f"DS_Proc: Validated year: '{parsed_llm_meta['year']}' for '{input_file}'")
                                    
                                    author_to_sort = parsed_llm_meta.get('author', "UnknownAuthor")
                                    corrected_author = "UnknownAuthor" 
                                    
                                    if author_to_sort.lower() not in ["lastname firstname", "unknownauthor", "unknown author", "main author: lastname firstname", "none", ""] and valid_author_name(author_to_sort):
                                        if self._debug: logging.debug(f"DS_Proc: Attempting to sort author name '{author_to_sort}' for '{input_file}'")
                                        corrected_author = sort_author_names(
                                            author_names_input=author_to_sort, provider_arg=llm_provider_instance,
                                            temperature=0.2, max_tokens=100, verbose=self._debug, 
                                            filename_for_logging=input_file,
                                            model_name_arg=llm_provider_instance.model_name if llm_provider_instance else None,
                                            api_key_arg=llm_provider_instance.api_key if llm_provider_instance else None )
                                        if self._debug: logging.debug(f"DS_Proc: Sorted author name: '{corrected_author}' for '{input_file}'")
                                    else:
                                        if self._debug: logging.debug(f"DS_Proc: Author ('{author_to_sort}') is placeholder or invalid; using as-is or 'UnknownAuthor' for '{input_file}'.")
                                        corrected_author = author_to_sort if valid_author_name(author_to_sort) else "UnknownAuthor"
                                    parsed_llm_meta['author'] = corrected_author

                                    if not valid_author_name(corrected_author) or \
                                       not parsed_llm_meta.get('title') or \
                                       parsed_llm_meta.get('title','').lower() in ['unknowntitle', 'unknown', 'the full title', '']:
                                        logging.warning(f"LLM metadata invalid for sorting '{input_file}': Author='{corrected_author}', Title='{parsed_llm_meta.get('title')}'. Adding to unparseables.")
                                        if rename_script_base_path_for_unparseables:
                                            unparseables_path = Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst"
                                            with file_lock:
                                                with open(unparseables_path, "a", encoding='utf-8') as f_unp:
                                                    f_unp.write(f"{input_file} - Invalid LLM metadata (Author: {corrected_author}, Title: {parsed_llm_meta.get('title')})\n")
                                    else: 
                                        if self._debug: logging.debug(f"DS_Proc: Valid metadata for rename. Author='{corrected_author}', Year='{parsed_llm_meta['year']}', Title='{parsed_llm_meta['title']}' for '{input_file}'")
                                        rename_info_dict = add_rename_command(
                                            rename_script_paths=initialized_rename_script_paths, 
                                            source_path_original_file=input_file, # The original file being processed
                                            actual_text_content_file_path=actual_text_content_path_for_rename, # Path to the .txt content
                                            target_dir_name=parsed_llm_meta['author'], 
                                            new_filename_base=f"{parsed_llm_meta['year']} {parsed_llm_meta['title']}",
                                            output_dir_for_sorted_files=str(current_output_dir_base), 
                                            debug=self._debug
                                        )
                                        result["renamed_info"] = rename_info_dict
                                        if self._debug and rename_info_dict: logging.debug(f"DS_Proc: Rename command generated for {input_file}. Info: {rename_info_dict}")
                                        elif self._debug and not rename_info_dict: logging.warning(f"DS_Proc: add_rename_command returned None for {input_file}.")
                                else: 
                                    logging.warning(f"Failed to parse LLM metadata for '{input_file}' from string: '{metadata_llm_str[:100]}...'")
                                    if rename_script_base_path_for_unparseables:
                                        with file_lock: (Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst").open("a", encoding='utf-8').write(f"{input_file} - Failed to parse LLM metadata from string\n")
                            else: 
                                logging.warning(f"LLM did not return metadata string for '{input_file}'")
                                if rename_script_base_path_for_unparseables:
                                    with file_lock: (Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst").open("a", encoding='utf-8').write(f"{input_file} - LLM returned no metadata string\n")
                        except Exception as e_sort:
                            logging.error(f"Error during LLM sorting process for '{input_file}': {e_sort}", exc_info=self._debug)
                            result["error"] = f"Sorting error: {e_sort}"
                            if rename_script_base_path_for_unparseables:
                                with file_lock: (Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst").open("a", encoding='utf-8').write(f"{input_file} - Error in sorting logic: {e_sort}\n")
                    elif sort: # sort=True but LLM provider or script paths missing
                        if not llm_provider_instance and self._debug: logging.warning(f"DS_Proc: Sort is True for '{input_file}', but LLM provider instance is missing. Skipping sort.")
                        if not initialized_rename_script_paths and self._debug: logging.warning(f"DS_Proc: Sort is True for '{input_file}', but initialized_rename_script_paths is None. Skipping rename command generation.")
            
            # (Rest of the error handling and metadata extraction for result['inherent_metadata'])
            elif not result['success']: 
                if not result.get("error"): result['error'] = "Text extraction failed or produced no text."
                if self._debug: logging.debug(f"DS_Proc: Text extraction failed for '{input_file}'. Error: {result['error']}")
            elif not result['text']: 
                 result['error'] = "Text extraction successful but produced empty text."
                 result['success'] = False 
                 if self._debug: logging.debug(f"DS_Proc: Text extraction yielded empty text for '{input_file}'. Marked as failure.")

            if not result.get('skipped'):
                try:
                    result['inherent_metadata'] = self._extract_document_inherent_metadata(input_file)
                    if self._debug: logging.debug(f"DS_Proc: Extracted inherent metadata for '{input_file}': {str(result['inherent_metadata'])[:200]}...")
                except Exception as e_meta_final:
                    if self._debug: logging.debug(f"DS_Proc: Could not get inherent metadata for '{input_file}' at the end: {e_meta_final}")

        except Exception as e: # Catch-all for _process_single_file
            result['error'] = f"Overall processing of '{input_file}' failed in _process_single_file: {str(e)}"
            logging.error(result['error'], exc_info=self._debug)
            result['success'] = False
            if sort and rename_script_base_path_for_unparseables: 
                try:
                    with file_lock: (Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst").open("a", encoding='utf-8').write(f"{input_file} - Overall Error in _process_single_file: {str(e)}\n")
                except Exception as e_unp: logging.error(f"Failed to write to unparseables.lst: {e_unp}")
        
        if self._debug: 
            logging.debug(f"DS_Proc: Finished _process_single_file for: '{input_file}', "
                          f"Success: {result['success']}, Error: {result.get('error')}, "
                          f"Renamed: {bool(result.get('renamed_info'))}, "
                          f"Skipped: {result.get('skipped')}, "
                          f"Final text content path in result: {result.get('output_path')}")
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
                      rename_script_path: Optional[str] = None, # This is the base name/path from args
                      llm_provider_arg: Optional[Any] = None, 
                      temperature: float = 0.7,       
                      max_tokens: int = 300,      
                      **kwargs) -> Dict[str, Any]:
        
        if self._debug:
            logging.debug(f"process_files: Starting. Total input_files: {len(input_files)}")
            logging.debug(f"process_files: output_dir='{output_dir}', sort='{sort}', rename_script_path='{rename_script_path}'")

        final_results: Dict[str, Any] = {'processed': {}, 'failed': [], 'skipped': []}
        counters = {'total': len(input_files), 'processed_ok': 0, 'skipped': 0, 
                    'failed_extraction': 0, 'sorted_commands_generated': 0, 'sort_failed_metadata': 0}

        eff_output_dir = Path(output_dir or '.').resolve()
        eff_output_dir.mkdir(parents=True, exist_ok=True)
        if self._debug: logging.debug(f"process_files: Effective output directory: {eff_output_dir}")

        llm_provider_instance: Optional[LLMProvider] = None
        if sort: 
            if isinstance(llm_provider_arg, LLMProvider):
                llm_provider_instance = llm_provider_arg
            elif isinstance(llm_provider_arg, str) or llm_provider_arg is None:
                provider_name_str = llm_provider_arg if isinstance(llm_provider_arg, str) else "ollama"
                try:
                    llm_model_from_kwargs = kwargs.get('llm_model')
                    api_key_from_kwargs = kwargs.get('api_key')
                    if self._debug: 
                        logging.debug(f"process_files: Initializing LLM provider '{provider_name_str}' "
                                      f"with model='{llm_model_from_kwargs}', api_key_present={bool(api_key_from_kwargs)}")
                    llm_provider_instance = get_llm_provider(provider_name_str, llm_model_from_kwargs, api_key_from_kwargs)
                except Exception as e:
                    logging.error(f"Failed to initialize LLM provider '{provider_name_str}' for sorting: {e}. Sorting may be impacted.")
            else:
                logging.warning(f"Invalid llm_provider_arg for process_files: {type(llm_provider_arg)}. Sorting may be impacted.")

        actual_max_workers = max_workers
        if actual_max_workers is None:
            is_local_ollama_for_sort = sort and (not llm_provider_instance or isinstance(llm_provider_instance, OllamaProvider))
            cpu_count = os.cpu_count() or 1
            actual_max_workers = min(4, cpu_count) if is_local_ollama_for_sort else min(8, cpu_count)
        
        logging.info(f"Using up to {actual_max_workers} worker threads.")

        # ----MODIFIED SECTION for rename script initialization----
        # This is the base string path for the rename script (e.g., "rename_commands.sh")
        # It's used to determine the parent directory for unparseables.lst
        actual_rename_script_base_path_str: Optional[str] = None
        # This will hold the dictionary like {'bash_script': '/path/to/script.sh', 'batch_script': '/path/to/script.bat'}
        initialized_script_paths_map: Optional[Dict[str, Optional[str]]] = None

        if sort and rename_script_path: # rename_script_path is from args (e.g., "rename_commands.sh")
            if os.path.isabs(rename_script_path) or os.path.dirname(rename_script_path):
                # If it's an absolute path or contains directory components, resolve it directly
                actual_rename_script_base_path_str = str(Path(rename_script_path).resolve())
            else:
                # If it's just a filename, make it relative to the effective output directory
                temp_path_obj = eff_output_dir / rename_script_path 
                actual_rename_script_base_path_str = str(temp_path_obj.resolve())
            
            if self._debug: logging.debug(f"process_files: Determined rename script base path: {actual_rename_script_base_path_str}")

            # Initialize rename scripts ONCE. This creates/clears the files.
            # It returns a dictionary mapping script type (e.g., 'bash_script') to its actual path.
            initialized_script_paths_map = initialize_rename_scripts(actual_rename_script_base_path_str)
            if self._debug: logging.debug(f"process_files: Rename scripts initialized. Paths map: {initialized_script_paths_map}")

        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            futures = {}
            for input_file_item in input_files:
                if shutdown_flag.is_set():
                    logging.info(f"Shutdown initiated, no more files will be submitted. {len(futures)} already submitted.")
                    break
                
                if self._debug: logging.debug(f"process_files: Submitting job for file: {input_file_item}")
                futures[executor.submit(
                    self._process_single_file,
                    input_file_item, 
                    str(eff_output_dir), # Pass resolved output dir as string
                    method, ocr_method, password,
                    extract_tables, force_ocr, noskip, sort, 
                    actual_rename_script_base_path_str,    # Pass the base string path for unparseables.lst logic
                    initialized_script_paths_map,          # Pass the DICT of initialized script paths
                    llm_provider_instance, temperature, max_tokens, **kwargs
                )] = input_file_item
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Overall File Processing", unit="file"):
                if shutdown_flag.is_set() and not future.done(): 
                    if self._debug: logging.debug(f"process_files: Cancelling future for {futures[future]} due to shutdown.")
                    future.cancel()
                
                input_file_path_completed = futures[future]
                try:
                    single_file_result = future.result() 
                    if self._debug: logging.debug(f"process_files: Result for {input_file_path_completed}: Success={single_file_result.get('success')}, Skipped={single_file_result.get('skipped')}, RenamedInfo={bool(single_file_result.get('renamed_info'))}")
                    
                    final_results['processed'][input_file_path_completed] = single_file_result
                    if single_file_result.get('skipped'):
                        counters['skipped'] += 1
                        final_results['skipped'].append(input_file_path_completed)
                    elif single_file_result.get('success'):
                        counters['processed_ok'] += 1
                        # Check if rename command was generated
                        if single_file_result.get('renamed_info'): 
                            counters['sorted_commands_generated'] +=1
                            if self._debug: logging.debug(f"process_files: Incremented sorted_commands_generated for {input_file_path_completed}")
                        # If sort was True, metadata was extracted, but no rename info (e.g., invalid author/title)
                        elif sort and single_file_result.get('metadata_llm') and not single_file_result.get('renamed_info'):
                            counters['sort_failed_metadata'] +=1
                            if self._debug: logging.debug(f"process_files: Incremented sort_failed_metadata for {input_file_path_completed} (metadata present, but no rename_info)")
                        elif sort and not single_file_result.get('metadata_llm') and not single_file_result.get('renamed_info'):
                            # This case means sorting was attempted, text was extracted, but metadata parsing failed OR LLM returned nothing for metadata.
                            counters['sort_failed_metadata'] +=1
                            if self._debug: logging.debug(f"process_files: Incremented sort_failed_metadata for {input_file_path_completed} (no metadata_llm and no rename_info, sort was True)")

                    else: # Not skipped, not success
                        counters['failed_extraction'] += 1
                        final_results['failed'].append({'file': input_file_path_completed, 'error': single_file_result.get('error', 'Unknown processing error')})
                        if self._debug: logging.debug(f"process_files: Incremented failed_extraction for {input_file_path_completed}")
                        # If sorting was on, and extraction failed, it's an extraction failure, not a sort metadata failure.
                        # If sort was on, but the file was processed_ok=False, and an error occurred during the sorting phase itself (rare after success=False),
                        # it would be logged by _process_single_file; here it's already counted as failed_extraction.

                except Exception as e_future:
                    logging.error(f"Exception processing future for {input_file_path_completed}: {e_future}", exc_info=self._debug)
                    final_results['failed'].append({'file': input_file_path_completed, 'error': str(e_future)})
                    counters['failed_extraction'] += 1 
                    if self._debug: logging.debug(f"process_files: Exception for {input_file_path_completed}, incremented failed_extraction.")
        
        logging.info("\n--- Processing Summary ---")
        logging.info(f"Total files attempted: {counters['total']}")
        logging.info(f"Successfully processed (text extracted/loaded): {counters['processed_ok']}")
        if sort:
            logging.info(f"Rename commands generated for: {counters['sorted_commands_generated']}")
            # Adjusting how sort_failed_metadata is interpreted. It's about files that had text but failed the sort/rename step.
            # Files that failed extraction are not sort failures.
            # Total files that *could* have been sorted = processed_ok.
            # Sort failures = processed_ok - sorted_commands_generated (if sort was True for them).
            # The current counters['sort_failed_metadata'] counts specific metadata/LLM issues during sort for files that otherwise had text.
            logging.info(f"Sorting failed (metadata issues/LLM errors for successfully extracted files): {counters['sort_failed_metadata']}")
        logging.info(f"Skipped (e.g., output existed and not --noskip): {counters['skipped']}")
        logging.info(f"Failed extraction/critical processing errors: {counters['failed_extraction']}")
        if final_results['failed']:
            logging.warning("Details for failed files:")
            for fail_info in final_results['failed']:
                logging.warning(f"  - {fail_info['file']}: {fail_info['error']}")
        logging.info("------------------------")
        
        return final_results