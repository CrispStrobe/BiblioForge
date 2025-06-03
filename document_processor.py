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
import traceback # For more detailed error logging if needed

# --- Corrected Imports for sibling modules ---
from extraction_manager import ExtractionManager
from utils import (
    file_lock, parse_metadata, sanitize_filename, 
    validate_and_fix_year, add_rename_command, 
    shutdown_flag, valid_author_name, initialize_rename_scripts
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

    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = {}
        if PdfReader is None:
            if self._debug: logging.debug(f"_extract_pdf_metadata: pypdf not available for {file_path}.")
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
            metadata.update({
                'filename': file_info.name,
                'size': file_info.stat().st_size,
                'modified_datetime': datetime.datetime.fromtimestamp(file_info.stat().st_mtime).isoformat()
            })
            
            file_suffix_lower = file_info.suffix.lower()
            if file_suffix_lower == '.pdf':
                metadata.update(self._extract_pdf_metadata(file_path))
            elif file_suffix_lower == '.epub':
                metadata.update(self._extract_epub_metadata(file_path))
            
            if self._debug: logging.debug(f"_extract_document_inherent_metadata: Extracted for {file_path}: {str(metadata)[:200]}...")
        except Exception as e:
            if self._debug: logging.error(f"_extract_document_inherent_metadata: Failed for {file_path}: {e}", exc_info=self._debug)
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
                             # temperature and max_tokens are now expected to be in **kwargs if needed by llm_extract_metadata etc.
                             # However, llm_extract_metadata was already modified to take them explicitly.
                             temperature_for_metadata: float, # Explicit for metadata
                             max_tokens_for_metadata: int,    # Explicit for metadata
                             **kwargs_from_main) -> Dict[str, Any]: # Catches all other CLI args like provider-specific ones
        
        if self._debug: 
            logging.debug(f"DS_Proc: Starting _process_single_file for: '{input_file}'")
            logging.debug(f"DS_Proc: Effective current_output_dir_base_str='{current_output_dir_base_str}'")
            logging.debug(f"DS_Proc: noskip={noskip}, effective_sort_flag={effective_sort_flag}")
            logging.debug(f"DS_Proc: rename_script_base_path_for_unparseables='{rename_script_base_path_for_unparseables}'")
            logging.debug(f"DS_Proc: initialized_rename_script_paths='{initialized_rename_script_paths}'")
            logging.debug(f"DS_Proc: llm_provider_instance type='{type(llm_provider_instance)}', model='{llm_provider_instance.model_name if llm_provider_instance else 'N/A'}'")
            logging.debug(f"DS_Proc: temperature_for_metadata={temperature_for_metadata}, max_tokens_for_metadata={max_tokens_for_metadata}")
            logging.debug(f"DS_Proc: Received **kwargs_from_main: {kwargs_from_main}")

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
            is_source_direct_text_type = input_file.lower().endswith(('.txt', '.md'))
            text_loaded_from_file = False
            actual_text_content_path_for_llm_and_rename: Optional[str] = None 

            if is_source_direct_text_type:
                if self._debug: logging.debug(f"DS_Proc: Input '{input_file}' is a direct text type. Reading content directly.")
                # The "output_path" in the result dict for a source .txt file refers to its original path
                # as that's where the content is coming from.
                result['output_path'] = input_file 
                actual_text_content_path_for_llm_and_rename = input_file

                try:
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
            
            if not text_loaded_from_file:
                if self._debug and is_source_direct_text_type : 
                    logging.debug(f"DS_Proc: Direct read of source text file '{input_file}' failed or was empty. Manager.extract will not be called for this .txt file.")
                    # If it's a source .txt and read failed, we don't proceed to manager.extract to avoid _1.txt
                    # The result['success'] is already False.
                
                elif not is_source_direct_text_type: # Only do manager.extract for non-direct-text types
                    # This path is where the extracted .txt will be saved by the manager.
                    determined_extraction_output_path = self._get_unique_output_path(input_file, current_output_dir_base_str, noskip)
                    result['output_path'] = determined_extraction_output_path 
                    actual_text_content_path_for_llm_and_rename = determined_extraction_output_path

                    if not noskip and os.path.exists(determined_extraction_output_path):
                        if self._debug: logging.debug(f"DS_Proc: Extracted text file '{determined_extraction_output_path}' exists and noskip=False for '{input_file}'.")
                        if not effective_sort_flag: 
                            logging.info(f"Skipping '{input_file}' - output '{determined_extraction_output_path}' exists, noskip=False, and sort=False.")
                            result['skipped'] = True; result['success'] = True; result['text'] = "" 
                            try: result['inherent_metadata'] = self._extract_document_inherent_metadata(input_file)
                            except Exception as e_meta:
                                if self._debug: logging.debug(f"Could not get inherent metadata for skipped file '{input_file}': {e_meta}")
                            return result 
                        else: 
                            if self._debug: logging.debug(f"DS_Proc: noskip=False, sort=True. Loading text from existing '{determined_extraction_output_path}' for '{input_file}'.")
                            try:
                                with open(determined_extraction_output_path, 'r', encoding='utf-8') as f:
                                    result['text'] = f.read()
                                if result['text'].strip():
                                    result['success'] = True; text_loaded_from_file = True
                                    if self._debug: logging.debug(f"DS_Proc: Reused text (len: {len(result['text'])}) from '{determined_extraction_output_path}' for sorting '{input_file}'.")
                                else:
                                    logging.warning(f"Existing '{determined_extraction_output_path}' for '{input_file}' is empty. Re-extracting.")
                                    result['success'] = False 
                            except Exception as e_read:
                                logging.warning(f"Could not read '{determined_extraction_output_path}' for sorting '{input_file}': {e_read}. Re-extracting.")
                                result['success'] = False 
                    
                    if not text_loaded_from_file: # If not skipped and not loaded, then extract
                        if self._debug: logging.debug(f"DS_Proc: Calling manager.extract for '{input_file}' to output at '{actual_text_content_path_for_llm_and_rename}'.")
                        extraction_result = self.manager.extract(
                            input_path=input_file,
                            output_path=actual_text_content_path_for_llm_and_rename, 
                            method=method, ocr_method=ocr_method, password=password,
                            extract_tables=extract_tables, force_ocr=force_ocr, 
                            temperature=temperature_for_metadata, # Pass main temp/tokens
                            max_tokens=max_tokens_for_metadata,   
                            **kwargs_from_main 
                        )
                        result['text'] = extraction_result.get('text', '')
                        result['success'] = extraction_result.get('success', False)
                        result['tables'] = extraction_result.get('tables', []) 
                        if extraction_result.get('error'):
                            result['error'] = extraction_result.get('error')
                        
                        if result['success']:
                            result['output_path'] = actual_text_content_path_for_llm_and_rename 
                            if self._debug: logging.debug(f"DS_Proc: manager.extract successful for '{input_file}'. Text length: {len(result['text'])} saved to '{actual_text_content_path_for_llm_and_rename}'.")
                        elif self._debug:
                            logging.debug(f"DS_Proc: manager.extract failed for '{input_file}'. Error: {result.get('error')}")

            # --- Sorting Logic ---
            if result['success'] and result['text'] and actual_text_content_path_for_llm_and_rename:
                if self._debug: logging.debug(f"DS_Proc: Text available for '{input_file}'. Sort flag: {effective_sort_flag}. Text from: '{actual_text_content_path_for_llm_and_rename}'")
                if effective_sort_flag:
                    if llm_provider_instance: 
                        if not initialized_rename_script_paths:
                            logging.warning(f"DS_Proc: Sort is True for '{input_file}', LLM provider available, but "
                                            "initialized_rename_script_paths is None. Cannot generate rename command script entries.")
                        
                        if self._debug: logging.debug(f"DS_Proc: Processing LLM metadata for sorting: '{input_file}'")
                        try:
                            metadata_llm_str = llm_extract_metadata(
                                text=result["text"], filename=input_file,
                                llm_provider_arg=llm_provider_instance, 
                                verbose=self._debug,
                                temperature_arg=temperature_for_metadata, # Explicitly pass
                                max_tokens_arg=max_tokens_for_metadata,   # Explicitly pass
                                **kwargs_from_main # Pass other configs for get_llm_provider if llm_provider_instance was string
                            )
                            if metadata_llm_str:
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
                                            author_names_input=author_to_sort, 
                                            provider_arg=llm_provider_instance, 
                                            verbose=self._debug, 
                                            filename_for_logging=input_file,
                                            # sort_author_names uses its own defaults for temp/tokens, or can take from kwargs
                                            **kwargs_from_main 
                                        )
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
                                    elif initialized_rename_script_paths: 
                                        if self._debug: logging.debug(f"DS_Proc: Valid metadata for rename. Author='{corrected_author}', Year='{parsed_llm_meta['year']}', Title='{parsed_llm_meta['title']}' for '{input_file}'")
                                        rename_info_dict = add_rename_command(
                                            rename_script_paths=initialized_rename_script_paths, 
                                            source_path_original_file=input_file, 
                                            actual_text_content_file_path=actual_text_content_path_for_llm_and_rename, 
                                            target_dir_name=parsed_llm_meta['author'], 
                                            new_filename_base=f"{parsed_llm_meta['year']} {parsed_llm_meta['title']}",
                                            output_dir_for_sorted_files=str(current_output_dir_base_str), 
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
                    elif effective_sort_flag: # sort=True but LLM provider or script paths missing
                        if not llm_provider_instance and self._debug: logging.warning(f"DS_Proc: Sort is True for '{input_file}', but LLM provider instance is missing. Skipping sort attempt.")
            
            elif not result['success']: 
                if not result.get("error"): result['error'] = "Text extraction or direct read failed/produced no text."
                if self._debug: logging.debug(f"DS_Proc: Text extraction/read failed for '{input_file}'. Error: {result['error']}")
            elif not result['text']:
                 result['error'] = "Text acquisition successful but produced empty text."
                 result['success'] = False 
                 if self._debug: logging.debug(f"DS_Proc: Text acquisition yielded empty text for '{input_file}'. Marked as failure.")

            if not result.get('skipped'):
                try:
                    result['inherent_metadata'] = self._extract_document_inherent_metadata(input_file)
                    if self._debug: logging.debug(f"DS_Proc: Extracted inherent metadata for '{input_file}'.")
                except Exception as e_meta_final:
                    if self._debug: logging.debug(f"DS_Proc: Could not get inherent metadata for '{input_file}' at the end: {e_meta_final}")

        except Exception as e: 
            result['error'] = f"Overall processing of '{input_file}' failed in _process_single_file: {str(e)}"
            logging.error(result['error'], exc_info=self._debug)
            result['success'] = False
            if effective_sort_flag and rename_script_base_path_for_unparseables: 
                try:
                    with file_lock: (Path(rename_script_base_path_for_unparseables).parent / "unparseables.lst").open("a", encoding='utf-8').write(f"{input_file} - Overall Error in _process_single_file: {str(e)}\n")
                except Exception as e_unp: logging.error(f"Failed to write to unparseables.lst: {e_unp}")
        
        if self._debug: 
            logging.debug(f"DS_Proc: Finished _process_single_file for: '{input_file}', "
                          f"Success: {result['success']}, Error: {result.get('error')}, "
                          f"Renamed: {bool(result.get('renamed_info'))}, "
                          f"Skipped: {result.get('skipped')}, "
                          f"Final text content path in result (result['output_path']): {result.get('output_path')}, "
                          f"Actual text content path used for LLM/Rename: {actual_text_content_path_for_llm_and_rename}")
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
                      llm_provider_arg: Optional[Any] = None, 
                      temperature: float = 0.7,       
                      max_tokens: int = 300,
                      # **kwargs will capture all other CLI args like ollama_host, llamacpp_repo_id, etc.
                      # and also sort_arg_from_main if BiblioForge.py passes it
                      **kwargs) -> Dict[str, Any]: 
        
        if self._debug:
            logging.debug(f"process_files: Starting. Total input_files: {len(input_files)}")
            logging.debug(f"process_files: output_dir='{output_dir}', sort (original intent)='{sort}', rename_script_path='{rename_script_path}'")
            logging.debug(f"process_files: Received explicit temperature={temperature}, max_tokens={max_tokens}")
            logging.debug(f"process_files: Received **kwargs: {kwargs}")

        final_results: Dict[str, Any] = {'processed': {}, 'failed': [], 'skipped': [], 'counters': {}}
        counters = {'total': len(input_files), 'processed_ok': 0, 'skipped': 0, 
                    'failed_extraction': 0, 'sorted_commands_generated': 0, 'sort_failed_metadata': 0}

        eff_output_dir = Path(output_dir or '.').resolve()
        eff_output_dir.mkdir(parents=True, exist_ok=True)
        if self._debug: logging.debug(f"process_files: Effective output directory: {eff_output_dir}")

        llm_provider_instance: Optional[LLMProvider] = None
        effective_sort_flag = sort # Start with the user's intent for sorting

        if effective_sort_flag:
            if isinstance(llm_provider_arg, LLMProvider):
                llm_provider_instance = llm_provider_arg
                if self._debug: logging.debug("process_files: Using pre-passed LLMProvider instance.")
            elif isinstance(llm_provider_arg, str) or llm_provider_arg is None:
                provider_name_str = llm_provider_arg if isinstance(llm_provider_arg, str) else "ollama"
                try:
                    # Extract general LLM args from kwargs (passed from BiblioForge.py)
                    # These keys must match what get_llm_provider expects.
                    model_name_for_provider = kwargs.get('llm_model') 
                    api_key_for_provider = kwargs.get('api_key')    
                    
                    # Gather all other provider-specific arguments from kwargs
                    # These keys should match parameters in get_llm_provider
                    provider_specific_constructor_args = {
                        "ollama_host": kwargs.get('ollama_host'),
                        "local_openai_base_url": kwargs.get('local_openai_base_url'),
                        "llamacpp_repo_id": kwargs.get('llamacpp_repo_id'),
                        "llamacpp_gguf_filename": kwargs.get('llamacpp_gguf_filename'),
                        "llamacpp_n_ctx": kwargs.get('llamacpp_n_ctx'),
                        "llamacpp_n_gpu_layers": kwargs.get('llamacpp_n_gpu_layers'),
                        "llamacpp_chat_format": kwargs.get('llamacpp_chat_format')
                        # Add other specific args here if get_llm_provider is updated to handle them
                    }
                    # Filter out None values so get_llm_provider uses its internal defaults for non-provided args
                    provider_specific_constructor_args = {k: v for k, v in provider_specific_constructor_args.items() if v is not None}

                    if self._debug: 
                        logging.debug(f"process_files: Attempting to initialize LLM provider '{provider_name_str}' "
                                      f"with model_name='{model_name_for_provider}', api_key_present={bool(api_key_for_provider)}, "
                                      f"constructor_specific_args={provider_specific_constructor_args}")
                    
                    llm_provider_instance = get_llm_provider(
                        provider_type=provider_name_str,
                        model_name=model_name_for_provider, # This is args.llm_model
                        api_key=api_key_for_provider,       # This is args.api_key
                        **provider_specific_constructor_args # Pass the collected specific args
                    )
                    if llm_provider_instance and self._debug:
                         logging.debug(f"process_files: LLM provider '{llm_provider_instance.__class__.__name__}' "
                                       f"for model '{llm_provider_instance.model_name}' initialized successfully.")

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
                                   'local_openai' in llm_provider_instance.__class__.__name__.lower()) # Check LocalOpenAIProvider too
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
            
            if self._debug: logging.debug(f"process_files: Determined rename script base path string: {actual_rename_script_base_path_str}")
            
            initialized_script_paths_map = initialize_rename_scripts(actual_rename_script_base_path_str)
            if self._debug: logging.debug(f"process_files: Rename scripts initialized. Paths map: {initialized_script_paths_map}")
        elif effective_sort_flag and not rename_script_path:
            logging.warning("Sorting is enabled but no rename script path was effectively determined. Rename command generation will be skipped.")
            # Keep effective_sort_flag for metadata, but initialized_script_paths_map will be None.

        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            futures = {}
            for input_file_item in input_files:
                if shutdown_flag.is_set():
                    logging.info(f"Shutdown initiated, no more files will be submitted. {len(futures)} already submitted.")
                    break
                
                if self._debug: logging.debug(f"process_files: Submitting job for file: {input_file_item}")
                
                # Pass explicit temperature and max_tokens for metadata, and **kwargs for provider init within helpers if necessary
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
                    temperature, # Temperature specifically for metadata extraction via llm_extract_metadata
                    max_tokens,  # Max_tokens specifically for metadata extraction via llm_extract_metadata
                    **kwargs     # Pass all other CLI args (like llm_model, api_key, provider_specifics)
                                 # This is for llm_extract_metadata/sort_author_names if they internally
                                 # call get_llm_provider (though they shouldn't if instance is passed)
                )] = input_file_item
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Overall File Processing", unit="file"):
                if shutdown_flag.is_set() and not future.done(): 
                    if self._debug: logging.debug(f"process_files: Cancelling future for {futures[future]} due to shutdown.")
                    future.cancel()
                
                input_file_path_completed = futures[future]
                try:
                    single_file_result = future.result() 
                    if self._debug: 
                        logging.debug(f"process_files: Result for '{input_file_path_completed}': "
                                      f"Success={single_file_result.get('success')}, "
                                      f"Skipped={single_file_result.get('skipped')}, "
                                      f"RenamedInfo_is_present={bool(single_file_result.get('renamed_info'))}, "
                                      f"Error='{single_file_result.get('error', 'None')}'")
                    
                    final_results['processed'][input_file_path_completed] = single_file_result
                    if single_file_result.get('skipped'):
                        counters['skipped'] += 1
                        final_results['skipped'].append(input_file_path_completed)
                    elif single_file_result.get('success'):
                        counters['processed_ok'] += 1
                        if single_file_result.get('renamed_info'): 
                            counters['sorted_commands_generated'] +=1
                            if self._debug: logging.debug(f"process_files: Incremented sorted_commands_generated for {input_file_path_completed}. Count: {counters['sorted_commands_generated']}")
                        elif effective_sort_flag: # If sort was intended, but no rename info generated (e.g. bad metadata)
                            counters['sort_failed_metadata'] +=1
                            if self._debug: logging.debug(f"process_files: Incremented sort_failed_metadata for {input_file_path_completed} (sort active but no rename_info). Count: {counters['sort_failed_metadata']}")
                    else: # Not skipped, not success (extraction or direct read failed)
                        counters['failed_extraction'] += 1
                        final_results['failed'].append({'file': input_file_path_completed, 'error': single_file_result.get('error', 'Unknown processing error')})
                        if self._debug: logging.debug(f"process_files: Incremented failed_extraction for {input_file_path_completed}. Error: {single_file_result.get('error')}")
                
                except Exception as e_future:
                    logging.error(f"Exception processing future for {input_file_path_completed}: {e_future}", exc_info=self._debug)
                    final_results['failed'].append({'file': input_file_path_completed, 'error': str(e_future)})
                    counters['failed_extraction'] += 1 
                    if self._debug: logging.debug(f"process_files: Exception for {input_file_path_completed}, incremented failed_extraction.")
        
        final_results['counters'] = counters 
        logging.info("\n--- Processing Summary ---")
        logging.info(f"Total files attempted: {counters['total']}")
        logging.info(f"Successfully processed (text extracted/loaded): {counters['processed_ok']}")
        if kwargs.get('sort_arg_from_main', False): # Use original sort intent from BiblioForge.py for summary
            logging.info(f"Rename commands generated for: {counters['sorted_commands_generated']}")
            logging.info(f"Sorting failed (metadata issues/LLM errors for successfully processed files): {counters['sort_failed_metadata']}")
        logging.info(f"Skipped (e.g., output existed and not --noskip): {counters['skipped']}")
        logging.info(f"Failed extraction/critical processing errors: {counters['failed_extraction']}")
        if final_results['failed']:
            logging.warning("Details for files that failed critical processing:")
            for fail_info in final_results['failed']:
                logging.warning(f"  - {fail_info['file']}: {fail_info['error']}")
        logging.info("------------------------")

        return final_results