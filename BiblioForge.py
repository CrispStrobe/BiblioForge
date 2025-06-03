#!/usr/bin/env python3
"""
BiblioForge: Cross-platform PDF, EPUB, DJVU, MOBI, and Text Document Extraction and Sorting Tool
"""
import warnings
# Suppress common warnings early
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import os
import sys
import logging
import argparse
import textwrap
import platform
import signal
import threading
import json # For handling --json output
from pathlib import Path
from typing import Optional, List, Dict, Any # For type hinting
import glob

# --- Custom Module Imports ---
# Assuming llm_providers.py, utils.py, extraction_manager.py, document_processor.py
# and the 'extractors' directory are in the same directory as BiblioForge.py
# or are otherwise in the PYTHONPATH.

try:
    from llm_providers import get_llm_provider, LLMProvider
    # Specific LLM functions like send_to_llm, extract_metadata, sort_author_names
    # are now primarily used internally by DocumentProcessor or ExtractionManager,
    # which import them directly from llm_providers.
except ImportError as e:
    print(f"Critical Error: Could not import 'llm_providers' module. {e}", file=sys.stderr)
    print("Please ensure llm_providers.py is in the correct location.", file=sys.stderr)
    sys.exit(1)

try:
    from utils import (
        setup_logging,
        initialize_rename_scripts,
        execute_rename_commands,
        shutdown_flag, # For signal_handler
        active_processes, # For signal_handler
        extraction_in_progress # For signal_handler
    )
except ImportError as e:
    print(f"Critical Error: Could not import 'utils' module. {e}", file=sys.stderr)
    print("Please ensure utils.py is in the correct location.", file=sys.stderr)
    sys.exit(1)

try:
    from extraction_manager import ExtractionManager # Used by DocumentProcessor
except ImportError as e:
    print(f"Critical Error: Could not import 'extraction_manager' module. {e}", file=sys.stderr)
    print("Please ensure extraction_manager.py is in the correct location.", file=sys.stderr)
    sys.exit(1)

try:
    from document_processor import DocumentProcessor
except ImportError as e:
    print(f"Critical Error: Could not import 'document_processor' module. {e}", file=sys.stderr)
    print("Please ensure document_processor.py is in the correct location.", file=sys.stderr)
    sys.exit(1)

# --- Global Constants & Variables ---
# MODEL_NAME for LLM is now primarily handled within llm_providers.py as DEFAULT_OLLAMA_MODEL_NAME
# or passed via --llm-model argument.

# Signal handler (kept in main script due to its global nature and process control)
def signal_handler(signum, frame):
    """
    Enhanced signal handler for SIGINT, SIGTERM.
    Sets the shutdown flag and attempts to terminate active processes.
    """
    # Uses shutdown_flag, active_processes, extraction_in_progress from utils.py
    if not shutdown_flag.is_set():
        signal_name_map = {signal.SIGINT: "SIGINT (Ctrl+C)", signal.SIGTERM: "SIGTERM"}
        if platform.system() != 'Windows': signal_name_map[signal.SIGTSTP] = "SIGTSTP (Ctrl+Z)"
        signal_name = signal_name_map.get(signum, f"Signal {signum}")
        
        logging.info(f"\nReceived {signal_name}. Initiating graceful shutdown...")
        shutdown_flag.set()
        
        # Attempt to terminate active processes tracked by run_process in utils
        logging.info(f"Attempting to terminate {len(active_processes)} active subprocess(es)...")
        for proc in list(active_processes): # Iterate over a copy
            if proc and proc.poll() is None:
                try:
                    logging.info(f"Terminating process {proc.pid}...")
                    proc.terminate()
                    proc.wait(timeout=2) # Wait briefly for graceful termination
                    if proc.poll() is None: # If still running
                        logging.warning(f"Process {proc.pid} did not terminate gracefully, killing...")
                        proc.kill()
                except Exception as e:
                    logging.error(f"Error terminating process {proc.pid}: {e}")
        
        active_processes.clear() # Clear the list

        # If extraction threads are potentially blocking, this event helps them check
        if extraction_in_progress.is_set():
            logging.info("Signaling ongoing extractions to halt if possible.")
            # Threads should check shutdown_flag periodically.
            # This event is more for the run_process context.

        logging.info("Shutdown sequence initiated. Main thread will exit after current tasks complete or are cancelled.")
        # Depending on threading model, further cleanup or sys.exit might be needed here
        # For ThreadPoolExecutor, letting it complete or cancel existing futures is usually enough.
        # The main thread will then exit naturally.
    else:
        logging.info("Shutdown already in progress.")


def main():
    # --- Argument Parser Setup (ensure all new LLM args are defined here as in previous full file) ---
    parser = argparse.ArgumentParser(
        description="BiblioForge: Document Text Extraction & Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          %(prog)s input.pdf
          %(prog)s -o output_texts/ *.pdf *.epub
          %(prog)s --sort --llm-provider=ollama --output-dir ./sorted_library/ documents/
          %(prog)s --sort --llm-provider=llama_cpp --llamacpp-repo-id="TheBloke/phi-2-GGUF" --llamacpp-gguf-filename="phi-2.Q4_K_M.gguf" ./scans/
          %(prog)s --sort --llm-provider=local_openai --local-openai-base-url="http://localhost:1234/v1" --llm-model="local-model" books/
        """)
    )

    parser.add_argument('files', nargs='*', help="Input files or patterns to process.")
    parser.add_argument('-o', '--output-dir', default='.', help="Base directory for outputs.")
    parser.add_argument('-m', '--method', default=None, help="Preferred primary extraction method.")
    parser.add_argument('--ocr-method', choices=['auto', 'tesseract', 'paddleocr', 'doctr', 'easyocr', 'kraken', 'kraken_cli'], default='auto', help="Preferred OCR method.")
    parser.add_argument('--force-ocr', action='store_true', help="Force OCR processing.")
    parser.add_argument('-r', '--recursive', action='store_true', help="Process files recursively.")
    parser.add_argument('-p', '--password', default=None, help="Password for encrypted documents.")
    parser.add_argument('-t', '--tables', action='store_true', help="Attempt to extract tables.")
    parser.add_argument('-j', '--json', default=None, help="Save detailed results to a JSON file (path).")
    parser.add_argument('-w', '--workers', type=int, default=None, help="Maximum number of worker threads.")
    parser.add_argument('--noskip', action='store_true', help="Re-process files even if output text exists.")
    parser.add_argument('--file-types', default=None, help="Comma-separated list of file extensions (e.g., 'pdf,epub').")
    
    parser.add_argument('--sort', action='store_true', help="Enable LLM-based metadata extraction and sorting.")
    parser.add_argument('--rename-script', default="rename_commands.sh", help="Filename for the generated rename script.")
    parser.add_argument('--execute-rename', action='store_true', help="Automatically execute the rename script.")
    
    parser.add_argument(
        '--llm-provider', 
        choices=['ollama', 'groq', 'cohere', 'openai', 'glhf', 'huggingface', 'poe', 'local_openai', 'llama_cpp'], 
        default='ollama', help="LLM provider for --sort."
    )
    parser.add_argument('--llm-model', default=None, help="Specific model name for the chosen LLM provider OR repo_id for LlamaCPP if specific repo arg not used.")
    parser.add_argument('--api-key', default=None, help="API key for cloud-based LLM providers.")
    parser.add_argument('--temperature', type=float, default=0.3, help="LLM temperature (0.0-2.0).") # Explicit parameter for process_files
    parser.add_argument('--max-tokens', type=int, default=300, help="LLM max tokens for metadata extraction.") # Explicit parameter for process_files

    parser.add_argument('--ollama-host', default=os.environ.get("OLLAMA_HOST"), help="Host for Ollama server (e.g., http://localhost:11434). Uses library default if not set.")
    parser.add_argument('--local-openai-base-url', default=os.environ.get("LOCAL_OPENAI_BASE_URL"), help="Base URL for Local OpenAI compatible servers (e.g., LM Studio's http://localhost:1234/v1/).")
    
    parser.add_argument('--llamacpp-repo-id', default=None, help="HuggingFace Repo ID for LlamaCPP GGUF model (e.g., TheBloke/phi-2-GGUF). Overrides --llm-model for LlamaCPP repo.")
    parser.add_argument('--llamacpp-gguf-filename', default=None, help="Specific GGUF filename from the HF repo for LlamaCPP (e.g., phi-2.Q4_K_M.gguf).")
    parser.add_argument('--llamacpp-n-ctx', type=int, default=None, help="Context size (n_ctx) for LlamaCPP (e.g., 2048).")
    parser.add_argument('--llamacpp-n-gpu-layers', type=int, default=None, help="Number of layers to offload to GPU for LlamaCPP (-1 for all, 0 for CPU).")
    parser.add_argument('--llamacpp-chat-format', default=None, help="Chat format for LlamaCPP (e.g., llama-2, chatml).")

    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity: -v INFO, -vv DEBUG.")
    parser.add_argument('-d', '--debug', action='store_const', const=2, dest='verbose', help="Enable debug logging (shortcut for -vv).")
    
    args = parser.parse_args()
    setup_logging(args.verbose) 
    logging.debug(f"Arguments received: {args}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if platform.system() != 'Windows':
        try:
            signal.signal(signal.SIGTSTP, signal_handler)
        except AttributeError:
            logging.warning("SIGTSTP signal not available on this platform.")

    # --- File Discovery ---
    input_files_discovered: List[str] = []
    all_supported_extensions = list(ExtractionManager.SUPPORTED_EXTENSIONS.keys())
    active_supported_extensions = all_supported_extensions

    if args.file_types:
        user_types = [f".{ext.strip().lower()}" for ext in args.file_types.split(',') if ext.strip()]
        active_supported_extensions = [ext for ext in user_types if ext in all_supported_extensions]
        if not active_supported_extensions:
            logging.error(f"No valid file types specified in --file-types: '{args.file_types}'. Please check against supported types.")
            logging.info(f"Available types: {', '.join(k.lstrip('.') for k in all_supported_extensions)}")
            return 1
        logging.info(f"Processing user-specified file types: {', '.join(active_supported_extensions)}")
    else:
        logging.info(f"Processing all supported file types: {', '.join(k.lstrip('.') for k in active_supported_extensions)}") # Cleaned up logging

    if not args.files: 
        scan_path = Path(".").resolve()
        logging.info(f"No input files/patterns specified. Scanning '{str(scan_path)}' for specified types.")
        if args.recursive:
            for item in scan_path.rglob('*'):
                if item.is_file() and item.suffix.lower() in active_supported_extensions:
                    input_files_discovered.append(str(item))
        else:
            for item in scan_path.glob('*'):
                if item.is_file() and item.suffix.lower() in active_supported_extensions:
                    input_files_discovered.append(str(item))
    else:
        for pattern_item in args.files:
            potential_path = Path(pattern_item)
            if potential_path.is_dir():
                if args.recursive:
                    for item in potential_path.rglob('*'):
                        if item.is_file() and item.suffix.lower() in active_supported_extensions:
                            input_files_discovered.append(str(item))
                else: 
                    for item in potential_path.glob('*'):
                        if item.is_file() and item.suffix.lower() in active_supported_extensions:
                            input_files_discovered.append(str(item))
            else: 
                expanded_items = glob.glob(pattern_item, recursive=args.recursive)
                for item_str_path in expanded_items:
                    item_path_obj = Path(item_str_path)
                    if item_path_obj.is_file() and item_path_obj.suffix.lower() in active_supported_extensions:
                        input_files_discovered.append(str(item_path_obj))
                    elif args.recursive and item_path_obj.is_dir(): 
                         for sub_item in item_path_obj.rglob('*'):
                            if sub_item.is_file() and sub_item.suffix.lower() in active_supported_extensions:
                                input_files_discovered.append(str(sub_item))

    final_input_files = sorted(list(set(str(Path(p).resolve()) for p in input_files_discovered)))

    if not final_input_files:
        logging.error("No input files found to process with the given criteria.")
        return 1
    
    logging.info(f"Found {len(final_input_files)} unique file(s) to process.")
    if args.verbose > 1: 
        for f_idx, f_path in enumerate(final_input_files[:20]):
            logging.debug(f"  File {f_idx+1}: {f_path}")
        if len(final_input_files) > 20: logging.debug(f"  ... and {len(final_input_files)-20} more files.")

    processor = DocumentProcessor(debug=(args.verbose > 1))
    
    # Prepare kwargs for LLM provider initialization.
    # These are passed to process_files, which then passes them to get_llm_provider.
    llm_config_kwargs = {
        # 'llm_model' is for the specific model identifier within the provider
        'llm_model': args.llm_model, 
        'api_key': args.api_key,
        'ollama_host': args.ollama_host,
        'local_openai_base_url': args.local_openai_base_url,
        'llamacpp_repo_id': args.llamacpp_repo_id,
        'llamacpp_gguf_filename': args.llamacpp_gguf_filename,
        'llamacpp_n_ctx': args.llamacpp_n_ctx,
        'llamacpp_n_gpu_layers': args.llamacpp_n_gpu_layers,
        'llamacpp_chat_format': args.llamacpp_chat_format,
        # Pass the main debug flag for providers to use
        'debug': (args.verbose > 1) 
    }
    # Filter out None values, so defaults in get_llm_provider/provider classes are used
    llm_config_kwargs_to_pass = {k: v for k, v in llm_config_kwargs.items() if v is not None}
    
    # Add sort_arg_from_main for summary display logic in process_files
    # This specific kwarg is for process_files itself, not get_llm_provider.
    llm_config_kwargs_to_pass['sort_arg_from_main'] = args.sort 

    if args.verbose > 1:
        logging.debug(f"main: Filtered LLM config kwargs to pass to process_files: {llm_config_kwargs_to_pass}")

    try:
        processing_results_summary = processor.process_files(
            input_files=final_input_files,
            output_dir=args.output_dir,
            method=args.method,
            ocr_method=args.ocr_method,
            password=args.password,
            extract_tables=args.tables,
            force_ocr=args.force_ocr,
            max_workers=args.workers,
            noskip=args.noskip,
            sort=args.sort, 
            rename_script_path=args.rename_script,
            llm_provider_arg=args.llm_provider, 
            temperature=args.temperature,   # Explicitly passed
            max_tokens=args.max_tokens,     # Explicitly passed
            **llm_config_kwargs_to_pass     # Pass all other relevant configs
        )

        if args.json:
            json_output_path = args.json
            logging.info(f"Saving detailed processing results to {json_output_path}")
            try:
                summary_counters = processing_results_summary.get('counters', {})
                serializable_summary = {
                    "total_files_attempted": summary_counters.get('total', len(final_input_files)),
                    "successfully_processed_text": summary_counters.get('processed_ok', 0),
                    "rename_commands_generated": summary_counters.get('sorted_commands_generated', 0),
                    "sorting_metadata_failures": summary_counters.get('sort_failed_metadata', 0),
                    "skipped": summary_counters.get('skipped', 0),
                    "failed_extraction_critical": summary_counters.get('failed_extraction', 0),
                }
                full_json_output = {
                    "run_summary_counts": serializable_summary,
                    "detailed_results_per_file": processing_results_summary.get('processed', {}),
                    "critical_failures_list": processing_results_summary.get('failed', [])
                }
                Path(json_output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(json_output_path, 'w', encoding='utf-8') as f_json:
                    json.dump(full_json_output, f_json, indent=2, ensure_ascii=False)
            except Exception as e_json:
                logging.error(f"Failed to save JSON results to '{json_output_path}': {e_json}")

        final_rename_script_path_for_user: Optional[str] = None
        if args.sort and args.rename_script: # Only determine if sorting was on and script name provided
            if os.path.isabs(args.rename_script) or os.path.dirname(args.rename_script):
                # If args.rename_script is already a path (absolute or relative with directory components)
                final_rename_script_path_for_user = str(Path(args.rename_script).resolve())
            else:
                # If args.rename_script is just a filename, combine with output_dir and then resolve
                path_obj = Path(args.output_dir).resolve() / args.rename_script
                final_rename_script_path_for_user = str(path_obj.resolve())
        
        if args.sort and args.execute_rename and final_rename_script_path_for_user:
            if os.path.exists(final_rename_script_path_for_user):
                logging.info(f"Attempting to execute rename commands from: {final_rename_script_path_for_user}")
                execute_rename_commands(final_rename_script_path_for_user)
            else:
                logging.warning(f"Rename script '{final_rename_script_path_for_user}' not found. Cannot execute.")
        elif args.sort and final_rename_script_path_for_user:
            if os.path.exists(final_rename_script_path_for_user):
                logging.info(f"Rename script generated at: {final_rename_script_path_for_user}")
                if platform.system() == "Windows":
                    batch_equivalent = os.path.splitext(final_rename_script_path_for_user)[0] + ".bat"
                    if os.path.exists(batch_equivalent):
                        logging.info(f"  For Windows, you can also run: {batch_equivalent}")
                    else:
                         logging.info(f"  To run on Unix-like systems (or WSL/Git Bash): bash {final_rename_script_path_for_user}")
                else:
                    logging.info(f"  To execute: bash {final_rename_script_path_for_user}")
            else:
                 logging.info(f"Rename script was configured for '{final_rename_script_path_for_user}' but may not have been created (e.g., no files successfully sorted).")
        
        return 0 if not processing_results_summary.get('failed') and processing_results_summary.get('counters', {}).get('failed_extraction', 0) == 0 else 1

    except KeyboardInterrupt:
        logging.warning("\nOperation cancelled by user (KeyboardInterrupt in main).")
        if args.sort and args.rename_script:
            script_path_to_check = args.rename_script
            if not (os.path.isabs(script_path_to_check) or os.path.dirname(script_path_to_check)):
                script_path_to_check = str(Path(args.output_dir).resolve() / script_path_to_check)
            else:
                script_path_to_check = str(Path(script_path_to_check).resolve())
            if os.path.exists(script_path_to_check):
                 logging.info(f"Partial rename script may exist at {script_path_to_check}")
        return 130 
    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution: {e}", exc_info=args.verbose > 1)
        return 1
    finally:
        logging.info("BiblioForge processing finished.")


if __name__ == '__main__':
    # Ensure that the current working directory is in sys.path for module resolution
    # This is often needed if scripts in subdirectories try to import from the root
    # or sibling directories without the project being installed as a package.
    # However, with relative imports like `from .utils import ...` inside packages,
    # this should be less of an issue if the script is run as `python BiblioForge.py`
    # from its own directory, or `python -m BiblioForge.BiblioForge` if structured as a package.
    # For direct script execution, '.' is usually in sys.path by default.

    sys.exit(main())