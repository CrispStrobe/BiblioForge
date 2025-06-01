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
    """Command-line interface entry point"""
    # Argument Parser Setup
    parser = argparse.ArgumentParser(
        description="BiblioForge: Document Text Extraction & Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          %(prog)s input.pdf
          %(prog)s -o output_texts/ *.pdf *.epub
          %(prog)s --method=pymupdf --ocr-method=tesseract document.pdf
          %(prog)s --sort --llm-provider=ollama --rename-script=my_renames.sh ./*.djvu
          %(prog)s --force-ocr --output-dir ocr_output/ scanned_collection/
          %(prog)s --file-types pdf,epub --recursive ./my_library/
        """)
    )

    parser.add_argument(
        'files', nargs='*',
        help="Input files or patterns to process (e.g., *.pdf, 'docs/*.epub'). "
             "If no files are given, processes supported files in current directory."
    )
    parser.add_argument(
        '-o', '--output-dir', default='.',
        help="Directory for extracted text files and sorted/renamed files (default: current directory)."
    )
    parser.add_argument(
        '-m', '--method', default=None,
        help="Preferred primary extraction method (e.g., 'pymupdf' for PDF, 'ebooklib' for EPUB)."
    )
    parser.add_argument(
        '--ocr-method', choices=['auto', 'tesseract', 'paddleocr', 'doctr', 'easyocr', 'kraken', 'kraken_cli'],
        default='auto', help="Preferred OCR method if OCR is needed (default: auto)."
    )
    parser.add_argument(
        '--force-ocr', action='store_true',
        help="Force OCR processing for all pages, even if a text layer exists."
    )
    parser.add_argument(
        '-r', '--recursive', action='store_true',
        help="Process files recursively in subdirectories of specified paths."
    )
    parser.add_argument(
        '-p', '--password', default=None,
        help="Password for encrypted documents."
    )
    parser.add_argument(
        '-t', '--tables', action='store_true',
        help="Attempt to extract tables (primarily for PDF files)."
    )
    parser.add_argument(
        '-j', '--json', default=None,
        help="Save a JSON file with detailed results of the processing."
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=None,
        help="Maximum number of worker threads (default: auto-detected, less for --sort)."
    )
    parser.add_argument(
        '--noskip', action='store_true',
        help="Process files and overwrite/create unique output text files even if they already exist."
    )
    parser.add_argument(
        '--file-types', default=None,
        help="Comma-separated list of file extensions to process (e.g., 'pdf,epub')."
    )
    parser.add_argument(
        '--sort', action='store_true',
        help="Enable LLM-based metadata extraction, sorting, and generation of rename commands."
    )
    parser.add_argument(
        '--rename-script', default="rename_commands.sh",
        help="Filename for the generated rename script when --sort is active (default: rename_commands.sh)."
    )
    parser.add_argument(
        '--execute-rename', action='store_true',
        help="Automatically execute the generated rename script after processing (use with caution)."
    )
    parser.add_argument(
        '--llm-provider', choices=['ollama', 'groq', 'cohere', 'openai', 'glhf', 'huggingface', 'poe'],
        default='ollama', help="LLM provider for --sort (default: ollama)."
    )
    parser.add_argument(
        '--llm-model', default=None,
        help="Specific model name for the chosen LLM provider."
    )
    parser.add_argument(
        '--api-key', default=None,
        help="API key for cloud-based LLM providers (if not set as environment variable)."
    )
    parser.add_argument(
        '--temperature', type=float, default=0.3,
        help="LLM temperature for metadata/author name tasks (0.0-2.0)."
    )
    parser.add_argument(
        '--max-tokens', type=int, default=250,
        help="LLM max tokens for metadata/author name tasks (default: 250)."
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="Increase verbosity level (-v for INFO, -vv for DEBUG)."
    )
    # --debug is a common alias for max verbosity
    parser.add_argument(
        '-d', '--debug', action='store_const', const=2, dest='verbose',
        help="Enable debug logging (shortcut for -vv)."
    )
    args = parser.parse_args()

    # Setup logging (now imported from utils)
    # Verbosity: 0 = WARNING, 1 = INFO, 2 = DEBUG
    setup_logging(args.verbose)
    logging.debug(f"Arguments received: {args}")

    # Register signal handlers (uses shutdown_flag and active_processes from utils)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if platform.system() != 'Windows': # SIGTSTP not available on Windows
        try:
            signal.signal(signal.SIGTSTP, signal_handler)
        except AttributeError: # Some minimal Unix might not have SIGTSTP
            logging.warning("SIGTSTP signal not available on this platform.")


    # Prepare list of input files
    input_files: List[str] = []
    
    # Determine supported extensions for filtering
    # ExtractionManager.SUPPORTED_EXTENSIONS is a class variable now.
    # We need an instance of ExtractionManager to access it, or make it static/module-level.
    # For simplicity, let's define a local copy for main's file discovery logic.
    # Or better, ExtractionManager can provide this as a static method or class variable.
    # Let's assume ExtractionManager.SUPPORTED_EXTENSIONS is accessible.
    # Since ExtractionManager is imported, we can use it.
    
    # If user specified file types, use them, otherwise use all supported
    active_supported_extensions = list(ExtractionManager.SUPPORTED_EXTENSIONS.keys())
    if args.file_types:
        user_types = [f".{ext.strip().lower()}" for ext in args.file_types.split(',')]
        active_supported_extensions = [ext for ext in user_types if ext in ExtractionManager.SUPPORTED_EXTENSIONS.keys()]
        if not active_supported_extensions:
            logging.error(f"No valid file types specified in --file-types: {args.file_types}")
            logging.info(f"Available types: {', '.join(k.lstrip('.') for k in ExtractionManager.SUPPORTED_EXTENSIONS.keys())}")
            return 1
    
    if not args.files: # No files/patterns given, scan current directory
        logging.info(f"No input files specified. Scanning current directory for: {', '.join(active_supported_extensions)}")
        for ext_pattern in active_supported_extensions:
            input_files.extend(glob.glob(f"*{ext_pattern}"))
        if args.recursive:
            for root, _, files_in_dir in os.walk("."):
                for f_name in files_in_dir:
                    if os.path.splitext(f_name)[1].lower() in active_supported_extensions:
                        input_files.append(os.path.join(root, f_name))
    else:
        for pattern_group in args.files:
            # Handle if shell already expanded a pattern into a single file argument
            if os.path.isfile(pattern_group) and os.path.splitext(pattern_group)[1].lower() in active_supported_extensions:
                 input_files.append(pattern_group)
                 continue
            
            for pattern in pattern_group.split(): # Allow space-separated patterns in one arg
                if args.recursive and ("*" in pattern or "?" in pattern): # Glob with recursion
                    # Handle potential base directory in pattern
                    base_dir = os.path.dirname(pattern) or "."
                    glob_pattern = os.path.basename(pattern)
                    for root, _, files_in_dir in os.walk(base_dir):
                        for f_name in files_in_dir:
                            if Path(f_name).match(glob_pattern) and os.path.splitext(f_name)[1].lower() in active_supported_extensions:
                                input_files.append(os.path.join(root, f_name))
                elif os.path.isdir(pattern): # If it's a directory
                     if args.recursive:
                        for root, _, files_in_dir in os.walk(pattern):
                            for f_name in files_in_dir:
                                if os.path.splitext(f_name)[1].lower() in active_supported_extensions:
                                    input_files.append(os.path.join(root, f_name))
                     else: # Non-recursive directory scan
                        for f_name in os.listdir(pattern):
                            full_path = os.path.join(pattern, f_name)
                            if os.path.isfile(full_path) and os.path.splitext(f_name)[1].lower() in active_supported_extensions:
                                input_files.append(full_path)
                else: # Standard glob for files/patterns
                    input_files.extend(glob.glob(pattern))

    # Filter final list by extension again and remove duplicates
    temp_files = []
    seen_files = set()
    for f_path in input_files:
        abs_f_path = os.path.abspath(f_path)
        if os.path.isfile(abs_f_path) and os.path.splitext(abs_f_path)[1].lower() in active_supported_extensions:
            if abs_f_path not in seen_files:
                temp_files.append(abs_f_path)
                seen_files.add(abs_f_path)
    input_files = temp_files

    if not input_files:
        logging.error("No input files found to process with the given criteria.")
        return 1
    
    logging.info(f"Found {len(input_files)} file(s) to process.")
    if args.verbose > 1: # Debug level
        for f_idx, f_path in enumerate(input_files[:10]): # Log first 10
            logging.debug(f"  File {f_idx+1}: {f_path}")
        if len(input_files) > 10: logging.debug(f"  ... and {len(input_files)-10} more files.")


    # Initialize DocumentProcessor
    processor = DocumentProcessor(debug=(args.verbose > 1))

    # LLM Provider instance for sorting (if enabled)
    llm_provider_instance: Optional[LLMProvider] = None
    actual_rename_script_path: Optional[str] = None

    if args.sort:
        try:
            logging.info(f"Initializing LLM provider '{args.llm_provider}' for sorting features...")
            llm_provider_instance = get_llm_provider(
                provider_type=args.llm_provider,
                model_name=args.llm_model,
                api_key=args.api_key
            )
            model_used = llm_provider_instance.model_name if llm_provider_instance else "N/A"
            logging.info(f"LLM provider '{args.llm_provider}' (model: {model_used}) initialized for sorting.")
            
            # Ensure output_dir exists for rename script
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            # Rename script path is relative to output_dir IF it's just a filename
            if os.path.basename(args.rename_script) == args.rename_script:
                actual_rename_script_path = str(Path(args.output_dir) / args.rename_script)
            else: # User provided a full or relative path
                actual_rename_script_path = args.rename_script
            
            initialize_rename_scripts(actual_rename_script_path) # From utils.py
        except Exception as e_llm_init:
            logging.error(f"Failed to initialize LLM provider or rename script for sorting: {e_llm_init}. Sorting will be disabled.")
            args.sort = False # Disable sorting if setup fails
            llm_provider_instance = None
            actual_rename_script_path = None
    
    # Process files
    try:
        processing_results_summary = processor.process_files(
            input_files=input_files,
            output_dir=args.output_dir, # Base directory for .txt and sorted files
            method=args.method,
            ocr_method=args.ocr_method,
            password=args.password,
            extract_tables=args.tables,
            force_ocr=args.force_ocr,
            max_workers=args.workers,
            noskip=args.noskip,
            sort=args.sort, # Use potentially modified args.sort
            rename_script_path=actual_rename_script_path, # Pass the path for the script
            llm_provider_arg=llm_provider_instance, # Pass the initialized LLMProvider instance
            temperature=args.temperature,
            max_tokens=args.max_tokens
            # Any other kwargs for DocumentProcessor.process_files can go here
        )

        if args.json:
            json_output_path = args.json
            logging.info(f"Saving detailed processing results to {json_output_path}")
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f_json:
                    # Make a serializable version of the results
                    serializable_results = {
                        "summary": {
                            "total_files_scanned": len(input_files),
                            "successfully_processed_text": len(processing_results_summary.get('processed', {})) - len(processing_results_summary.get('failed', [])),
                            "skipped": len(processing_results_summary.get('skipped', [])),
                            "failed": len(processing_results_summary.get('failed', [])),
                        },
                        "details": processing_results_summary.get('processed', {}),
                        "failures": processing_results_summary.get('failed', [])
                    }
                    json.dump(serializable_results, f_json, indent=2, ensure_ascii=False)
            except Exception as e_json:
                logging.error(f"Failed to save JSON results: {e_json}")


        if args.sort and args.execute_rename and actual_rename_script_path:
            logging.info(f"Attempting to execute rename commands from: {actual_rename_script_path}")
            execute_rename_commands(actual_rename_script_path) # From utils.py
        elif args.sort and actual_rename_script_path:
            logging.info(f"Rename script generated at: {actual_rename_script_path}")
            if platform.system() == "Windows":
                 batch_equivalent = os.path.splitext(actual_rename_script_path)[0] + ".bat"
                 if os.path.exists(batch_equivalent):
                     logging.info(f"  For Windows, you can also run: {batch_equivalent}")
                 else:
                     logging.info(f"  To run on Unix-like systems (or WSL): bash {actual_rename_script_path}")
            else:
                 logging.info(f"  To execute: bash {actual_rename_script_path}")
        
        return 0 if not processing_results_summary.get('failed') else 1

    except KeyboardInterrupt:
        # shutdown_flag should have been set by signal_handler
        logging.warning("\nOperation cancelled by user (KeyboardInterrupt in main).")
        if args.sort and actual_rename_script_path and os.path.exists(actual_rename_script_path):
             logging.info(f"Partial rename script may exist at {actual_rename_script_path}")
        return 130 # Standard exit code for Ctrl+C
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