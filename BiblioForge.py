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
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

# === LOAD .ENV FILE FOR API KEYS ===
try:
    from dotenv import load_dotenv, dotenv_values
    import os
    from pathlib import Path
    
    # Get current working directory (where user ran the command)
    cwd_env = Path.cwd() / '.env'
    
    print(f"Loading .env from: {cwd_env}")
    
    if cwd_env.exists():
        # First, read the file directly to see what's in it
        env_vars = dotenv_values(cwd_env)
        print(f"Found {len(env_vars)} variables in .env file:")
        for key in env_vars.keys():
            print(f"  - {key}")
        
        # Now actually load it with override=True
        load_dotenv(cwd_env, override=True, verbose=True)
        print("✓ Loaded .env file for API keys")
        
        # Verify each key
        print(f"\n=== DEBUG: Environment Variables ===")
        for key in ['POE_API_KEY', 'OPENROUTER_API_KEY', 'MISTRAL_API_KEY', 'SCALEWAY_API_KEY']:
            val = os.environ.get(key)
            if val:
                print(f"✓ {key}: {val[:10]}...{val[-4:] if len(val) > 14 else ''} (len={len(val)})")
            else:
                print(f"✗ {key}: NOT SET")
        print(f"===================================\n")
    else:
        print(f"⚠ No .env file found at {cwd_env}")
    
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")

import sys
import logging
import argparse
import textwrap
import platform
import signal
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import glob

# --- Custom Module Imports ---
try:
    from llm_providers import get_llm_provider, LLMProvider
except ImportError as e:
    print(f"Critical Error: Could not import 'llm_providers' module. {e}", file=sys.stderr)
    print("Please ensure llm_providers.py is in the correct location.", file=sys.stderr)
    sys.exit(1)

try:
    from utils import (
        setup_logging, _restore_logging_if_corrupted,
        initialize_rename_scripts,
        execute_rename_commands,
        shutdown_flag,
        active_processes,
        extraction_in_progress,
    )
except ImportError as e:
    print(f"Critical Error: Could not import 'utils' module. {e}", file=sys.stderr)
    print("Please ensure utils.py is in the correct location.", file=sys.stderr)
    sys.exit(1)

try:
    from extraction_manager import ExtractionManager
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

# Signal handler
def signal_handler(signum, frame):
    """Enhanced signal handler for SIGINT, SIGTERM."""
    if not shutdown_flag.is_set():
        signal_name_map = {signal.SIGINT: "SIGINT (Ctrl+C)", signal.SIGTERM: "SIGTERM"}
        if platform.system() != 'Windows': 
            signal_name_map[signal.SIGTSTP] = "SIGTSTP (Ctrl+Z)"
        signal_name = signal_name_map.get(signum, f"Signal {signum}")
        
        logging.info(f"\nReceived {signal_name}. Initiating graceful shutdown...")
        log_hanging_threads()
        shutdown_flag.set()
        
        logging.info(f"Attempting to terminate {len(active_processes)} active subprocess(es)...")
        for proc in list(active_processes):
            if proc and proc.poll() is None:
                try:
                    logging.info(f"Terminating process {proc.pid}...")
                    proc.terminate()
                    proc.wait(timeout=2)
                    if proc.poll() is None:
                        logging.warning(f"Process {proc.pid} did not terminate gracefully, killing...")
                        proc.kill()
                except Exception as e:
                    logging.error(f"Error terminating process {proc.pid}: {e}")
        
        active_processes.clear()
        
        if extraction_in_progress.is_set():
            logging.info("Signaling ongoing extractions to halt if possible.")
        
        logging.info("Shutdown sequence initiated.")
    else:
        logging.info("Shutdown already in progress.")

def filter_out_extracted_txt_files(file_list: List[str]) -> List[str]:
    """Filter out .txt files that are likely extracted versions of other files."""
    txt_files = []
    non_txt_files = []
    
    for file_path in file_list:
        if file_path.lower().endswith('.txt'):
            txt_files.append(file_path)
        else:
            non_txt_files.append(file_path)
    
    extracted_txt_files = set()
    
    for txt_file in txt_files:
        txt_base = os.path.splitext(txt_file)[0]
        for non_txt_file in non_txt_files:
            non_txt_base = os.path.splitext(non_txt_file)[0]
            if txt_base == non_txt_base:
                extracted_txt_files.add(txt_file)
                logging.debug(f"Excluding extracted .txt file: {txt_file}")
                break
    
    standalone_txt_files = [txt for txt in txt_files if txt not in extracted_txt_files]
    filtered_files = non_txt_files + standalone_txt_files
    
    if extracted_txt_files:
        logging.info(f"Filtered out {len(extracted_txt_files)} extracted .txt files")
        
    return sorted(filtered_files)

def log_hanging_threads():
    """Log all thread states for debugging hangs"""
    import threading
    print("\n=== THREAD DUMP ===", file=sys.stderr)
    for thread in threading.enumerate():
        print(f"\nThread: {thread.name} (ID: {thread.ident})", file=sys.stderr)
        print(f"  Alive: {thread.is_alive()}", file=sys.stderr)
        print(f"  Daemon: {thread.daemon}", file=sys.stderr)
    print("=== END THREAD DUMP ===\n", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="BiblioForge: Document Text Extraction & Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Basic extraction
          %(prog)s input.pdf
          %(prog)s -o output_texts/ *.pdf *.epub
          
          # OCR with Nanonets (Ollama - easiest)
          %(prog)s scans/ --use-nanonets-ocr2 --force-ocr
          
          # OCR with DocStrange (cloud - fastest)
          %(prog)s scans/ --use-docstrange --force-ocr
          
          # 100%% private processing
          %(prog)s docs/ --use-nanonets-ocr2 --force-ocr --sort --llm-provider=ollama
          
          # Metadata extraction with VQA
          %(prog)s docs/ --use-nanonets-ocr2 --nanonets-for-metadata --sort
          
          # Traditional sorting
          %(prog)s --sort --llm-provider=ollama --output-dir ./sorted_library/ documents/
        """)
    )

    # Basic arguments
    parser.add_argument('files', nargs='*', help="Input files or patterns to process.")
    parser.add_argument('-o', '--output-dir', default='.', help="Base directory for outputs.")
    parser.add_argument('-m', '--method', default=None, help="Preferred primary extraction method.")
    parser.add_argument('--ocr-method', choices=['auto', 'tesseract', 'paddleocr', 'doctr', 'easyocr', 'kraken', 'kraken_cli', 'crispembed'], default='auto', help="Preferred OCR method.")
    parser.add_argument('--crispembed-ocr-model', default='got-ocr2',
                        help="CrispEmbed OCR model (registry name) when --ocr-method crispembed (default got-ocr2).")
    parser.add_argument('--crispembed-ocr-dpi', type=int, default=150,
                        help="Render DPI for CrispEmbed OCR (default 150).")
    parser.add_argument('--crispembed-ocr-cpu', action='store_true',
                        help="Force CPU for CrispEmbed OCR (avoids ggml Metal unsupported-op aborts).")
    parser.add_argument('--force-ocr', action='store_true', help="Force OCR processing.")
    parser.add_argument('--scan-cleanup', choices=['off', 'auto', 'on'], default='auto',
                        help="Pre-OCR scan cleanup (deskew/crop/whiten) via CrispEmbed, if available. "
                             "'auto'/'on' clean page images before OCR; 'off' disables. No-op without CrispEmbed.")
    parser.add_argument('--scan-cleanup-binarize', choices=['off', 'otsu', 'sauvola'], default='off',
                        help="Adaptive binarization during scan cleanup (default off).")
    parser.add_argument('--sr', choices=['off', 'auto', 'on'], default='off',
                        help="Pre-OCR super-resolution for low-resolution pages via CrispEmbed. "
                             "'auto' upscales only pages narrower than --sr-min-width; 'on' upscales "
                             "any page below the safety cap. No-op without CrispEmbed. Default off.")
    parser.add_argument('--sr-engine', choices=['pan', 'swinir', 'hat', 'esrgan', 'safmn'], default='swinir',
                        help="Super-resolution engine (default swinir).")
    parser.add_argument('--sr-min-width', type=int, default=1500,
                        help="In --sr auto mode, upscale pages narrower than this many pixels (default 1500).")
    parser.add_argument('-r', '--recursive', action='store_true', help="Process files recursively.")
    parser.add_argument('-p', '--password', default=None, help="Password for encrypted documents.")
    parser.add_argument('-t', '--tables', action='store_true', help="Attempt to extract tables.")
    parser.add_argument('-j', '--json', default=None, help="Save detailed results to a JSON file (path).")
    parser.add_argument('-w', '--workers', type=int, default=None, help="Maximum number of worker threads.")
    parser.add_argument('--noskip', action='store_true', help="Re-process files even if output text exists.")
    parser.add_argument('--file-types', default=None, help="Comma-separated list of file extensions (e.g., 'pdf,epub').")
    
    # Sorting arguments
    parser.add_argument('--sort', action='store_true', help="Enable LLM-based metadata extraction and sorting.")
    parser.add_argument('--metadata-backend', choices=['llm', 'crispembed-ner', 'hybrid'], default='llm',
                        help="Metadata source when sorting: 'llm' (default), 'crispembed-ner' (local "
                             "GLiNER + language ID, no LLM/network), or 'hybrid' (NER first, LLM fallback).")
    parser.add_argument('--rename-script', default="rename_commands.sh", help="Filename for the generated rename script.")
    parser.add_argument('--execute-rename', action='store_true', help="Automatically execute the rename script.")
    parser.add_argument('--reset', action='store_true', help="Force creation of fresh rename script.")
    parser.add_argument('--noremove', action='store_true', help="Skip cleanup of rename script entries for non-existent files.")
    
    # LLM provider arguments
    parser.add_argument('--llm-provider', 
        choices=['ollama', 'groq', 'cohere', 'openai', 'glhf', 'huggingface', 'poe', 
                'local_openai', 'llama_cpp', 'mistral', 'nebius', 'scaleway', 'openrouter'], 
        default='ollama', 
        help="LLM provider for --sort.")
    parser.add_argument('--llm-model', default=None, help="Specific model name for the chosen LLM provider.")
    parser.add_argument('--api-key', default=None, help="API key for cloud-based LLM providers.")
    parser.add_argument('--temperature', type=float, default=0.3, help="LLM temperature (0.0-2.0).")
    parser.add_argument('--max-tokens', type=int, default=300, help="LLM max tokens for metadata extraction.")
    
    # Provider-specific arguments
    parser.add_argument('--ollama-host', default=os.environ.get("OLLAMA_HOST"), help="Host for Ollama server.")
    parser.add_argument('--local-openai-base-url', default=os.environ.get("LOCAL_OPENAI_BASE_URL"), help="Base URL for Local OpenAI compatible servers.")
    parser.add_argument('--ollama-allow-fallback', action='store_true', help="Allow falling back to another local Ollama model.")
    parser.add_argument('--ollama-fallback-order', default=None, help="Comma-separated list of preferred fallback models for Ollama.")
    parser.add_argument('--llamacpp-repo-id', default=None, help="HuggingFace Repo ID for LlamaCPP GGUF model.")
    parser.add_argument('--llamacpp-gguf-filename', default=None, help="Specific GGUF filename from the HF repo for LlamaCPP.")
    parser.add_argument('--llamacpp-n-ctx', type=int, default=None, help="Context size (n_ctx) for LlamaCPP.")
    parser.add_argument('--llamacpp-n-gpu-layers', type=int, default=None, help="Number of layers to offload to GPU for LlamaCPP.")
    parser.add_argument('--llamacpp-chat-format', default=None, help="Chat format for LlamaCPP.")
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity: -v INFO, -vv DEBUG.")
    parser.add_argument('-d', '--debug', action='store_const', const=2, dest='verbose', help="Enable debug logging.")
    
    # Nanonets-OCR2 arguments
    nanonets_group = parser.add_argument_group('Nanonets-OCR2 Options')
    nanonets_group.add_argument('--use-nanonets-ocr2', action='store_true', help='Use Nanonets-OCR2 for document extraction')
    nanonets_group.add_argument('--nanonets-model', choices=['3b', '1.5b', 'custom'], default='3b', help='Model size (3b: 6GB VRAM, custom: Ollama/GGUF)')
    nanonets_group.add_argument('--nanonets-model-path', default=None, help='Custom model path or Ollama model name')
    nanonets_group.add_argument('--nanonets-gguf-mmproj-path', default=None, 
                            help='Path to the GGUF mmproj file for llama.cpp')
    nanonets_group.add_argument('--nanonets-gguf-chat-format', default='llava-1.5', help='Chat format for GGUF model (e.g., llava-1.5)')
    nanonets_group.add_argument('--nanonets-use-gguf', action='store_true', help='Use GGUF model with llama.cpp')
    nanonets_group.add_argument('--nanonets-gguf-quant', choices=['q4', 'q8', 'f16'], default='q4', help='GGUF quantization (q4: 2GB, q8: 4GB, f16: 6GB)')
    nanonets_group.add_argument('--nanonets-use-ollama', action='store_true', help='Use Ollama for Nanonets')
    nanonets_group.add_argument('--nanonets-ollama-model', default='benhaotang/Nanonets-OCR-s:latest', help='Ollama model name')
    nanonets_group.add_argument('--nanonets-vllm-server', action='store_true', help='Use vLLM server')
    nanonets_group.add_argument('--nanonets-vllm-url', default='http://localhost:8000/v1', help='vLLM server URL')
    nanonets_group.add_argument('--nanonets-max-tokens', type=int, default=15000, help='Maximum output tokens')
    nanonets_group.add_argument('--nanonets-for-metadata', action='store_true', help='Use Nanonets VQA for metadata extraction')
    nanonets_group.add_argument('--nanonets-device', default='auto', help='Device (auto/cuda/cpu)')
    nanonets_group.add_argument('--nanonets-n-ctx', type=int, default=8192, help='Context size for GGUF mode')
    nanonets_group.add_argument('--nanonets-n-gpu-layers', type=int, default=-1, help='GPU layers for GGUF (-1: all, 0: CPU)')
    
    # DocStrange arguments
    docstrange_group = parser.add_argument_group('DocStrange Options')
    docstrange_group.add_argument('--use-docstrange', action='store_true', help='Use DocStrange for document extraction')
    docstrange_group.add_argument('--docstrange-mode', choices=['cloud', 'local_gpu', 'local_cpu', 'auto'], default='auto', help='Processing mode')
    docstrange_group.add_argument('--docstrange-api-key', default=None, help='API key for DocStrange cloud (10k docs/month)')
    docstrange_group.add_argument('--docstrange-login', action='store_true', help='Run authentication flow for 10k docs/month free')
    docstrange_group.add_argument('--docstrange-output-format', choices=['markdown', 'json', 'html', 'csv', 'text', 'markdown-financial-docs'], default='markdown', help='Output format')
    docstrange_group.add_argument('--docstrange-for-metadata', action='store_true', help='Use DocStrange for metadata extraction')
    docstrange_group.add_argument('--docstrange-metadata-fields', default=None, help='Comma-separated fields to extract')
    docstrange_group.add_argument('--docstrange-metadata-schema', default=None, help='JSON schema file for metadata extraction')
    docstrange_group.add_argument('--docstrange-local-only', action='store_true', help='Force local processing only (privacy mode)')

    # llama-mtmd-cli VL arguments
    parser.add_argument('--use-llama-mtmd-vl', action='store_true',
                        help="Use llama-mtmd-cli Vision Language model for OCR and extraction")
    parser.add_argument('--llama-mtmd-model', default='LiquidAI/LFM2-VL-3B-GGUF:Q4_0', 
                        help="HuggingFace model ID for llama-mtmd-cli. "
                            "For GGUF: 'repo:quant' (e.g., LiquidAI/LFM2-VL-3B-GGUF:Q4_0). "
                            "For regular: 'repo' (e.g., LiquidAI/LFM2-VL-3B). "
                            "(default: LiquidAI/LFM2-VL-3B-GGUF:Q4_0)")
    parser.add_argument('--llama-mtmd-for-metadata', action='store_true',
                        help="Use VL model for direct metadata extraction from document images")
    parser.add_argument('--llama-mtmd-max-tokens', type=int, default=4096,
                        help="Max tokens for llama-mtmd-cli (default: 4096)")
    parser.add_argument('--llama-mtmd-temp', type=float, default=0.1,
                        help="Temperature for llama-mtmd-cli (default: 0.1 for OCR accuracy)")
    parser.add_argument('--llama-mtmd-min-p', type=float, default=0.15, 
                        help="Min-p sampling for llama-mtmd-cli (default: 0.15)")
    parser.add_argument('--llama-mtmd-repeat-penalty', type=float, default=1.05, 
                        help="Repetition penalty for llama-mtmd-cli (default: 1.05)")
    
    # MLX-VLM arguments (Apple Silicon optimized)
    parser.add_argument('--use-mlx-vlm', action='store_true',
                        help="Use mlx-vlm Vision Language model (optimized for Apple Silicon)")
    parser.add_argument('--mlx-vlm-model', default='mlx-community/LFM2-VL-3B-8bit',
                        help="MLX-VLM model ID (default: mlx-community/LFM2-VL-3B-8bit)")
    parser.add_argument('--mlx-vlm-for-metadata', action='store_true',
                        help="Use MLX-VLM for direct metadata extraction")
    parser.add_argument('--mlx-vlm-max-tokens', type=int, default=512,
                        help="Max tokens for MLX-VLM (default: 512)")
    parser.add_argument('--mlx-vlm-temp', type=float, default=0.0,
                        help="Temperature for MLX-VLM (default: 0.0 for OCR)")
    
    # Parse arguments ONCE
    args = parser.parse_args()
    
    # Store verbosity for restoration
    logging._biblioforge_verbosity = args.verbose
    logging._biblioforge_target_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(args.verbose, logging.DEBUG)
    
    # Setup logging BEFORE any operations
    setup_logging(args.verbose)
    logging.debug(f"Arguments received: {args}")
    
    # Handle DocStrange login if requested (AFTER logging is set up)
    if args.docstrange_login:
        try:
            import subprocess
            logging.info("Starting DocStrange authentication...")
            result = subprocess.run(['docstrange', 'login'], check=True)
            logging.info("✓ DocStrange authentication complete (10k docs/month free)")
            logging.info("  Use --use-docstrange to start processing")
            return 0
        except FileNotFoundError:
            logging.error("DocStrange CLI not found. Install with: pip install docstrange")
            return 1
        except Exception as e:
            logging.error(f"DocStrange login failed: {e}")
            return 1
    
    # Configure DocStrange
    docstrange_config = None
    if args.use_docstrange:
        docstrange_mode = args.docstrange_mode
        
        # Auto mode detection
        if docstrange_mode == 'auto':
            try:
                import torch
                # CORRECTED: Check for both CUDA (NVIDIA) and MPS (Apple)
                if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    docstrange_mode = 'local_gpu'
                    gpu_type = "CUDA" if torch.cuda.is_available() else "MPS"
                    logging.info(f"Auto mode: {gpu_type} GPU detected → LOCAL GPU (100% private)")
                else:
                    docstrange_mode = 'cloud'
                    logging.info("Auto mode: No supported GPU detected → CLOUD")
            except ImportError:
                docstrange_mode = 'cloud'
                logging.info("Auto mode: PyTorch not available → CLOUD")
        
        # Check local-only constraint
        if args.docstrange_local_only:
            if docstrange_mode not in ['local_gpu', 'local_cpu']:
                logging.error("--docstrange-local-only specified but local processing unavailable")
                logging.error("Install PyTorch with CUDA support for local processing")
                return 1
            logging.info("✓ Privacy mode: 100% local processing (no data sent anywhere)")
        
        # Parse metadata fields
        metadata_fields = None
        if args.docstrange_metadata_fields:
            metadata_fields = [f.strip() for f in args.docstrange_metadata_fields.split(',')]
        
        # Load metadata schema
        metadata_schema = None
        if args.docstrange_metadata_schema:
            try:
                with open(args.docstrange_metadata_schema, 'r') as f:
                    metadata_schema = json.load(f)
                logging.info(f"Loaded metadata schema: {args.docstrange_metadata_schema}")
            except Exception as e:
                logging.error(f"Failed to load metadata schema: {e}")
                return 1
        
        docstrange_config = {
            'mode': docstrange_mode,
            'api_key': args.docstrange_api_key,
            'output_format': args.docstrange_output_format,
            'use_for_metadata': args.docstrange_for_metadata,
            'metadata_fields': metadata_fields,
            'metadata_schema': metadata_schema,
            'local_only': args.docstrange_local_only
        }
        
        # Show configuration
        mode_display = {
            'cloud': '☁️  CLOUD (10k docs/month free with authentication)',
            'local_gpu': '🔒 LOCAL GPU (100% Private)',
            'local_cpu': '🔒 LOCAL CPU (100% Private)'
        }
        logging.info(f"DocStrange: {mode_display[docstrange_mode]}")
    
    # Configure Nanonets-OCR2
    nanonets_config = None
    if args.use_nanonets_ocr2:
        use_gguf = args.nanonets_use_gguf
        use_ollama = args.nanonets_use_ollama
        use_vllm = args.nanonets_vllm_server
        
        # Auto-detect Ollama model format
        if args.nanonets_model == 'custom' and args.nanonets_model_path:
            if ':' in args.nanonets_model_path and '/' in args.nanonets_model_path:
                use_ollama = True
                logging.info(f"Auto-detected Ollama model: {args.nanonets_model_path}")
        
        nanonets_config = {
            'model_name': args.nanonets_model_path if args.nanonets_model == 'custom' else None,
            'use_gguf': use_gguf,
            'gguf_quantization': args.nanonets_gguf_quant,
            'gguf_mmproj_path': args.nanonets_gguf_mmproj_path,
            'gguf_chat_format': args.nanonets_gguf_chat_format,
            'use_ollama': use_ollama,
            'ollama_model': args.nanonets_ollama_model if use_ollama else args.nanonets_model_path,
            'use_vllm_server': use_vllm,
            'vllm_base_url': args.nanonets_vllm_url,
            'device_map': args.nanonets_device,
            'n_ctx': args.nanonets_n_ctx,
            'n_gpu_layers': args.nanonets_n_gpu_layers,
            'max_tokens': args.nanonets_max_tokens,
            'use_for_metadata': args.nanonets_for_metadata
        }
        
        # Show configuration
        if use_ollama:
            logging.info(f"Nanonets-OCR2: Ollama mode ({nanonets_config['ollama_model']})")
        elif use_gguf:
            logging.info(f"Nanonets-OCR2: GGUF mode ({args.nanonets_gguf_quant} quantization)")
        elif use_vllm:
            logging.info(f"Nanonets-OCR2: vLLM server ({args.nanonets_vllm_url})")
        else:
            logging.info(f"Nanonets-OCR2: Transformers mode (model: {args.nanonets_model})")

    # llama-mtmd-cli VL configuration
    llama_mtmd_config = None
    if args.use_llama_mtmd_vl:
        llama_mtmd_config = {
            'model_id': args.llama_mtmd_model,
            'max_tokens': args.llama_mtmd_max_tokens,
            'temperature': args.llama_mtmd_temp,
            'min_p': args.llama_mtmd_min_p,
            'repetition_penalty': args.llama_mtmd_repeat_penalty,
            'extract_metadata': args.llama_mtmd_for_metadata
        }
        if args.verbose > 0:
            logging.info(f"Using llama-mtmd-cli VL with model: {args.llama_mtmd_model}")

    # MLX-VLM configuration
    mlx_vlm_config = None
    if args.use_mlx_vlm:
        mlx_vlm_config = {
            'model_id': args.mlx_vlm_model,
            'max_tokens': args.mlx_vlm_max_tokens,
            'temperature': args.mlx_vlm_temp,
            'extract_metadata': args.mlx_vlm_for_metadata
        }
        if args.verbose > 0:
            logging.info(f"Using mlx-vlm with model: {args.mlx_vlm_model}")
    
    # Periodic logging check
    def check_and_restore_logging():
        root_logger = logging.getLogger()
        if not root_logger.handlers or root_logger.level != logging._biblioforge_target_level:
            print("WARNING: Logging corrupted, restoring...", file=sys.stderr)
            setup_logging(args.verbose)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if platform.system() != 'Windows':
        try:
            signal.signal(signal.SIGTSTP, signal_handler)
        except AttributeError:
            logging.warning("SIGTSTP not available on this platform")
    
    # File discovery
    input_files_discovered: List[str] = []
    all_supported_extensions = list(ExtractionManager.SUPPORTED_EXTENSIONS.keys())
    active_supported_extensions = all_supported_extensions
    
    if args.file_types:
        user_types = [f".{ext.strip().lower()}" for ext in args.file_types.split(',') if ext.strip()]
        active_supported_extensions = [ext for ext in user_types if ext in all_supported_extensions]
        if not active_supported_extensions:
            logging.error(f"No valid file types in --file-types: '{args.file_types}'")
            logging.info(f"Available types: {', '.join(k.lstrip('.') for k in all_supported_extensions)}")
            return 1
        logging.info(f"Processing types: {', '.join(active_supported_extensions)}")
    else:
        logging.info(f"Processing all types: {', '.join(k.lstrip('.') for k in active_supported_extensions)}")
    
    if not args.files:
        scan_path = Path(".").resolve()
        logging.info(f"No input specified. Scanning '{scan_path}'")
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
                # Check if pattern_item is an existing file (shell already expanded it)
                potential_file = Path(pattern_item)
                if potential_file.is_file():
                    # Shell-expanded file - use directly
                    if potential_file.suffix.lower() in active_supported_extensions:
                        input_files_discovered.append(str(potential_file))
                else:
                    # It's a pattern - use glob
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
    
    if final_input_files:
        logging.debug(f"Found {len(final_input_files)} files before filtering")
        final_input_files = filter_out_extracted_txt_files(final_input_files)
        logging.info(f"Processing {len(final_input_files)} files (after filtering)")
    
    if not final_input_files:
        logging.error("No input files found")
        return 1
    
    logging.info(f"Found {len(final_input_files)} file(s) to process")
    if args.verbose > 1:
        for f_idx, f_path in enumerate(final_input_files[:20]):
            logging.debug(f"  File {f_idx+1}: {f_path}")
        if len(final_input_files) > 20:
            logging.debug(f"  ... and {len(final_input_files)-20} more files")
    
    # Initialize processor
    processor = DocumentProcessor(debug=(args.verbose > 1))
    
    # Prepare LLM kwargs
    llm_config_kwargs = {
        'llm_model': args.llm_model,
        'api_key': args.api_key,
        'ollama_host': args.ollama_host,
        'local_openai_base_url': args.local_openai_base_url,
        'ollama_allow_fallback': args.ollama_allow_fallback,
        'ollama_fallback_order': args.ollama_fallback_order,
        'llamacpp_repo_id': args.llamacpp_repo_id,
        'llamacpp_gguf_filename': args.llamacpp_gguf_filename,
        'llamacpp_n_ctx': args.llamacpp_n_ctx,
        'llamacpp_n_gpu_layers': args.llamacpp_n_gpu_layers,
        'llamacpp_chat_format': args.llamacpp_chat_format,
        'debug': (args.verbose > 1)
    }
    llm_config_kwargs_to_pass = {k: v for k, v in llm_config_kwargs.items() if v is not None}
    llm_config_kwargs_to_pass['sort_arg_from_main'] = args.sort

    # Pre-OCR scan cleanup (CrispEmbed). None = disabled; otherwise a dict the
    # PDF extractor uses to clean each page image before OCR. No-op if CrispEmbed
    # isn't installed (see crispembed_adapter / PLAN.md Phase 2).
    scan_cleanup_config = None
    if args.scan_cleanup != 'off':
        _binarize_method = {'off': None, 'otsu': 0, 'sauvola': 1}[args.scan_cleanup_binarize]
        scan_cleanup_config = {
            'mode': args.scan_cleanup,
            'params': {
                'deskew': True,
                'crop_borders': True,
                'whiten_background': True,
                'binarize': _binarize_method is not None,
                'binarize_method': _binarize_method or 0,
            },
        }

    # Optional pre-OCR super-resolution (CrispEmbed). None = disabled.
    super_resolution_config = None
    if args.sr != 'off':
        super_resolution_config = {
            'mode': args.sr,
            'engine': args.sr_engine,
            'min_width': args.sr_min_width,
        }

    # Optional CrispEmbed single-pass OCR backend (only when explicitly selected).
    crispembed_ocr_config = None
    if args.ocr_method == 'crispembed':
        crispembed_ocr_config = {
            'model': args.crispembed_ocr_model,
            'dpi': args.crispembed_ocr_dpi,
            'force_cpu': args.crispembed_ocr_cpu,
        }

    if args.verbose > 1:
        logging.debug(f"LLM config kwargs: {llm_config_kwargs_to_pass}")
    
    try:
        check_and_restore_logging()
        
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
            reset_rename_script=args.reset,
            noremove_from_script=args.noremove,
            llm_provider_arg=args.llm_provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            nanonets_config=nanonets_config,
            docstrange_config=docstrange_config,
            llama_mtmd_config=llama_mtmd_config,
            mlx_vlm_config=mlx_vlm_config,
            scan_cleanup_config=scan_cleanup_config,
            super_resolution_config=super_resolution_config,
            crispembed_ocr_config=crispembed_ocr_config,
            metadata_backend=args.metadata_backend,
            **llm_config_kwargs_to_pass
        )
        
        check_and_restore_logging()

        # Initialize rename scripts
        rename_script_path, existing_script_entries = initialize_rename_scripts(
            args.output_dir, 
            args.rename_script, 
            args.reset,
            args.noremove
        )
        
        # === ADD THIS LOGGING ===
        if args.sort:
            logging.info(f"📝 Rename script path: {rename_script_path}")
            logging.info(f"📊 Existing entries in rename script: {len(existing_script_entries)}")
            if args.verbose >= 1:
                logging.debug(f"Existing script entries: {list(existing_script_entries)[:10]}...")  # Show first 10
        # ========================
        
        # Save JSON if requested
        if args.json:
            json_output_path = args.json
            logging.info(f"Saving results to {json_output_path}")
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
                logging.error(f"Failed to save JSON: {e_json}")
        
        # Handle rename script
        final_rename_script_path_for_user: Optional[str] = None
        if args.sort and args.rename_script:
            if os.path.isabs(args.rename_script) or os.path.dirname(args.rename_script):
                final_rename_script_path_for_user = str(Path(args.rename_script).resolve())
            else:
                path_obj = Path(args.output_dir).resolve() / args.rename_script
                final_rename_script_path_for_user = str(path_obj.resolve())
        
        if args.sort and args.execute_rename and final_rename_script_path_for_user:
            if os.path.exists(final_rename_script_path_for_user):
                logging.info(f"Executing rename commands: {final_rename_script_path_for_user}")
                execute_rename_commands(final_rename_script_path_for_user)
            else:
                logging.warning(f"Rename script not found: {final_rename_script_path_for_user}")
        elif args.sort and final_rename_script_path_for_user:
            if os.path.exists(final_rename_script_path_for_user):
                logging.info(f"Rename script: {final_rename_script_path_for_user}")
                if platform.system() == "Windows":
                    batch_equivalent = os.path.splitext(final_rename_script_path_for_user)[0] + ".bat"
                    if os.path.exists(batch_equivalent):
                        logging.info(f"  Windows: {batch_equivalent}")
                    else:
                        logging.info(f"  Unix: bash {final_rename_script_path_for_user}")
                else:
                    logging.info(f"  To execute: bash {final_rename_script_path_for_user}")
        
        return 0 if not processing_results_summary.get('failed') and processing_results_summary.get('counters', {}).get('failed_extraction', 0) == 0 else 1
    
    except KeyboardInterrupt:
        logging.warning("\nCancelled by user")
        if args.sort and args.rename_script:
            script_path_to_check = args.rename_script
            if not (os.path.isabs(script_path_to_check) or os.path.dirname(script_path_to_check)):
                script_path_to_check = str(Path(args.output_dir).resolve() / script_path_to_check)
            else:
                script_path_to_check = str(Path(script_path_to_check).resolve())
            if os.path.exists(script_path_to_check):
                logging.info(f"Partial rename script: {script_path_to_check}")
        return 130
    except Exception as e:
        check_and_restore_logging()
        logging.error(f"Unexpected error: {e}", exc_info=args.verbose > 1)
        return 1
    finally:
        logging.info("BiblioForge processing finished")

if __name__ == '__main__':
    _exit_code = main()
    # If a CrispEmbed native engine was loaded, bypass C++ static-destructor
    # teardown with os._exit: ggml's Metal backend can abort at exit (a known
    # torch-MPS + ggml-Metal interaction) even though all work succeeded. Output
    # files, JSON, and rename scripts are already written by the time main()
    # returns, so this is safe; we just flush first.
    try:
        import crispembed_adapter as _ca
        if _ca.native_engine_loaded():
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(_exit_code if isinstance(_exit_code, int) else 0)
    except Exception:
        pass
    sys.exit(_exit_code)