import os
import re
import logging
import signal
import platform
import threading
import time
import shutil
import shlex
from typing import Optional, List, Dict, Any, Callable, Tuple, Union 
from datetime import datetime
from pathlib import Path
from tqdm import tqdm 
import subprocess # For run_process and Popen type hint
import sys # <<< ADDED IMPORT FOR SYS

# --- Global Variables / Flags ---
shutdown_flag = threading.Event()
file_lock = threading.Lock()
# Using string forward reference for Popen as a robust way for linters
active_processes: List['subprocess.Popen'] = [] 
extraction_in_progress = threading.Event()


DEBUG_PARSE_METADATA = True # Or manage via environment variable/config

# --- Classes ---
class timeout:
    """Context manager for timeout"""
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.timer = None
        self.original_handler = None

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self):
        if self.seconds > 0 and platform.system() != 'Windows':
            self.original_handler = signal.getsignal(signal.SIGALRM)
            try:
                self.timer = signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.seconds)
            except ValueError as e: 
                logging.debug(f"Timeout: Cannot set SIGALRM: {e}. Timeout will not be enforced by signal.")
                self.timer = None 
                if self.original_handler is not None: 
                     signal.signal(signal.SIGALRM, self.original_handler)
            except Exception as e_alarm:
                logging.debug(f"Timeout: Error setting alarm: {e_alarm}. Timeout will not be enforced by signal.")
                self.timer = None
                if self.original_handler is not None:
                    signal.signal(signal.SIGALRM, self.original_handler)
        return self # It's good practice for context managers to return self


    def __exit__(self, exc_type, exc_value, traceback):
        if self.seconds > 0 and platform.system() != 'Windows':
            signal.alarm(0) 
            if self.timer is not None: 
                current_handler = signal.getsignal(signal.SIGALRM)
                if current_handler == self._timeout_handler: 
                    signal.signal(signal.SIGALRM, self.original_handler if self.original_handler is not None else signal.SIG_DFL)
            elif self.original_handler is not None:
                 signal.signal(signal.SIGALRM, self.original_handler)

class ProgressProxy:
    """Proxy for tqdm progress bar that can update description with engine info"""
    def __init__(self, progress_bar: tqdm): # Type hint for progress_bar
        self.progress_bar = progress_bar
        self.current_engine: Optional[str] = None
        
    def update(self, n: int = 1, engine: Optional[str] = None):
        if engine and engine != self.current_engine:
            self.current_engine = engine
            if hasattr(self.progress_bar, 'set_description'): 
                self.progress_bar.set_description(f"Extracting text [{engine}]")
        if hasattr(self.progress_bar, 'update'):
            self.progress_bar.update(n)

class ImportCache:
    """Global cache for imports and their availability"""
    _instance: Optional['ImportCache'] = None  # Type hint for the singleton instance itself

    # Class-level type hints for instance attributes
    _modules: Dict[str, Any]
    _available: Dict[str, bool]

    def __new__(cls):
        if cls._instance is None:
            # Create the new instance using the superclass's __new__
            new_instance = super().__new__(cls)
            # Initialize instance-specific attributes on this new_instance
            new_instance._modules = {}
            new_instance._available = {}
            # Assign the fully initialized new_instance to the class's _instance
            cls._instance = new_instance
        return cls._instance

    # Methods (is_available, import_module) remain the same
    def is_available(self, module_name: str, submodules: Optional[List[str]] = None) -> bool:
        import importlib.util
        cache_key = f"{module_name}:{','.join(submodules) if submodules else ''}"
        if cache_key not in self._available:
            try:
                if importlib.util.find_spec(module_name) is None:
                    self._available[cache_key] = False
                    return False
                if submodules:
                    for submodule in submodules:
                        full_name = f"{module_name}.{submodule}"
                        if importlib.util.find_spec(full_name) is None:
                            self._available[cache_key] = False
                            return False
                self._available[cache_key] = True
            except Exception:
                self._available[cache_key] = False
        return self._available[cache_key]

    def import_module(self, module_name: str, submodule: Optional[str] = None) -> Any:
        import importlib
        cache_key = f"{module_name}{f'.{submodule}' if submodule else ''}"
        if cache_key not in self._modules:
            try:
                if submodule:
                    main_module = importlib.import_module(module_name)
                    self._modules[cache_key] = getattr(main_module, submodule)
                else:
                    self._modules[cache_key] = importlib.import_module(module_name)
            except ImportError as e:
                logging.debug(f"ImportCache: Failed to import {cache_key}: {e}")
                raise
            except AttributeError as e:
                logging.debug(f"ImportCache: Attribute error for {cache_key}: {e}")
                raise
        return self._modules[cache_key]

# --- Logging Setup ---
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord): # Added type hint for record
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr) # sys is now imported
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(verbosity: int = 0):
    # (Implementation from before, unchanged)
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = levels.get(verbosity, logging.DEBUG)
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    format_str = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s' if verbosity > 1 \
                 else '%(asctime)s - %(levelname)s - %(message)s' if verbosity > 0 \
                 else '%(message)s'
    
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter(format_str))
    root_logger.addHandler(console_handler)
    
    try:
        log_file_path = Path('biblioforge.log').resolve()
        # Use 'a' for append mode to keep logs across runs if desired, or 'w' to overwrite
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8') 
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'))
        root_logger.addHandler(file_handler)
        if level <= logging.INFO: logging.info(f"Logging to console and file: {log_file_path}")
    except Exception as e:
        # Use root logger to log this critical error, as it might happen before console handler is fully up
        root_logger.error(f"Failed to set up file logging to biblioforge.log: {e}")

    import warnings # Local import for this function's scope is fine
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    libraries_to_quiet = ['PIL', 'pdf2image', 'pytesseract', 'pdfminer', 'pypdf', 
                          'camelot', 'pymupdf', 'matplotlib', 'h5py', 'tensorflow']
    for lib_name in libraries_to_quiet:
        logging.getLogger(lib_name).setLevel(logging.WARNING if verbosity < 2 else logging.INFO)
    if verbosity < 2: logging.getLogger('pypdf').setLevel(logging.ERROR)


# --- Filename and Text Utilities ---
def sanitize_filename(name: str) -> str:
    
    if not isinstance(name, str): name = str(name)
    unsafe_chars = {'\\': '-', '/': '-', ':': '-', '*': '', '?': '', '"': '', "'": "", '<': '', '>': '', '|': '-', ';': '', '`': '', '$': '', '&': 'and', '!': '', '#': '', '=': ''}
    for char, replacement in unsafe_chars.items(): name = name.replace(char, replacement)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'\.{2,}', '.', name).rstrip('.')
    if " " in name: 
        name = re.sub(r'(\w)\.', r'\1', name) 
        parts = name.split()
        new_parts = []
        current_initials = ""
        for part in parts:
            if len(part) == 1 and part.isalpha(): current_initials += part
            else:
                if current_initials: new_parts.append(current_initials)
                current_initials = ""
                new_parts.append(part)
        if current_initials: new_parts.append(current_initials)
        name = " ".join(new_parts)
    return name.strip()

def clean_author_name(author_name: str) -> str:
    # (Implementation from before)
    if not isinstance(author_name, str): author_name = str(author_name)
    author_name = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|Rev|Sir)\.?\s+', '', author_name, flags=re.IGNORECASE)
    author_name = re.sub(r'&[a-zA-Z]+;', ' ', author_name)
    author_name = re.sub(r'[^\w\s.\'-áéíóúàèìòùäëïöüÄËÏÖÜâêîôûÂÊÎÔÛñÑçÇ]', ' ', author_name, flags=re.UNICODE)
    author_name = re.sub(r'\s+', ' ', author_name).strip()
    return author_name

def valid_author_name_old(author_name: str) -> bool:
    # (Implementation from before)
    if not author_name or len(author_name.strip()) < 2: return False
    parts = author_name.strip().split()
    if not parts: return False 
    lower_name = author_name.lower()
    if lower_name == "unknown" and len(parts) == 1: return True
    if "unknown" in lower_name and len(parts) == 2 and parts[1].lower() == "unknown": return True
    if any(p in lower_name for p in ["unknownauthor", "lastname", "surname", "firstname", "author", "n a"]): return False
    if not re.match(r'^[\w\s.\'-áéíóúàèìòùäëïöüÄËÏÖÜâêîôûÂÊÎÔÛñÑçÇ]+$', author_name, re.UNICODE): return False
    if any(len(part) < 1 or len(part) > 30 for part in parts): return False
    return True

def valid_author_name(author_name: Optional[str]) -> bool:
    if not author_name:
        return False
    name_lower = author_name.lower().strip()
    # Remove potential prefixes for the check itself
    name_to_check = re.sub(r"^(>?(Main Author|Author):?\s*)", "", name_lower, flags=re.IGNORECASE).strip()
    
    if name_to_check in ["unknownauthor", "unknown author", "n/a", "", "lastname firstname"]: # Added "lastname firstname"
        return False
    if len(name_to_check) < 2: # Minimum length for a meaningful name
        return False
    # Avoid if it's just placeholder words like "author" or "editor"
    if name_to_check in ["author", "editor", "various authors", "et al"]:
        return False
    # Avoid if it's just the prefix '>'
    if name_to_check == ">":
        return False
    return True

def validate_and_fix_year(year_str: str) -> str:
    if not year_str or year_str.lower() == "unknownyear" or year_str.upper() == "YYYY":
        return "UnknownYear"
    
    match = re.search(r'\b(\d{4})\b', year_str)
    if match:
        year = int(match.group(1))
        # Adjust min_year and max_year as needed
        min_year = 500 # Example: manuscripts can be old
        current_year_plus_one = datetime.now().year + 1
        if min_year <= year <= current_year_plus_one:
            return str(year)
    return "UnknownYear"

def validate_and_fix_year_old(year: Optional[str], filename: Optional[str] = None, text_sample: Optional[str] = None) -> str:
    # (Implementation from before)
    current_year_dt = datetime.now().year # Renamed for clarity
    if year and isinstance(year, str) and re.fullmatch(r'\d{4}', year) and 1500 <= int(year) <= current_year_dt + 2:
        return year
    if filename:
        filename_year = extract_year_from_filename(filename)
        if filename_year: return filename_year
    if text_sample:
        year_patterns = [
            r'copyright\s*(?:©|\(c\))?\s*(\d{4})', r'published\s+(?:in\s+)?(\d{4})',
            r'date\s*:\s*(\d{4})', r'\b(1[5-9]\d\d|20[0-2]\d|2030)\b'
        ]
        for pattern in year_patterns:
            matches = re.findall(pattern, text_sample, re.IGNORECASE)
            valid_years_from_pattern = [y for y in matches if y and 1500 <= int(y) <= current_year_dt + 2]
            if valid_years_from_pattern: return max(valid_years_from_pattern)
    return "UnknownYear"

def escape_special_chars(filename: str) -> str:
    # shlex.quote is generally preferred for shell escaping
    return shlex.quote(str(filename))

def extract_year_from_filename(filename: str) -> Optional[str]:
    # (Implementation from before)
    current_year_dt = datetime.now().year # Renamed
    matches = re.findall(r'\b(1[5-9]\d\d|20[0-2]\d|2030)\b', filename)
    if matches:
        valid_years = [y for y in matches if 1500 <= int(y) <= current_year_dt + 2]
        if valid_years: return max(valid_years)
    return None

# --- Language Detection ---
def detect_language(text: str, min_text_length: int = 30, max_sample_length: int = 2000, verbose: bool = False) -> Optional[str]:
    # (Implementation from before)
    if not text or len(text.strip()) < min_text_length:
        if verbose: logging.debug(f"Text too short ({len(text.strip())} chars) for language detection.")
        return None
    sample = text[:max_sample_length].strip()
    lang_code = None
    try:
        from langdetect import detect, DetectorFactory, LangDetectException # Local import
        DetectorFactory.seed = 0 
        lang_code = detect(sample)
        if verbose: logging.debug(f"langdetect detected: {lang_code}")
        return lang_code
    except ImportError:
        if verbose: logging.debug("langdetect not available for language detection.")
    except LangDetectException:
        if verbose: logging.debug(f"langdetect could not detect language for sample: '{sample[:50]}...'")
    try:
        import langid # Local import
        lang_code, confidence = langid.classify(sample)
        if verbose: logging.debug(f"langid detected: {lang_code} (confidence: {confidence})")
        return lang_code
    except ImportError:
        if verbose: logging.debug("langid not available.")
    except Exception as e_langid: 
        if verbose: logging.debug(f"langid error: {e_langid}")
    try:
        import cld3 # Local import
        prediction = cld3.get_language(sample)
        if prediction and prediction.is_reliable:
            lang_code = prediction.language
            if verbose: logging.debug(f"cld3 detected: {lang_code}")
            return lang_code
    except ImportError:
        if verbose: logging.debug("cld3 not available.")
    except Exception as e_cld3:
        if verbose: logging.debug(f"cld3 error: {e_cld3}")
    return _fallback_detect_language(sample)

def _fallback_detect_language(text: str) -> Optional[str]:
    # (Implementation from before)
    counts = {'en': 0, 'de': 0, 'fr': 0, 'es': 0, 'it': 0, 'zh': 0, 'ja': 0, 'ko': 0, 'ru': 0, 'ar': 0}
    common_words = {
        'de': ['der', 'die', 'das', 'und', 'ist', 'ein', 'sich'],
        'fr': ['le', 'la', 'les', 'et', 'est', 'un', 'une', 'pour'],
        'es': ['el', 'la', 'los', 'las', 'y', 'es', 'un', 'una', 'de'],
        'it': ['il', 'la', 'lo', 'gli', 'le', 'e', 'è', 'un', 'una', 'di']
    }
    text_lower_words = text.lower().split()[:100] 
    for lang_key, words in common_words.items():
        if sum(1 for word in text_lower_words if word in words) > 2: return lang_key
    for char in text[:500]: 
        if '\u4e00' <= char <= '\u9fff': counts['zh'] += 1
        elif '\u3040' <= char <= '\u30ff': counts['ja'] += 1
        elif '\uac00' <= char <= '\ud7a3': counts['ko'] += 1
        elif '\u0400' <= char <= '\u04ff': counts['ru'] += 1
        elif '\u0600' <= char <= '\u06ff': counts['ar'] += 1
        elif 'a' <= char.lower() <= 'z' or char in 'äöüÄÖÜß': 
            counts['en'] += 1
            if char in 'äöüÄÖÜß': counts['de'] +=1 
    if counts['de'] > 5 : return 'de' # Added condition for German specific chars
    dominant_non_latin_lang, max_non_latin_count = None, 0
    for lang_key in ['zh', 'ja', 'ko', 'ru', 'ar']:
        if counts[lang_key] > max_non_latin_count and counts[lang_key] > 5: 
            max_non_latin_count = counts[lang_key]
            dominant_non_latin_lang = lang_key
    if dominant_non_latin_lang: return dominant_non_latin_lang
    if counts['en'] > 10: return 'en'
    return None

# --- Process Management ---
def run_process(cmd: List[str], timeout_sec: Optional[int] = None, **kwargs) -> subprocess.CompletedProcess:
    # (Implementation from before, ensuring globals are from this module)
    global active_processes, extraction_in_progress 
    extraction_in_progress.set()
    cmd_str = [str(c) for c in cmd]
    process = subprocess.Popen(
        cmd_str, stdout=kwargs.get('stdout', subprocess.PIPE),
        stderr=kwargs.get('stderr', subprocess.PIPE), text=kwargs.get('text', True),
        creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' and not kwargs.get('shell') else 0
    )
    active_processes.append(process)
    stdout, stderr = None, None
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        logging.warning(f"Process {' '.join(cmd_str)} timed out after {timeout_sec}s. Terminating.")
        process.kill()
        try: stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired: logging.warning(f"Process {process.pid} did not respond to kill quickly.")
        except Exception as e_comm_kill: logging.debug(f"Error during communicate after kill for {process.pid}: {e_comm_kill}")
        returncode = -9 
    except KeyboardInterrupt: 
        logging.warning(f"Process {' '.join(cmd_str)} interrupted by user.")
        process.terminate()
        try: process.wait(timeout=2)
        except subprocess.TimeoutExpired: process.kill()
        raise 
    except Exception as e_comm:
        logging.error(f"Error during process communication for {' '.join(cmd_str)}: {e_comm}")
        returncode = process.returncode if process.poll() is not None else -1 
        if process.poll() is None: process.kill()
    finally:
        if process in active_processes: active_processes.remove(process)
        if not active_processes: extraction_in_progress.clear()
    return subprocess.CompletedProcess(args=cmd_str, returncode=returncode, stdout=stdout or "", stderr=stderr or "")

# --- File Operations & Renaming Script Logic ---
def is_file_in_rename_script(rename_script_path: str, input_file: str) -> bool:
    # (Implementation from before)
    is_windows = platform.system() == 'Windows'
    script_ext = os.path.splitext(rename_script_path)[1].lower()
    scripts_to_check = [rename_script_path]
    if is_windows and script_ext != '.bat':
        batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
        if os.path.exists(batch_script_path): scripts_to_check.append(batch_script_path)
    abs_input_file = os.path.abspath(input_file) # Normalize input file path

    for script_path_check in scripts_to_check: # Renamed script_path
        if not os.path.exists(script_path_check): continue
        try:
            with open(script_path_check, 'r', encoding='utf-8') as f: content = f.read()
            # Check various forms of the path
            
            # Construct the doubly escaped path for batch files separately
            win_path_doubly_escaped_for_fstring_literal = abs_input_file.replace("/", "\\\\")

            paths_to_check_in_script = [
                abs_input_file, 
                abs_input_file.replace('/', '\\'), 
                shlex.quote(abs_input_file),
                f'"{abs_input_file}"', 
                f'"{win_path_doubly_escaped_for_fstring_literal}"' # Use the pre-constructed string
            ]
            if any(p in content for p in paths_to_check_in_script): return True
        except Exception as e:
            logging.error(f"Error checking rename script {script_path_check}: {e}")
    return False

def initialize_rename_scripts(rename_script_path_base: str) -> Dict[str, Optional[str]]:
    # (Implementation from before)
    is_windows = platform.system() == 'Windows'
    base_name, main_ext = os.path.splitext(rename_script_path_base)
    actual_bash_path, actual_batch_path = rename_script_path_base, rename_script_path_base
    if main_ext.lower() == '.bat': actual_bash_path = base_name + '.sh'
    elif main_ext.lower() == '.sh':
        if is_windows: actual_batch_path = base_name + '.bat'
    else:
        actual_bash_path = rename_script_path_base 
        if is_windows: actual_batch_path = base_name + '.bat'
    
    # Create bash script if it's the main path or if not on windows (even if main path is .bat)
    if not is_windows or actual_bash_path == rename_script_path_base or main_ext.lower() != '.bat':
        with file_lock:
            with open(actual_bash_path, "w", encoding='utf-8') as bash_file:
                bash_file.write("#!/bin/bash\n# Encoding: UTF-8\nset -e\n\n")
        try: os.chmod(actual_bash_path, 0o755)
        except Exception as e: logging.warning(f"Could not chmod {actual_bash_path}: {e}")
        logging.debug(f"Initialized bash script: {actual_bash_path}")
    
    if is_windows: # Always create/overwrite .bat on Windows
        with file_lock:
            with open(actual_batch_path, "w", encoding='utf-8') as batch_file:
                batch_file.write("@echo off\nchcp 65001 > nul\nsetlocal enabledelayedexpansion\n\nrem Rename script for Windows\n\n")
        logging.debug(f"Initialized batch script: {actual_batch_path}")
        
    return {
        'bash_script': actual_bash_path if (not is_windows or actual_bash_path == rename_script_path_base or main_ext.lower() != '.bat') else None,
        'batch_script': actual_batch_path if is_windows else None
    }


def add_rename_command(
    rename_script_paths: Dict[str, Optional[str]], 
    source_path: str, 
    target_dir_name: str, 
    new_filename_base: str, 
    output_dir_for_sorted_files: str, 
    debug: bool = False
) -> Optional[Dict[str, str]]:
    """
    Adds rename commands to the specified script files and returns rename details.
    """
    if debug:
        logging.debug(
            f"add_rename_command: Input - source_path='{source_path}', target_dir_name='{target_dir_name}', "
            f"new_filename_base='{new_filename_base}', output_dir_for_sorted_files='{output_dir_for_sorted_files}'"
        )
        logging.debug(f"add_rename_command: Script paths received: {rename_script_paths}")

    target_dir_name_sanitized = sanitize_filename(target_dir_name.replace(',', ''))
    new_filename_base_sanitized = sanitize_filename(new_filename_base)
    
    if not target_dir_name_sanitized: 
        target_dir_name_sanitized = "UnknownAuthor"
        if debug: logging.debug("Sanitized target_dir_name defaulted to 'UnknownAuthor'")
    if not new_filename_base_sanitized: 
        new_filename_base_sanitized = "UnknownTitle"
        if debug: logging.debug("Sanitized new_filename_base defaulted to 'UnknownTitle'")

    # Ensure output_dir_for_sorted_files is an absolute path
    # Path(output_dir_for_sorted_files).resolve() handles this.
    # The target directory for the author will be inside this.
    full_target_dir_abs = Path(output_dir_for_sorted_files).resolve() / target_dir_name_sanitized
    
    orig_ext = Path(source_path).suffix.lower()
    # Ensure final_new_filename doesn't have double extensions if orig_ext is already in new_filename_base_sanitized
    if new_filename_base_sanitized.lower().endswith(orig_ext) and orig_ext: # Check if orig_ext is not empty
        final_new_filename = new_filename_base_sanitized
    else:
        final_new_filename = new_filename_base_sanitized + orig_ext
    
    final_new_filename = re.sub(r'\.{2,}', '.', final_new_filename) # Consolidate multiple dots

    # Determine paths for the associated .txt file
    # Assume .txt files were initially created in output_dir_for_sorted_files (same as eff_output_dir in DocumentProcessor)
    txt_source_basename = Path(source_path).stem + ".txt"
    txt_source_path_abs = Path(output_dir_for_sorted_files).resolve() / txt_source_basename
    
    txt_target_filename = Path(final_new_filename).stem + ".txt"
    txt_target_path_abs = full_target_dir_abs / txt_target_filename

    if debug:
        logging.debug(f"add_rename_command: Original source='{source_path}'")
        logging.debug(f"add_rename_command: Target directory for author='{full_target_dir_abs}'")
        logging.debug(f"add_rename_command: New filename with extension='{final_new_filename}'")
        if os.path.exists(str(txt_source_path_abs)):
            logging.debug(f"add_rename_command: Associated TXT source='{txt_source_path_abs}'")
            logging.debug(f"add_rename_command: Associated TXT target='{txt_target_path_abs}'")
        else:
            logging.debug(f"add_rename_command: No associated TXT file found at '{txt_source_path_abs}'")

    commands_written_to_any_script = False

    bash_script_p = rename_script_paths.get('bash_script')
    if bash_script_p:
        if debug: logging.debug(f"add_rename_command: Appending to BASH script: {bash_script_p}")
        with file_lock:
            with open(bash_script_p, "a", encoding='utf-8') as bash_f:
                bash_f.write(f"\n# Renaming for: {os.path.basename(source_path)}\n")
                bash_f.write(f"mkdir -p {escape_special_chars(str(full_target_dir_abs))}\n")
                bash_f.write(f"mv -v {escape_special_chars(os.path.abspath(source_path))} {escape_special_chars(str(full_target_dir_abs / final_new_filename))}\n")
                if os.path.exists(str(txt_source_path_abs)):
                    bash_f.write(f"if [ -f {escape_special_chars(str(txt_source_path_abs))} ]; then\n")
                    bash_f.write(f"  mv -v {escape_special_chars(str(txt_source_path_abs))} {escape_special_chars(str(txt_target_path_abs))}\n")
                    bash_f.write(f"fi\n")
                bash_f.write("\n")
        commands_written_to_any_script = True
    
    batch_script_p = rename_script_paths.get('batch_script')
    if batch_script_p:
        if debug: logging.debug(f"add_rename_command: Appending to BATCH script: {batch_script_p}")
        with file_lock:
            with open(batch_script_p, "a", encoding='utf-8') as batch_f:
                batch_f.write(f"\nREM Renaming for: {os.path.basename(source_path)}\n")
                batch_f.write(f"if not exist \"{str(full_target_dir_abs)}\" mkdir \"{str(full_target_dir_abs)}\"\n")
                batch_f.write(f"move /Y \"{os.path.abspath(source_path)}\" \"{str(full_target_dir_abs / final_new_filename)}\"\n")
                if os.path.exists(str(txt_source_path_abs)):
                    batch_f.write(f"if exist \"{str(txt_source_path_abs)}\" (\n")
                    batch_f.write(f"  move /Y \"{str(txt_source_path_abs)}\" \"{str(txt_target_path_abs)}\"\n")
                    batch_f.write(f")\n")
                batch_f.write("\n")
        commands_written_to_any_script = True

    if commands_written_to_any_script:
        rename_details = {
            "original_source_path": source_path,
            "target_directory_path": str(full_target_dir_abs),
            "new_filename_with_ext": final_new_filename,
            "associated_txt_original_path": str(txt_source_path_abs) if os.path.exists(str(txt_source_path_abs)) else None,
            "associated_txt_new_path": str(txt_target_path_abs) if os.path.exists(str(txt_source_path_abs)) else None,
        }
        if debug: logging.debug(f"add_rename_command: Successfully formulated rename details: {rename_details}")
        return rename_details
    else:
        if debug: logging.debug("add_rename_command: No script paths provided or an issue occurred. No commands written.")
        return None


def execute_rename_commands(script_path_to_execute: str):
    # (Implementation from before)
    if not os.path.exists(script_path_to_execute):
        logging.error(f"Rename script '{script_path_to_execute}' not found.")
        return
    try:
        if platform.system() == "Windows":
            if not script_path_to_execute.lower().endswith(".bat"):
                logging.warning(f"Attempting to execute non-batch script '{script_path_to_execute}' on Windows. This might fail.")
            subprocess.run(['cmd', '/c', script_path_to_execute], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.run(['bash', script_path_to_execute], check=True)
        logging.info(f"Successfully executed rename commands from {script_path_to_execute}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing {script_path_to_execute}: {e}. Stdout: {e.stdout}. Stderr: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error executing {script_path_to_execute}: {e}")


# --- Metadata Parsing ---
def _extract_tag_content(content: str, tag: str, default: str = "") -> str:
    logging.debug(f"Extracting tag content from {content} for {tag}.")
    # Allow optional spaces around tag name in closing tag and content
    # Handle attributes in opening tag
    match = re.search(f"<{tag}[^>]*>(.*?)</\s*{tag}\s*>", content, re.DOTALL | re.IGNORECASE)
    # If no match, or match is empty after stripping, return default
    if not match or not match.group(1).strip():
        return default
    found = match.group(1).strip()

    logging.debug(f"... find: {found}.")
    return found

def parse_metadata(content: str, filename: str = "Unknown Filename") -> Dict[str, str]:
    raw_title = _extract_tag_content(content, 'TITLE')
    raw_year = _extract_tag_content(content, 'YEAR')
    raw_author = _extract_tag_content(content, 'AUTHOR')
    raw_lang = _extract_tag_content(content, 'LANGUAGE')

    title = re.sub(r"", "", raw_title).strip()
    if not title: title = "UnknownTitle"

    year = raw_year
    # Check if year is empty, "YYYY", or clearly not a year before passing to validate_and_fix_year
    if not year or year.upper() == "YYYY" or not re.match(r"^\d{4}$", year):
        year = "UnknownYear" # validate_and_fix_year will also confirm this
    
    author = raw_author # The raw content of the AUTHOR tag
    if not author: author = "UnknownAuthor"
    # Remove "Main Author: Lastname Firstname:" or "Main Author:" like prefixes if LLM adds them *inside* the tag
    author = re.sub(r"^(Main Author: Lastname Firstname:|Main Author:)\s*", "", author, flags=re.IGNORECASE).strip()


    lang = raw_lang.lower()
    # Further sanitize if it ends with '</language>' due to bad LLM output inside the tag
    # Handle cases like <LANGUAGE>de</ LANGUAGE >
    if lang.endswith("</language>"): # A defensive check if LLM includes closing tag in content
        lang = lang[:-len("</language>")].strip()

    if not lang or lang == "lg" or lang == "language": # "lg" or "language" are placeholders from prompt
        lang = "ul" # Use "ul" for unknown language
    elif len(lang) > 2: # If LLM gave full language name
        lang_map = {
            "english": "en", "german": "de", "deutsch": "de", "french": "fr", "français": "fr",
            "spanish": "es", "español": "es", "italian": "it", "italiano": "it", "dutch": "nl", "nederlands": "nl"
        }
        norm_lang = lang.split('(')[0].strip().lower() # Handle "English (US)"
        if norm_lang in lang_map:
            lang = lang_map[norm_lang]
        elif re.match(r"^[a-z]{2}$", norm_lang[:2]): # If first 2 letters look like a code e.g. "de-DE"
            lang = norm_lang[:2]
        else:
            lang = "ul" # Fallback if longer than 2 and not recognized
    elif not re.match(r"^[a-z]{2}$", lang): # If it's 1 or 2 letters but not valid form
        lang = "ul"


    parsed = {'title': title, 'year': year, 'author': author, 'language': lang}
    if DEBUG_PARSE_METADATA:
        logging.debug(f"parse_metadata for {filename}: Raw extracted: Title='{raw_title}', Year='{raw_year}', Author='{raw_author}', Lang='{raw_lang}'")
        logging.debug(f"parse_metadata for {filename}: Cleaned for dict: Author='{author}', Lang='{lang}'")
        logging.debug(f"parse_metadata for {filename}: Final Parsed Dict: {parsed}")
    return parsed

def parse_metadata_old(content: str, verbose: bool = False) -> Optional[Dict[str, str]]:
    # (Implementation from before)
    if verbose: logging.debug(f"Parsing metadata content: '{content[:200]}...'")
    content = re.sub(r'<\?xml[^>]+\?>', '', content).strip()
    content = content.replace("<TITLE", "<TITLE>").replace("<AUTHOR", "<AUTHOR>")
    content = content.replace("<YEAR", "<YEAR>").replace("<LANGUAGE", "<LANGUAGE>")
    title_match = re.search(r'<TITLE[^>]*>(.*?)</TITLE>', content, re.DOTALL | re.IGNORECASE)
    year_match = re.search(r'<YEAR[^>]*>(\d{4})</YEAR>', content, re.DOTALL | re.IGNORECASE)
    author_match = re.search(r'<AUTHOR[^>]*>(.*?)</AUTHOR>', content, re.DOTALL | re.IGNORECASE)
    lang_match = re.search(r'<LANGUAGE[^>]*>([a-zA-Z]{2,3})</LANGUAGE>', content, re.DOTALL | re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else None
    year = year_match.group(1).strip() if year_match else "UnknownYear"
    author = author_match.group(1).strip() if author_match else None
    language = lang_match.group(1).strip().lower() if lang_match else "en"
    if author:
        if ';' in author: author = author.split(';')[0].strip()
        author = re.sub(r'&[a-zA-Z]+;', ' ', author)
        author = re.sub(r'\b(Lastname|Firstname|Surname|Autor|Auteur)\b', '', author, flags=re.IGNORECASE)
        author = re.sub(r'\s+', ' ', author).strip()
        if author.lower() in ["lastname firstname", "surname firstname", "", "n/a", "na", "unknown"]: author = None # Added unknown
    if not title or not author :
        if verbose: logging.warning(f"Metadata parsing failed: Title='{title}', Author='{author}'. Content: '{content[:100]}...'")
        return None
    title = re.sub(r'\s*\([^)]*edition\)$|\s*\([^)]*ed\.\)$', '', title, flags=re.IGNORECASE).strip()
    final_metadata = {'author': author, 'year': year, 'title': title, 'language': language}
    if verbose: logging.debug(f"Successfully parsed metadata: {final_metadata}")
    return final_metadata

# --- GPU and Memory Utilities (NEWLY ADDED/MOVED HERE) ---
def _validate_text(text: str, min_length: int = 50, printable_threshold:float = 0.7) -> bool:
    """Validate extracted text quality."""
    if not text or not isinstance(text, str) or len(text.strip()) < min_length:
        logging.debug(f"_validate_text: Text too short or not string (length: {len(text.strip()) if text else 0})")
        return False
    
    # Check for excessive non-printable characters
    text_len = len(text) # Total length including whitespace for ratio
    if text_len == 0: return False

    non_printable_count = sum(1 for c in text if not c.isprintable() and c not in '\n\r\t ')
    printable_char_ratio = 1.0 - (non_printable_count / text_len)
    
    if printable_char_ratio < printable_threshold:
        logging.debug(f"_validate_text: Low printable character ratio: {printable_char_ratio:.2f} (threshold: {printable_threshold})")
        return False
        
    # Check for very short average word length if there are enough words
    words = text.split()
    if len(words) > 10: # Only check if there's a decent number of words
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length < 2.5: # Heuristic for garbled text
            logging.debug(f"_validate_text: Average word length too short: {avg_word_length:.2f}")
            return False
            
    # Check if text consists of mostly the same character (e.g., "AAAAA..." or "~~~~~...")
    if len(set(text.strip())) < 5 and len(text.strip()) > min_length: # Few unique chars
        logging.debug(f"_validate_text: Text has very few unique characters: {set(text.strip())}")
        return False
        
    return True

def _recover_from_error(error: Exception, context: str = "") -> Optional[str]:
    """Try to suggest recovery steps based on error message."""
    error_str = str(error).lower()
    
    if isinstance(error, MemoryError) or "memory" in error_str:
        _clear_memory(verbose=True) # Attempt to clear memory
        return f"Memory error in {context} - cleared memory. Consider processing smaller files or fewer files at once."
    if "pdf file is encrypted" in error_str or "password" in error_str:
        return f"PDF in {context} is encrypted. Try providing a password with the -p option."
    if "damaged" in error_str or "corrupt" in error_str:
        return f"File in {context} appears to be damaged or corrupted."
    if "no text extractable" in error_str or "no text found" in error_str:
        return f"No direct text found in {context}. If it's an image-based file, ensure OCR is attempted/forced."
    if "permission" in error_str or "access denied" in error_str:
        return f"Permission error for file in {context}. Check file access rights."
    if "timeout" in error_str or "timed out" in error_str:
        return f"Operation timed out during {context}. The file might be too complex or large. Try again or increase timeout if possible."
    if "not found" in error_str and ("tesseract" in error_str or "gs" in error_str or "poppler" in error_str or "djvutxt" in error_str):
        return f"A required system dependency for {context} (e.g., Tesseract, Ghostscript, Poppler, DjVuLibre) was not found. Please install it."
    
    # Specific library errors
    if "AttributeError" in str(error) and "Kraken" in context : # Example specific error
         return f"Kraken OCR error in {context}, possibly due to model or TensorFlow issues. Check Kraken setup. Error: {error_str[:100]}"

    return f"An undefined error occurred in {context}: {error_str[:150]}" # Generic if no specific match

def _clear_memory(verbose: bool = False):
    """Attempts to clear memory, including GPU memory if supported libraries are present."""
    # (Implementation from before)
    import gc
    gc.collect()
    if verbose: logging.debug("Ran Python garbage collection.")
    try:
        torch_module = ImportCache().import_module('torch')
        if torch_module and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
            if verbose: logging.debug("Cleared PyTorch CUDA cache.")
    except Exception: pass
    try:
        tf_module = ImportCache().import_module('tensorflow')
        if tf_module and hasattr(tf_module.keras.backend, 'clear_session'):
            tf_module.keras.backend.clear_session()
            if verbose: logging.debug("Cleared TensorFlow Keras session.")
    except Exception: pass
    try:
        paddle_module = ImportCache().import_module('paddle')
        if paddle_module and hasattr(paddle_module.device.cuda, 'empty_cache') and paddle_module.device.cuda.device_count() > 0 :
            paddle_module.device.cuda.empty_cache()
            if verbose: logging.debug("Cleared PaddlePaddle CUDA cache.")
    except Exception: pass


def _check_gpu_available(verbose: bool = False) -> bool:
    """Checks for common GPU libraries and logs availability."""
    # (Implementation from before)
    try:
        torch_module = ImportCache().import_module('torch')
        if torch_module and torch_module.cuda.is_available():
            if verbose: logging.info(f"PyTorch CUDA available: {torch_module.cuda.get_device_name(0)}")
            return True
        if torch_module and hasattr(torch_module.backends, 'mps') and torch_module.backends.mps.is_available():
            if verbose: logging.info("PyTorch MPS (Apple Silicon GPU) available.")
            return True
    except Exception: pass
    try:
        tf_module = ImportCache().import_module('tensorflow')
        if tf_module:
            gpus = tf_module.config.list_physical_devices('GPU')
            if gpus:
                if verbose: logging.info(f"TensorFlow GPU available: {gpus}")
                return True
    except Exception: pass
    if verbose: logging.info("No primary GPU (CUDA/MPS for PyTorch, TensorFlow GPU) detected by utils._check_gpu_available.")
    return False


# --- Fallback get_openai_client (should be phased out) ---
def get_openai_client():
    """Initialize or return thread-local OpenAI client specifically for Ollama (legacy)."""
    # (Implementation from before)
    try:
        from openai import OpenAI as OpenAIClient_ # Local import
    except ImportError:
        logging.critical("OpenAI client library not available for get_openai_client. pip install openai")
        raise

    if not hasattr(thread_local, "ollama_direct_client_legacy"):
        try:
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1/"
            logging.debug(f"Utils: Initializing legacy OpenAI client for Ollama: {ollama_url}")
            thread_local.ollama_direct_client_legacy = OpenAIClient_(base_url=ollama_url, api_key="ollama")
        except Exception as e:
            logging.critical(f"Failed to initialize legacy OpenAI client for Ollama: {e}")
            raise
    return thread_local.ollama_direct_client_legacy