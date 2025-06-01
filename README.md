# BiblioForge: Document Text Extraction & Organization Tool

BiblioForge is a versatile, cross-platform command-line tool designed to extract text from a wide array of document formats and intelligently organize them. It leverages multiple extraction engines, OCR capabilities, and AI-powered metadata analysis to sort and rename your documents into a structured library.

**Status:** This project is under active development. While functional, expect ongoing improvements and potential changes.

## Core Features

* **Multi-Format Text Extraction:** Supports PDF, EPUB, DJVU, MOBI, various text formats (TXT, MD), HTML, and common office formats (DOCX, RTF, ODT via Calibre).
* **Smart Fallbacks:** Employs multiple extraction methods for each format, ensuring the best possible text recovery.
* **OCR for Scanned Documents:** Integrated OCR engines (Tesseract, PaddleOCR, EasyOCR, DocTR, Kraken) to process image-based PDFs and other scanned documents.
* **Table Extraction:** Capable of extracting tabular data from PDF files (using Camelot).
* **AI-Powered Sorting & Renaming:**
    * Utilizes Large Language Models (LLMs) to analyze content and extract key metadata (author, title, year, language).
    * Generates a shell script (`rename_commands.sh` or custom) to organize files into an `Author/Year Title.ext` structure.
    * Option to automatically execute the rename script.
* **Flexible LLM Integration:** Supports local LLMs via Ollama (recommended for privacy and cost) and various cloud-based LLM providers (OpenAI, Groq, Cohere, etc.).
* **Customizable Processing:** Offers fine-grained control over extraction methods, OCR engines, file types, and more.
* **Cross-Platform:** Designed to run on Windows, macOS, and Linux.

## Installation

### 1. Python Environment

Python 3.8+ is recommended.

**Create a virtual environment (recommended):**
```bash
python -m venv biblioforge_env
# On macOS/Linux:
source biblioforge_env/bin/activate
# On Windows (cmd):
# biblioforge_env\Scripts\activate.bat
# On Windows (PowerShell):
# biblioforge_env\Scripts\Activate.ps1
```

### 2. Python Dependencies

Install the required Python packages using pip:

```bash
pip install pymupdf pdfplumber pypdf pdfminer.six pytesseract pdf2image tqdm openai \
            ebooklib beautifulsoup4 html2text mobi chardet ftfy lxml requests \
            groq cohere huggingface_hub easyocr paddleocr python-doctr ocrmypdf \
            python-Levenshtein # For potential string matching improvements (optional but good)
            
# For table extraction:
pip install "camelot-py[cv]"

# For Kraken OCR (can be complex, ensure system deps like Rust are met if building from source):
pip install kraken
```

**Note:** Some packages like `paddleocr` or `kraken` might have additional system-level build dependencies (e.g., C++ compilers, Rust).

### 3. System Dependencies

Certain functionalities, especially OCR and some format conversions, rely on external system tools.

**General Recommendation:**

* **Calibre:** For robust conversion of many formats (DOCX, RTF, etc.) to text. Install the Calibre application. BiblioForge will try to use its `ebook-converter` command-line tool if it's in your system's PATH.

**Platform-Specific:**

#### macOS:
```bash
brew install tesseract poppler ghostscript djvulibre
# For Calibre, download from the website or use:
# brew install --cask calibre (if available and preferred)
```

Ensure tesseract data files for your desired languages are installed (e.g., `brew install tesseract-lang`).

#### Linux (Debian/Ubuntu based):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-all poppler-utils ghostscript djvulibre-bin \
                        libgl1-mesa-glx libglib2.0-0 # Common deps for CV/OCR libs
# For Calibre:
# sudo -v && wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sudo sh /dev/stdin
```

#### Windows:

Manual installation of the following is typically required. Ensure they are added to your system's PATH.

* **Tesseract OCR:** Download installer from [UB-Mannheim Tesseract builds](https://github.com/tesseract-ocr/tesseract). Install language data during setup.
* **Poppler for Windows:** Download binaries (e.g., from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)). Add the `bin/` directory to PATH.
* **Ghostscript:** Download installer from [Ghostscript releases](https://www.ghostscript.com/download/gsdnld.html). Add its `bin/` and `lib/` directories to PATH.
* **DjVuLibre for Windows:** May be available from projects like DjView4 which might bundle the command-line tools.
* **Calibre:** Download and install from the [Calibre website](https://calibre-ebook.com/). Ensure its installation directory (containing `ebook-converter.exe`) is in PATH.

### 4. LLM Setup (for --sort feature)

Choose one of the following:

#### Option A: Local LLM with Ollama (Recommended)

1. **Install Ollama:** Follow instructions at [ollama.ai](https://ollama.ai).
2. **Pull a Model:** A smaller, faster model is often sufficient for metadata tasks.
   ```bash
   ollama pull llama3:8b # A good general-purpose model
   # or for very fast responses:
   ollama pull qwen2:1.5b 
   ollama pull phi3
   # The script defaults to 'cas/llama-3.2-3b-instruct:latest' or a similar small model if available
   ```
3. **Ensure Ollama Server is Running:** Usually, Ollama runs as a background service after installation. If not, you might need to start it manually (`ollama serve`).

#### Option B: Cloud LLM Providers

Set the appropriate environment variable for your chosen provider:

```bash
# For OpenAI (e.g., GPT-4, GPT-3.5-turbo)
export OPENAI_API_KEY="your-openai-api-key"

# For Groq (fast LLM inference)
export GROQ_API_KEY="your-groq-api-key"

# For Cohere
export COHERE_API_KEY="your-cohere-api-key"

# For GLHF.chat (HuggingFace models via OpenAI-compatible API)
export GLHF_API_KEY="your-glhf-api-key" # Often 'glhf-'. Visit glhf.chat for details.

# For HuggingFace Inference Endpoints/API (direct)
export HF_API_KEY="your-huggingface-api-key" 
# (Note: Direct HF API usage might require specific model endpoint configuration not covered by default)
```

On Windows, use `set` or `setx` in Command Prompt, or `$Env:VAR_NAME = "value"` in PowerShell to set environment variables.

## Usage Examples

```bash
python BiblioForge.py [options] file_or_pattern1 [file_or_pattern2 ...]
```

### Basic Extraction:
```bash
# Extract text from a single PDF to the current directory
python BiblioForge.py document.pdf

# Extract from all PDFs in current directory to an 'output' subdirectory
python BiblioForge.py -o output/ *.pdf

# Specify a preferred PDF extraction method
python BiblioForge.py --method=pymupdf document.pdf
```

### Recursive Processing & File Types:
```bash
# Process all supported files recursively in 'my_library'
python BiblioForge.py -r my_library/

# Process only PDF and EPUB files recursively
python BiblioForge.py --file-types="pdf,epub" -r my_library/
```

### OCR Processing:
```bash
# Force OCR using Tesseract on a scanned PDF
python BiblioForge.py --force-ocr --ocr-method=tesseract scanned_document.pdf
```

### Sorting and Renaming (Requires LLM Setup):
```bash
# Analyze PDFs, generate 'rename_commands.sh' to sort them
python BiblioForge.py --sort --noskip --output-dir ./organized_docs/ *.pdf

# Use a specific LLM provider (e.g., Groq) and execute renames
python BiblioForge.py --sort --llm-provider=groq --execute-rename --output-dir ./groq_sorted/ documents/*.epub

# Use a specific Ollama model
python BiblioForge.py --sort --llm-provider=ollama --llm-model=qwen2:1.5b documents/
```

**Notes:**
- `--noskip` is recommended with `--sort` to ensure all files are processed for metadata, even if .txt files exist.
- `--output-dir` with `--sort` specifies where the new `Author/Year Title.ext` structure will be created.

### Table Extraction (PDFs):
```bash
python BiblioForge.py --tables financial_report.pdf
```

### Debugging:
```bash
# See detailed logs for troubleshooting
python BiblioForge.py --debug document.pdf
```

## Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `files` | | Input files or patterns to process (e.g., `*.pdf`, `"docs/*.epub"`). | (None) |
| `--output-dir` | `-o` | Base directory for extracted .txt files and the sorted/renamed file structure. | `.` (current directory) |
| `--method` | `-m` | Preferred primary extraction method (e.g., `pymupdf` for PDF). Varies by file type. | (auto) |
| `--ocr-method` | | Preferred OCR method: `auto`, `tesseract`, `paddleocr`, `doctr`, `easyocr`, `kraken`, `kraken_cli`. | `auto` |
| `--force-ocr` | | Force OCR for all pages, even if a text layer is detected. | (False) |
| `--recursive` | `-r` | Process files recursively in subdirectories. | (False) |
| `--password` | `-p` | Password for encrypted documents. | (None) |
| `--tables` | `-t` | Attempt to extract tables (primarily for PDF files using Camelot). | (False) |
| `--json` | `-j` | Save detailed processing results (including extracted text and metadata) to a JSON file. Path to JSON file. | (None) |
| `--workers` | `-w` | Maximum number of worker threads for parallel processing. | (auto-detected) |
| `--noskip` | | Re-process files and overwrite/create unique .txt output, even if it already exists. Essential for `--sort`. | (False) |
| `--file-types` | | Comma-separated list of file extensions to process (e.g., `pdf,epub`). Processes all supported if not set. | (All supported) |
| `--sort` | | Enable LLM-based metadata extraction, sorting, and generation of rename commands. | (False) |
| `--rename-script` | | Filename for the generated rename script when `--sort` is active. Path is relative to `--output-dir`. | `rename_commands.sh` |
| `--execute-rename` | | Automatically execute the generated rename script after processing. Use with caution. | (False) |
| `--llm-provider` | | LLM provider for `--sort`: `ollama`, `groq`, `cohere`, `openai`, `glhf`, `huggingface`, `poe`. | `ollama` |
| `--llm-model` | | Specific model name for the chosen LLM provider (e.g., `llama3:8b`, `gpt-4-turbo`). | (Provider's default) |
| `--api-key` | | API key for cloud-based LLM providers (if not set as an environment variable). | (None) |
| `--temperature` | | LLM temperature for metadata/author name tasks (0.0-2.0). | `0.3` |
| `--max-tokens` | | LLM max tokens for metadata/author name tasks. | `250` |
| `--verbose / --debug` | `-v/-d` | Increase logging verbosity: `-v` for INFO, `-vv` or `-d` for DEBUG. | (WARNING level) |

## Available Extraction Methods by Format

BiblioForge tries methods in a preferred order, falling back if one fails.

* **PDF:** `pymupdf`, `calibre`, `pdfplumber`, `pypdf`, `pdfminer` (Text Layer); `tesseract`, `easyocr`, `paddleocr`, `doctr`, `kraken`, `kraken_cli` (OCR)
* **EPUB:** `ebooklib` (with BeautifulSoup), `bs4` (manual unpack with BeautifulSoup & html2text), `epub2txt` (if installed), `calibre`, `zipfile` (raw text)
* **DJVU:** `djvulibre` (Python bindings or djvutxt CLI), `pdf_conversion` (via ddjvu then PDF extraction), `ocr` (via ddjvu to images then Tesseract)
* **MOBI/AZW:** `mobi` (Python library), `kindleunpack` (if available), `calibre`, `zipfile` (raw text)
* **HTML/XHTML:** `bs4` (BeautifulSoup), `html2text`, `lxml`, `regex` (basic stripping)
* **TXT/MD** (and other text-like formats via Calibre, e.g., DOCX, RTF, ODT): `direct read`, `charset_detection` (chardet), `encoding_detection` (ftfy), `calibre` (for non-plain-text formats)

## Output Structure (with --sort)

When using `--sort`, files are organized into the `--output-dir` as follows:

```
<output_dir>/
├── <Author Name 1>/
│   ├── <Year> <Title>.<original_ext>
│   └── <Year> <Title>.txt (extracted text)
├── <Author Name 2>/
│   ├── <Year> <Another Title>.<original_ext>
│   └── <Year> <Another Title>.txt
└── ...
```

An `unparseables.lst` file may also be created in the output directory, listing files for which metadata parsing failed.

## Troubleshooting

* **Permissions:** Ensure BiblioForge has read/write permissions for input/output directories.
* **Dependencies:** Double-check that all Python and system dependencies are correctly installed and accessible in your system's PATH.
* **LLM Issues:**
  * For Ollama, ensure the server is running and the model is pulled.
  * For cloud providers, verify your API key and account status.
  * Try a different model or provider if one is consistently failing.
* **Debug Logs:** Use `-vv` or `--debug` for detailed logs to pinpoint issues:
  ```bash
  python BiblioForge.py --debug --sort your_file.pdf > debug_output.log 2>&1
  ```

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request on the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

BiblioForge relies on numerous excellent open-source libraries and tools. Credit and thanks to all their developers. Specific libraries are listed in the Python import sections.