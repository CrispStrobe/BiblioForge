# BiblioForge: Document Text Extraction & Organization Tool

BiblioForge is a versatile, cross-platform command-line tool designed to extract text from a wide array of document formats and intelligently organize them. It leverages multiple extraction engines, OCR capabilities, and AI-powered metadata analysis to sort and rename your documents into a structured library.

**Status:** This project is under active development. While functional, expect ongoing improvements and potential changes.

## Core Features

* **Multi-Format Text Extraction:** Supports PDF, EPUB, DJVU, MOBI, PowerPoint (PPTX), various text formats (TXT, MD), HTML, and common office formats (DOCX, DOC, RTF, ODT, FB2, PDB, LIT, LRF, CBZ, CBR, CHM, SNB, TCR via Calibre).
* **Smart Fallbacks:** Employs multiple extraction methods for each format, ensuring the best possible text recovery.
* **OCR for Scanned Documents:** Integrated OCR engines (Tesseract, PaddleOCR, EasyOCR, DocTR, Kraken) to process image-based PDFs and other scanned documents.
* **Table Extraction:** Capable of extracting tabular data from PDF files using Camelot.
* **AI-Powered Sorting & Renaming:**
    * Utilizes Large Language Models (LLMs) to analyze content and extract key metadata (author, title, year, language).
    * Generates shell scripts (`rename_commands.sh` and `.bat` for Windows) to organize files into an `Author/Year Title.ext` structure.
    * Option to automatically execute the rename script.
* **Flexible LLM Integration:** Supports local LLMs via Ollama (recommended for privacy), LlamaCPP (direct GGUF loading), local OpenAI-compatible servers, and various cloud-based providers.
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
# Core extraction dependencies
pip install pymupdf pdfplumber pypdf pdfminer.six pytesseract pdf2image tqdm \
            ebooklib beautifulsoup4 html2text mobi chardet ftfy lxml requests \
            python-pptx

# LLM providers
pip install ollama openai httpx groq cohere huggingface_hub python-dotenv

# Table extraction (optional but recommended)
pip install "camelot-py[cv]"

# OCR engines (choose based on your needs)
pip install easyocr paddleocr python-doctr ocrmypdf

# Advanced OCR (can be complex to install)
pip install kraken

# Local LLM support (optional)
pip install llama-cpp-python  # Note: May require specific build configurations for GPU support
```

**Important Notes:**
- Some packages like `paddleocr`, `easyocr`, and `llama-cpp-python` can be complex to install and may require system-level dependencies (CUDA, specific build tools, etc.).
- If you encounter installation issues, consider installing packages individually and consulting their specific documentation.
- For `llama-cpp-python` with GPU support, you may need to install with specific CMAKE arguments or use pre-built wheels.

### 3. System Dependencies

Certain functionalities, especially OCR and some format conversions, rely on external system tools.

**Essential Recommendation:**
* **[ebook-converter](https://github.com/gryf/ebook-converter):** For robust conversion of many formats (DOCX, RTF, etc.) to text. Install the ebook-converter application (based on Calibre codebase). BiblioForge will use the `ebook-converter` command-line tool if it's in your system's PATH.

#### macOS:
```bash
brew install tesseract poppler ghostscript djvulibre calibre
```

Ensure tesseract data files for your desired languages are installed:
```bash
brew install tesseract-lang
```

#### Linux (Debian/Ubuntu based):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-all poppler-utils ghostscript \
                        djvulibre-bin calibre libgl1-mesa-glx libglib2.0-0
```

#### Windows:

Manual installation of the following is typically required. Ensure they are added to your system's PATH.

* **Tesseract OCR:** Download installer from [UB-Mannheim Tesseract builds](https://github.com/tesseract-ocr/tesseract). Install language data during setup.
* **Poppler for Windows:** Download binaries from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases). Add the `bin/` directory to PATH.
* **Ghostscript:** Download installer from [Ghostscript releases](https://www.ghostscript.com/download/gsdnld.html). Add its `bin/` and `lib/` directories to PATH.
* **DjVuLibre for Windows:** Available from DjView4 distribution which bundles command-line tools.
* **Calibre:** Download from [calibre-ebook.com](https://calibre-ebook.com/download_windows). Ensure `ebook-converter.exe` is in PATH.

### 4. LLM Setup (for --sort feature)

Choose one of the following options:

#### Option A: Local LLM with Ollama (Recommended)

1. **Install Ollama:** Follow instructions at [ollama.ai](https://ollama.ai).
2. **Pull a Model:** A smaller, faster model is often sufficient for metadata tasks.
   ```bash
   ollama pull llama3.2:3b  # Fast and efficient for metadata extraction
   # or other options:
   ollama pull qwen2:1.5b   # Very fast
   ollama pull phi3:mini    # Microsoft's efficient model
   ```
3. **Ensure Ollama Server is Running:** Usually runs as a background service after installation. If not, start manually with `ollama serve`.

#### Option B: LlamaCPP (Direct GGUF Loading)

Download and use GGUF models directly from HuggingFace:

```bash
python BiblioForge.py --sort --llm-provider=llama_cpp \
    --llamacpp-repo-id="microsoft/Phi-3-mini-4k-instruct-gguf" \
    --llamacpp-gguf-filename="Phi-3-mini-4k-instruct-q4.gguf" \
    documents/
```

#### Option C: Local OpenAI-Compatible Server

For LM Studio, text-generation-webui, or similar local servers:

```bash
python BiblioForge.py --sort --llm-provider=local_openai \
    --local-openai-base-url="http://localhost:1234/v1" \
    --llm-model="local-model" \
    documents/
```

#### Option D: Cloud LLM Providers

BiblioForge supports `.env` file for API key management. Create a `.env` file in your working directory:
```bash
# === EU-COMPLIANT PROVIDERS (GDPR) ===
SCALEWAY_API_KEY=scw-xxxxx
MISTRAL_API_KEY=xxxxx
NEBIUS_API_KEY=xxxxx

# === US PROVIDERS ===
OPENROUTER_API_KEY=sk-or-v1-xxxxx
POE_API_KEY=xxxxx
GROQ_API_KEY=gsk_xxxxx
OPENAI_API_KEY=sk-xxxxx
COHERE_API_KEY=xxxxx
HF_API_KEY=hf_xxxxx
GLHF_API_KEY=xxxxx

# === LOCAL SERVERS ===
OLLAMA_HOST=http://localhost:11434
LOCAL_OPENAI_BASE_URL=http://localhost:1234/v1
```

For this to work, you must have install **python-dotenv**:
```bash
pip install python-dotenv
```

Or you can set the appropriate environment variable for your chosen provider, e.g.:

```bash
# For OpenAI (e.g., GPT-4, GPT-3.5-turbo)
export OPENAI_API_KEY="your-openai-api-key"

# For Groq (fast LLM inference)
export GROQ_API_KEY="your-groq-api-key"

# For Cohere
export COHERE_API_KEY="your-cohere-api-key"

# For GLHF.chat (HuggingFace models via OpenAI-compatible API)
export GLHF_API_KEY="your-glhf-api-key"

# For HuggingFace Inference API
export HF_API_KEY="your-huggingface-api-key"

# For Poe.com API
export POE_API_KEY="your-poe-api-key"
```

On Windows, use `set` or `setx` in Command Prompt, or `$Env:VAR_NAME = "value"` in PowerShell.

## CrispEmbed Acceleration (Optional)

BiblioForge can optionally use the sibling [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)
engine — a self-contained C++/ggml document-understanding library (no PyTorch/
transformers) — for pre-OCR image cleanup, super-resolution, a torch-free OCR
backend, and fully-local metadata extraction. **Everything is optional**: if
CrispEmbed isn't present, all of these features are silently no-ops and
BiblioForge behaves exactly as before.

### Setup

1. Clone and build CrispEmbed as a sibling directory (`../CrispEmbed`):
   ```bash
   git clone https://github.com/CrispStrobe/CrispEmbed
   cd CrispEmbed
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCRISPEMBED_BUILD_SHARED=ON \
         -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON   # macOS; omit Metal flags elsewhere
   cmake --build build -j8
   ```
   BiblioForge auto-discovers `../CrispEmbed/python` + `build/libcrispembed.dylib`.
   Override the location with `CRISPEMBED_PYTHON_DIR` if your layout differs.

2. Models are GGUF files auto-downloaded on first use into `~/.cache/crispembed`
   (override with `CRISPEMBED_CACHE_DIR`; a symlink to an external volume with a
   local fallback is supported).

### Features & flags

| Flag | What it does |
|------|--------------|
| `--scan-cleanup {off,auto,on}` | Deskew / crop / whiten each page image before OCR (default `auto`; classical, no model). `--scan-cleanup-binarize {off,otsu,sauvola}` adds binarization. |
| `--sr {off,auto,on}` | Super-resolve low-resolution pages before OCR. `--sr-engine` (pan/swinir/…), `--sr-min-width`. Default `off`. |
| `--ocr-method crispembed` | **Experimental.** Single-pass CrispEmbed VLM OCR. `--crispembed-ocr-model` (default `internvl2-1b`), `--crispembed-ocr-dpi`, `--crispembed-ocr-cpu` (recommended). Current models vary in quality (see CrispEmbed issue #25); BiblioForge's built-in OCR (tesseract/…) is the reliable default. |
| `--metadata-backend {llm,crispembed-ner,hybrid}` | Metadata source for `--sort`. `crispembed-ner` extracts author/title/year/language **locally** (GLiNER + language ID — no LLM, no network); `hybrid` tries NER first then falls back to the LLM. |

```bash
# Local, no-LLM sorting: extract metadata with on-device NER and build the rename script
python BiblioForge.py *.pdf --sort --metadata-backend crispembed-ner

# Improve OCR on skewed/noisy scans
python BiblioForge.py scan.pdf --force-ocr --scan-cleanup on
```

> **Note (macOS/Metal):** ggml's Metal backend can abort during process-exit
> teardown when loaded alongside PyTorch's MPS. BiblioForge works around this by
> calling `os._exit()` after all work completes when a CrispEmbed Metal engine
> was used; output is fully written before exit. Use `--crispembed-ocr-cpu` if a
> specific model hits an unsupported Metal op.

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

# Process PowerPoint files
python BiblioForge.py --file-types="pptx" presentations/
```

### OCR Processing:
```bash
# Force OCR using Tesseract on a scanned PDF
python BiblioForge.py --force-ocr --ocr-method=tesseract scanned_document.pdf

# Use PaddleOCR for better multilingual support
python BiblioForge.py --force-ocr --ocr-method=paddleocr multilingual_doc.pdf
```

### Sorting and Renaming (Requires LLM Setup):
```bash
# Analyze PDFs with Ollama, generate rename script to sort them
python BiblioForge.py --sort --noskip --output-dir ./organized_docs/ *.pdf

# Use a specific cloud provider and execute renames immediately
python BiblioForge.py --sort --llm-provider=groq --execute-rename \
    --output-dir ./groq_sorted/ documents/*.epub

# Use local LlamaCPP with custom model
python BiblioForge.py --sort --llm-provider=llama_cpp \
    --llamacpp-repo-id="TheBloke/phi-2-GGUF" \
    --llamacpp-gguf-filename="phi-2.Q4_K_M.gguf" \
    --temperature=0.2 documents/

# Use local OpenAI-compatible server (LM Studio, etc.)
python BiblioForge.py --sort --llm-provider=local_openai \
    --local-openai-base-url="http://localhost:1234/v1" \
    academic_papers/
```

**Important Notes:**
- `--noskip` is recommended with `--sort` to ensure all files are processed for metadata, even if .txt files exist.
- `--output-dir` with `--sort` specifies where the new `Author/Year Title.ext` structure will be created.

### Table Extraction (PDFs):
```bash
# Extract tables from financial reports
python BiblioForge.py --tables financial_report.pdf

# Save results including tables to JSON
python BiblioForge.py --tables --json=results.json quarterly_reports/*.pdf
```

### Debugging:
```bash
# See detailed logs for troubleshooting
python BiblioForge.py --debug document.pdf

# Debug with specific extraction method
python BiblioForge.py --debug --method=pdfminer --ocr-method=tesseract problematic.pdf
```

## Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `files` | | Input files or patterns to process (e.g., `*.pdf`, `"docs/*.epub"`). | (None - scans current directory) |
| `--output-dir` | `-o` | Base directory for extracted .txt files and sorted file structure. | `.` (current directory) |
| `--method` | `-m` | Preferred primary extraction method (varies by file type). | (auto-detection) |
| `--ocr-method` | | Preferred OCR method: `auto`, `tesseract`, `paddleocr`, `doctr`, `easyocr`, `kraken`, `kraken_cli`. | `auto` |
| `--force-ocr` | | Force OCR for all pages, even if a text layer is detected. | (False) |
| `--recursive` | `-r` | Process files recursively in subdirectories. | (False) |
| `--password` | `-p` | Password for encrypted documents. | (None) |
| `--tables` | `-t` | Attempt to extract tables (primarily for PDF files using Camelot). | (False) |
| `--json` | `-j` | Save detailed processing results to a JSON file. Specify the file path. | (None) |
| `--workers` | `-w` | Maximum number of worker threads for parallel processing. | (auto-detected) |
| `--noskip` | | Re-process files even if .txt output already exists. Essential for `--sort`. | (False) |
| `--file-types` | | Comma-separated list of file extensions (e.g., `pdf,epub,pptx`). | (All supported) |

### Sorting & LLM Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--sort` | Enable LLM-based metadata extraction and sorting. | (False) |
| `--rename-script` | Filename for the generated rename script (relative to `--output-dir`). | `rename_commands.sh` |
| `--execute-rename` | Automatically execute the generated rename script after processing. | (False) |
| `--llm-provider` | LLM provider: `ollama`, `local_openai`, `llama_cpp`, `scaleway` 🇪🇺, `mistral` 🇪🇺, `nebius` 🇪🇺, `openrouter`, `poe`, `openai`, `groq`, `cohere`, `glhf`, `huggingface`. | `ollama` |
| `--llm-model` | Specific model name for the chosen provider. | (Provider default) |
| `--api-key` | API key for cloud-based providers (if not set as environment variable). | (None) |
| `--temperature` | LLM temperature for metadata extraction (0.0-2.0). | `0.7` |
| `--max-tokens` | LLM max tokens for metadata extraction. | `300` |

### LLM Provider-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ollama-host` | Host for Ollama server (e.g., `http://localhost:11434`). | (Library default) |
| `--local-openai-base-url` | Base URL for local OpenAI-compatible servers. | `http://localhost:1234/v1/` |
| `--llamacpp-repo-id` | HuggingFace Repo ID for LlamaCPP GGUF model. | `TheBloke/phi-2-GGUF` |
| `--llamacpp-gguf-filename` | Specific GGUF filename from the HF repo. | `phi-2.Q4_K_M.gguf` |
| `--llamacpp-n-ctx` | Context size for LlamaCPP. | `2048` |
| `--llamacpp-n-gpu-layers` | Number of layers to offload to GPU (-1 for all, 0 for CPU). | `0` |
| `--llamacpp-chat-format` | Chat format for LlamaCPP (e.g., `llama-2`, `chatml`). | `llama-2` |

### Debug & Logging Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--verbose` | `-v` | Increase verbosity: `-v` for INFO, `-vv` for DEBUG. | (WARNING level) |
| `--debug` | `-d` | Enable debug logging (shortcut for `-vv`). | (False) |

## Available Extraction Methods by Format

BiblioForge tries methods in a preferred order, falling back if one fails.

### PDF
- **Text Layer:** `pymupdf`, `pdfplumber`, `calibre`, `pypdf`, `pdfminer`
- **OCR:** `tesseract`, `easyocr`, `paddleocr`, `doctr`, `kraken`, `kraken_cli`
- **Tables:** `camelot` (with lattice and stream methods)

### EPUB
- **Methods:** `ebooklib`, `bs4` (with html2text), `epub2txt`, `calibre`, `zipfile`

### DJVU
- **Methods:** `djvulibre` (Python bindings or djvutxt CLI), `pdf_conversion` (via ddjvu then PDF extraction), `ocr` (via ddjvu to images then Tesseract)

### MOBI/AZW
- **Methods:** `mobi`, `kindleunpack`, `calibre`, `zipfile`

### PowerPoint (PPTX/PPT)
- **Methods:** `pptx` (python-pptx library), `calibre`

### HTML/XHTML
- **Methods:** `bs4`, `html2text`, `lxml`, `regex`

### Text & Office Formats
- **Plain Text (TXT/MD):** `direct`, `charset_detection`, `encoding_detection`, `calibre`
- **Office Formats (DOCX, DOC, RTF, ODT, FB2, PDB, LIT, LRF, CBZ, CBR, CHM, SNB, TCR):** `calibre` (primary), `charset_detection`, `encoding_detection`, `direct`

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
├── rename_commands.sh (or .bat on Windows)
└── unparseables.lst (files with failed metadata extraction)
```

## Supported File Types

BiblioForge supports the following file extensions:

**Documents:** `.pdf`, `.epub`, `.djvu`, `.djv`, `.mobi`, `.azw`, `.azw3`, `.azw4`  
**Text:** `.txt`, `.text`, `.md`  
**Web:** `.html`, `.htm`, `.xhtml`  
**Office:** `.docx`, `.doc`, `.rtf`, `.odt`  
**Presentations:** `.pptx`, `.ppt`  
**Other Ebook:** `.fb2`, `.pdb`, `.lit`, `.lrf`, `.cbz`, `.cbr`, `.chm`, `.snb`, `.tcr`

## Troubleshooting

### Common Issues

**Dependencies:** Double-check that all Python and system dependencies are correctly installed and accessible in your system's PATH.

**LLM Issues:**
- For Ollama, ensure the server is running (`ollama serve`) and the model is pulled.
- For cloud providers, verify your API key and account status/credits.
- Try a different model or provider if one is consistently failing.
- For LlamaCPP, ensure you have sufficient RAM/VRAM for the model.

**OCR Problems:**
- Verify Tesseract is installed and language data is available.
- For GPU-based OCR (PaddleOCR, EasyOCR), ensure CUDA is properly configured.
- Large documents may require significant memory - consider processing smaller batches.

**File Access:**
- Ensure BiblioForge has read/write permissions for input/output directories.
- Check that encrypted PDFs have the correct password provided via `-p`.

### Debug Logging

Use detailed logging for troubleshooting:
```bash
python BiblioForge.py --debug --sort your_file.pdf > debug_output.log 2>&1
```

### Performance Optimization

- Use `--workers` to control parallelism (fewer workers for memory-constrained systems).
- For large document sets, consider processing in smaller batches.
- Local LLMs (Ollama, LlamaCPP) are generally faster than cloud APIs for batch processing.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request on the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

BiblioForge relies on numerous excellent open-source libraries and tools. Credit and thanks to all their developers, including but not limited to:

- PyMuPDF, pdfplumber, pypdf, pdfminer.six for PDF processing
- ebooklib for EPUB handling
- Tesseract, PaddleOCR, EasyOCR, DocTR, Kraken for OCR capabilities
- Camelot for table extraction
- python-pptx for PowerPoint processing
- BeautifulSoup, lxml for HTML/XML parsing
- Ollama, OpenAI, and other LLM providers for metadata extraction
- Calibre project for universal document conversion