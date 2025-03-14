# BiblioForge

BiblioForge is a multi-format document text extraction and automatic organization tool with smart metadata parsing. It extracts text from various document formats (PDF, EPUB, DJVU, MOBI, HTML, etc.) and intelligently organizes them by author and title using AI-powered content analysis.

## Features

- Extract text from multiple document formats (PDF, EPUB, DJVU, MOBI, TXT, HTML)
- Intelligently organize files based on content analysis
- Multiple extraction methods for each format with smart fallbacks
- OCR capabilities for scanned documents
- Table extraction for PDF files
- Language detection
- Sort and rename files based on extracted metadata
- Cross-platform support (Windows, macOS, Linux)

## Installation

### 1. Python Requirements

Python 3.7+ is required. Install the required Python packages:

```bash
# Core dependencies
pip install pymupdf pdfplumber pypdf pdfminer.six pytesseract pdf2image tqdm openai

# OCR-related packages
pip install easyocr paddleocr python-doctr ocrmypdf kraken

# Additional format support
pip install ebooklib beautifulsoup4 html2text mobi

# Optional dependencies for specific features
pip install camelot-py opencv-python chardet ftfy lxml
```

### 2. System Dependencies

#### macOS:

```bash
brew install tesseract poppler ghostscript djvulibre calibre
```

#### Linux:

```bash
sudo apt-get install tesseract-ocr poppler-utils ghostscript djvulibre-bin calibre
```

#### Windows:

Install the following:
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [Ghostscript](https://ghostscript.com/releases/gsdnld.html)
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/)
- [DjVuLibre](https://sourceforge.net/projects/djvu/files/DjVuLibre_Windows/)
- [Calibre](https://calibre-ebook.com/download)

### 3. LLM Support for Metadata Extraction

For the sorting feature, you'll need one of these options:

#### Option A: Local Ollama (Recommended)

1. Install [Ollama](https://ollama.ai/)
2. Pull a supported model:
   ```bash
   ollama pull llama3
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```

#### Option B: Cloud LLM Providers

For cloud-based LLM providers, set the corresponding API key:

```bash
# For OpenAI (GPT-4, GPT-3.5)
export OPENAI_API_KEY="your-key-here"

# For Groq
export GROQ_API_KEY="your-key-here"

# For Cohere
export COHERE_API_KEY="your-key-here"  

# For GLHF
export GLHF_API_KEY="your-key-here"
```

## Usage

### Basic Text Extraction

```bash
python BiblioForge.py input.pdf
python BiblioForge.py -o output_dir/ *.pdf
python BiblioForge.py --method=pymupdf input.pdf
```

### Process Multiple File Types

```bash
python BiblioForge.py "*.pdf *.epub *.djvu"
```

### Filter Specific File Types

```bash
python BiblioForge.py --file-types="pdf,epub" *.*
```

### Sorting and Organization

```bash
# Sort and generate rename commands
python BiblioForge.py --sort *.pdf

# Sort and automatically execute the rename commands
python BiblioForge.py --sort --execute-rename *.pdf

# Use a specific LLM provider
python BiblioForge.py --sort --llm-provider=groq --api-key=your-key *.pdf
```

### OCR Support

```bash
# Use OCR for scanned documents
python BiblioForge.py --ocr-method=tesseract input.pdf

# Try different OCR methods
python BiblioForge.py --ocr-method=paddleocr input.pdf
```

### Extract Tables from PDFs

```bash
python BiblioForge.py -t input.pdf
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `files` | Input files to process (supports wildcards and multiple file type patterns) |
| `-o, --output-dir` | Output directory for extracted text files (default: current directory) |
| `-m, --method` | Preferred extraction method (see below for options) |
| `-r, --recursive` | Process files recursively through subdirectories |
| `-p, --password` | Password for encrypted documents |
| `--ocr-method` | Preferred OCR method: auto, tesseract, paddleocr, doctr, easyocr, kraken, kraken_cli |
| `-t, --tables` | Extract tables (PDF only) |
| `-j, --json` | Save results to JSON file |
| `-w, --workers` | Maximum number of worker threads |
| `-d, --debug` | Enable debug logging |
| `--noskip` | Process files even if output text file already exists |
| `--sort` | Sort and rename files based on content analysis |
| `--execute-rename` | Automatically execute the generated rename commands |
| `--rename-script` | Path to write the rename commands (default: rename_commands.sh) |
| `--llm-provider` | LLM provider to use: ollama, groq, cohere, openai, glhf, huggingface |
| `--llm-model` | Model name to use with the LLM provider |
| `--api-key` | API key for cloud LLM providers |
| `--temperature` | Temperature setting for LLM generation (0.0-1.0) |
| `--max-tokens` | Maximum tokens in LLM response |
| `--file-types` | Only process specified file types (comma-separated, e.g., 'pdf,epub,djvu') |

## Extraction Methods

### PDF
- `pymupdf` - Fast native PDF parsing
- `pdfplumber` - Good balance of speed and accuracy
- `pypdf` - Simple but reliable
- `pdfminer` - Good layout preservation

### EPUB
- `ebooklib` - Native EPUB parsing
- `bs4` - BeautifulSoup-based extraction
- `zipfile` - Basic ZIP-based extraction

### DJVU
- `djvulibre` - Native DJVU parsing
- `pdf_conversion` - Convert to PDF then extract
- `ocr` - Optical Character Recognition

### MOBI
- `mobi` - Native MOBI parsing
- `kindleunpack` - KindleUnpack-based extraction
- `calibre` - Calibre-based conversion
- `zipfile` - Basic archive extraction

### HTML
- `bs4` - BeautifulSoup-based extraction
- `html2text` - HTML to markdown conversion
- `lxml` - LXML-based extraction
- `regex` - Basic regex-based extraction

### TXT
- `direct` - Direct file reading
- `charset_detection` - With encoding detection
- `encoding_detection` - With encoding fixing

## Output Structure

When using the `--sort` option, BiblioForge creates an organized structure:

```
output_dir/
  ├── Author Name/
  │    ├── Year Title.pdf
  │    └── Year Title.txt
  ├── Another Author/
  │    ├── Year Another Title.pdf
  │    └── Year Another Title.txt
  └── ...
```

## License

MIT

## Acknowledgments

BiblioForge uses several open-source libraries and tools. See the import section of the code for a full list.
