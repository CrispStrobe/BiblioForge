#!/usr/bin/env python3

import fitz  # PyMuPDF
import subprocess
import sys
import os
import argparse
import re
import shlex
import json
import tempfile
from pathlib import Path
from ebooklib import epub
from bs4 import BeautifulSoup

# Additional imports for extractors
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

# --- CALIBRE-INSPIRED UTILITIES ---

def clean_xml_chars(text):
    """Remove invalid XML characters, inspired by Calibre"""
    # Remove control chars except tab, newline, carriage return
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return cleaned

def xml_replace_entities(text):
    """Replace common entities that cause issues"""
    replacements = {
        '&nbsp;': ' ',
        '&#160;': ' ',
        '&#8201;': ' ',  # thin space
        '&#8202;': ' ',  # hair space
        '&#8203;': '',   # zero-width space
        '\u00a0': ' ',   # non-breaking space
        '\u2029': ' ',   # paragraph separator
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def detect_drm(pdf_path):
    """Simple DRM detection inspired by Calibre"""
    try:
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted or doc.needs_pass:
                return True
            # Try to extract first page as additional check
            if len(doc) > 0:
                try:
                    _ = doc[0].get_text()
                except:
                    return True
    except:
        pass
    return False

# --- ENHANCED HTML POST-PROCESSING ---

def clean_html_for_ebook(html_path, preserve_structure=False):
    """
    Enhanced HTML cleaning with more Calibre-inspired techniques
    """
    print(f"    [Info] Cleaning CSS and fixing structure in '{html_path}'...")
    try:
        with open(html_path, "r", encoding="utf-8", errors='replace') as f:
            raw_html = f.read()

        # Apply Calibre-style preprocessing
        raw_html = xml_replace_entities(raw_html)
        raw_html = clean_xml_chars(raw_html)
        
        # Replace self-closing br tags (Calibre does this)
        raw_html = raw_html.replace('<br/>', '<br>')
        raw_html = raw_html.replace('<BR/>', '<BR>')
        
        soup = BeautifulSoup(raw_html, "html.parser")
        
        if not preserve_structure:
            # 1. Remove all styles and stylesheets
            for element in soup.find_all(["style", "link"]):
                if element.name == 'link' and 'stylesheet' not in element.get('rel', []):
                    continue
                element.decompose()
            
            # 2. Remove all inline styles
            for element in soup.find_all(style=True):
                del element['style']
            
            # 3. Remove all class attributes (they're meaningless without CSS)
            for element in soup.find_all(class_=True):
                del element['class']

        # 4. Fix numeric anchors (enhanced version)
        # First pass: collect all numeric IDs/names
        numeric_ids = set()
        for element in soup.find_all(id=re.compile(r'^\d+$')):
            numeric_ids.add(element['id'])
            element['id'] = f"p{element['id']}"
        
        for element in soup.find_all(attrs={'name': re.compile(r'^\d+$')}):
            numeric_ids.add(element['name'])
            element['name'] = f"p{element['name']}"
        
        # Second pass: fix all links
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Handle both #123 and index.html#123 formats
            match = re.match(r'^([^#]*#)(\d+)$', href)
            if match:
                link['href'] = f"{match.group(1)}p{match.group(2)}"

        # 5. Add semantic structure if missing
        if not soup.find('head'):
            head = soup.new_tag('head')
            title = soup.new_tag('title')
            title.string = "Converted Document"
            head.append(title)
            if soup.html:
                soup.html.insert(0, head)
        
        # 6. Fix common PDF conversion artifacts
        cleaned_html = str(soup)
        
        # Remove excessive whitespace
        cleaned_html = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_html)
        
        # Fix hyphenated words at line breaks (common in PDFs)
        cleaned_html = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_html)
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(cleaned_html)
        
        return True

    except Exception as e:
        print(f"    [Warning] Failed to clean HTML: {e}")
        return False

# --- PDF TEXT EXTRACTORS ---

def extract_text_with_pymupdf(pdf_path, html_path, include_images=False, no_html_cleanup=False):
    """
    Enhanced PyMuPDF extraction that:
    1. Builds paragraphs line-by-line.
    2. Uses a hybrid heuristic for new paragraphs:
        a) Checks for significant (15pt+) change in base indentation.
        b) Checks for [Punctuation-ended line] + [Capitalized new line].
    3. Detects and formats headers/page numbers (e.g., "[XI]") as requested.
    4. Preserves <i> and <b> formatting.
    5. Includes robust page-stitching for paragraphs split across pages.
    """
    print(f"[*] Attempting (pymupdf) extraction with hybrid indent/punct logic...")
    
    try:
        doc = fitz.open(pdf_path)
        full_html = "<html><head><title>Extracted Text</title></head><body>\n"
        
        # Buffer for page-stitching (holds HTML spans)
        page_stitch_buffer_spans = []
        
        # Buffer for headers/page numbers
        pending_header_spans = []
        
        # State for paragraph-breaking logic
        current_para_spans = []
        current_para_base_indent = 0.0
        last_line_ended_with_punc = False

        # Heuristic for ending punctuation.
        end_punctuation_chars = ('.', '!', '?', '”', '’', ':', ';', '—', '}', ']', ')')
        end_punctuation_tags = ('.', '!', '?', '”', '’', ':', ';', '—', '}', ']', ')', '</i>', '</b>')

        def flush_paragraph(spans_list):
            """Helper to join, de-hyphenate, and return a <p> string."""
            if not spans_list:
                return ""
            
            para_html = " ".join(spans_list)

            # De-hyphenate words split by a normal hyphen OR a soft hyphen
            # (e.g., "philos- ophy" or "philos­ ophy")
            para_html = re.sub(r'(\w+)[\xad-]\s+', r'\1', para_html)
            para_html = re.sub(r'\s+', ' ', para_html).strip()
            
            if para_html:
                return f"<p>{para_html}</p>\n"
            return ""

        # --- Page Loop ---
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            if not blocks:
                continue
            
            content_blocks = [b for b in blocks if b['type'] == 0 and b['lines']]
            if not content_blocks:
                continue
            
            # If we have a buffer, add it to the spans list to start
            if page_stitch_buffer_spans:
                current_para_spans.extend(page_stitch_buffer_spans)
                page_stitch_buffer_spans = []
                # Get the indent from the *stitching buffer* to prevent a false break
                # This assumes the buffer has at least one span with 'bbox'
                # (We will adapt the buffer to store raw spans)
                if current_para_spans:
                    # Can't reliably get indent from HTML, so just prevent break
                    last_line_ended_with_punc = False
            
            is_first_line_of_page = True

            # --- Block Loop ---
            for i, b in enumerate(content_blocks):
                
                # --- Line Loop (This is the critical part) ---
                for l in b['lines']:
                    if not l['spans']: continue
                    
                    line_spans_raw = [] # To hold (span_text, span_html) tuples
                    line_text_plain = ""
                    current_line_indent = l['spans'][0]['bbox'][0]
                    
                    # --- Span Loop (process the current line) ---
                    for s in l['spans']:
                        span_text = s['text'].replace('\n', ' ').strip()
                        if not span_text: continue
                        
                        span_html = span_text
                        if s["flags"] & 16: span_html = f"<b>{span_html}</b>"
                        if s["flags"] & 2:  span_html = f"<i>{span_html}</i>"
                        
                        line_spans_raw.append((span_text, span_html))
                        line_text_plain += span_text + " "
                    
                    line_text_plain = line_text_plain.strip()
                    if not line_text_plain:
                        continue # Skip empty lines

                    # *** 1. HEADER/PAGE NUMBER HEURISTIC (User Request 3) ***
                    # Check if line matches a header/page number pattern
                    is_header_or_page_num_match = bool(
                        re.fullmatch(r'^(?:[IVXLCDM]+|[0-9]+)$', line_text_plain) or \
                        re.fullmatch(r'^(?:[IVXLCDM]+|[0-9]+)?\s*Preface\s*(?:[IVXLCDM]+|[0-9]+)?$', line_text_plain, re.IGNORECASE)
                    )
                    # Headers/page nums are short; ToC lines are long
                    is_header_or_page_num = is_header_or_page_num_match and len(line_text_plain) < 25
                    
                    if is_header_or_page_num:
                        # It's a header/page num. Buffer it and skip to next line.
                        pending_header_spans.append(f"[{line_text_plain}]")
                        continue # Don't process this line as a paragraph

                    # *** 2. PARAGRAPH BREAK HEURISTIC (User Request 1 & 2) ***
                    is_new_para = False
                    
                    if not current_para_spans:
                        is_new_para = True # It's the first line of a new paragraph
                    elif is_first_line_of_page:
                         is_new_para = False # Don't break if we just stitched
                    else:
                        # Heuristic A: Punctuation + Capitalization
                        starts_with_capital = bool(re.match(r'^[A-Z]', line_text_plain))
                        trigger_A = last_line_ended_with_punc and starts_with_capital
                        
                        # Heuristic B: Indentation Change
                        # A significant change (e.g., a "tab") from the base
                        indent_changed = abs(current_line_indent - current_para_base_indent) > 15.0 
                        trigger_B = indent_changed
                        
                        if trigger_A or trigger_B:
                            is_new_para = True
                    
                    if is_new_para:
                        # This new line starts a new paragraph.
                        # 1. Flush the *old* paragraph.
                        full_html += flush_paragraph(current_para_spans)
                        
                        # 2. Start new span list and add pending headers
                        current_para_spans = []
                        if pending_header_spans:
                            current_para_spans.extend(pending_header_spans)
                            pending_header_spans = []
                        
                        # 3. Set new base indent
                        current_para_base_indent = current_line_indent
                    
                    # Add this line's HTML spans to the current paragraph
                    current_para_spans.extend([s_html for s_text, s_html in line_spans_raw])
                    
                    # Update state for *next* line's check
                    if line_text_plain:
                        last_line_ended_with_punc = line_text_plain.endswith(end_punctuation_chars)
                    
                    is_first_line_of_page = False
            
            # --- End of Page ---

            # Flush any pending headers at the end of a page
            # if they haven't been attached to a paragraph.
            # This stops [XI] from carrying over to the next page.
            if pending_header_spans and not current_para_spans:
                full_html += flush_paragraph(pending_header_spans)
                pending_header_spans = []

            # Now we apply page-stitching logic to whatever is
            # left in `current_para_spans`.
            
            is_last_page = (page_num == len(doc) - 1)
            
            if current_para_spans:
                # Build the HTML, but don't add <p> tags yet
                para_html = " ".join(current_para_spans)
                para_html = re.sub(r'(\w+)-\s+', r'\1', para_html)
                para_html = re.sub(r'\s+', ' ', para_html).strip()
                
                if not para_html:
                    current_para_spans = [] # Nothing to buffer
                    continue

                # Check for hyphenation
                hyphen_match_html = re.search(r'(\w+)-\s*(<\/[bi]>)?\s*$', para_html)
                
                # Check for punctuation
                ends_properly = any(para_html.endswith(punc) for punc in end_punctuation_tags)
                
                if not is_last_page:
                    if hyphen_match_html:
                        # Buffer the de-hyphenated part
                        start_pos = hyphen_match_html.start(0)
                        root_word = hyphen_match_html.group(1)
                        tag = hyphen_match_html.group(2) if hyphen_match_html.group(2) else ""
                        page_stitch_buffer_spans = [para_html[:start_pos] + root_word + tag]
                        
                    elif not ends_properly:
                        # Buffer the whole thing
                        page_stitch_buffer_spans = current_para_spans
                        
                    else:
                        # It ends properly. Flush it.
                        full_html += f"<p>{para_html}</p>\n"
                        page_stitch_buffer_spans = [] # Clear buffer
                else:
                    # This is the last page. Flush it.
                    full_html += f"<p>{para_html}</p>\n"
                    page_stitch_buffer_spans = [] # Clear buffer
                
                # Clear the current spans list, it's either buffered or flushed
                current_para_spans = []

            if not is_last_page:
                full_html += '<div style="page-break-after:always"></div>\n'
                
        # --- End of Document ---
        
        # Flush any remaining buffer
        if page_stitch_buffer_spans:
             full_html += flush_paragraph(page_stitch_buffer_spans)
        
        # Flush any lingering headers
        if pending_header_spans:
            full_html += flush_paragraph(pending_header_spans)
            
        full_html += "</body></html>"
        doc.close()

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        
        if not no_html_cleanup:
            clean_html_for_ebook(html_path)

        return html_path, "html"
        
    except Exception as e:
        print(f"[Failed] PyMuPDF (hybrid-logic) extraction failed: {e}")
        return None, None

def extract_text_with_pdftohtml(pdf_path, html_path, include_images=False, no_html_cleanup=False):
    """
    Fallback method: Uses 'pdftohtml' and then
    aggressively cleans the HTML to remove CSS positioning.
    """
    print(f"[*] Attempting (pdftohtml) extraction...")
    
    command = ["pdftohtml", "-s", "-noframes"]
    
    if include_images:
        print("    [Info] --include-imgs: Including images.")
    else:
        print("    [Info] Default: Ignoring images.")
        command.append("-i") # Add the "ignore images" flag
        
    # *** FIX: The base name is the output path *without* the .html extension ***
    html_base_name = os.path.splitext(html_path)[0]
    command.extend([pdf_path, html_base_name])
    
    try:
        print(f"    [Exec] {' '.join(shlex.quote(arg) for arg in command)}")
        
        # *** FIX: The expected output file *is* html_path. ***
        # `pdftohtml <base_name>` creates `<base_name>.html`
        # `html_path` is already `<base_name>.html`
        # We no longer look for the "-s.html" file.
        
        if os.path.exists(html_path):
            os.remove(html_path)

        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        if result.stdout:
            print("    --- pdftohtml stdout ---", result.stdout, "    --- end stdout ---", sep="\n")
        if result.stderr:
            print("    --- pdftohtml stderr ---", result.stderr, "    --- end stderr ---", sep="\n")

        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)

        # *** FIX: Check for the *correct* file ***
        if not os.path.exists(html_path):
            print(f"[Failed] 'pdftohtml' ran successfully (code 0) but did NOT create the expected file.")
            print(f"    Expected file: {html_path}")
            return None, None

        # *** FIX: No rename needed, the file is already correct ***
            
        if not no_html_cleanup:
            clean_html_for_ebook(html_path)

        print(f"[Success] Extracted and cleaned HTML to '{html_path}'")
        return html_path, "html"
        
    except FileNotFoundError:
        print(f"[Failed] 'pdftohtml' command not found. (Poppler is not in PATH).")
        return None, None
    except subprocess.CalledProcessError:
        print(f"[Failed] 'pdftohtml' failed with a non-zero return code.")
        return None, None
    except Exception as e:
        print(f"[Failed] An unexpected error occurred with pdftohtml: {e}")
        return None, None

def extract_text_with_pdfplumber(pdf_path, html_path, **kwargs):
    """Extract using pdfplumber - good for tables and structured content"""
    if not HAS_PDFPLUMBER:
        print("[Skip] pdfplumber not installed")
        return None, None
    
    print(f"[*] Attempting (pdfplumber) extraction...")
    try:
        full_html = "<html><head><title>Extracted Text</title></head><body>"
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        # Convert to basic HTML paragraphs
                        paragraphs = text.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                full_html += f"<p>{para.strip()}</p>\n"
                    
                    # Extract tables if any
                    tables = page.extract_tables()
                    for table in tables:
                        full_html += "<table border='1'>\n"
                        for row in table:
                            full_html += "<tr>"
                            for cell in row:
                                full_html += f"<td>{cell or ''}</td>"
                            full_html += "</tr>\n"
                        full_html += "</table>\n"
                    
                    full_html += "<hr>\n"
                except Exception as e:
                    print(f"    [Warning] Error on page {page_num + 1}: {e}")
        
        full_html += "</body></html>"
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        
        if not kwargs.get('no_html_cleanup', False):
            clean_html_for_ebook(html_path)
        
        print(f"[Success] Extracted with pdfplumber to '{html_path}'")
        return html_path, "html"
        
    except Exception as e:
        print(f"[Failed] pdfplumber extraction failed: {e}")
        return None, None

def extract_text_with_pypdf(pdf_path, html_path, **kwargs):
    """Extract using PyPDF2/pypdf - fast but basic"""
    if not HAS_PYPDF:
        print("[Skip] pypdf not installed")
        return None, None
    
    print(f"[*] Attempting (pypdf) extraction...")
    try:
        reader = PdfReader(pdf_path)
        full_html = "<html><head><title>Extracted Text</title></head><body>"
        
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    # Basic paragraph detection
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            full_html += f"<p>{para.strip()}</p>\n"
                    full_html += "<hr>\n"
            except Exception as e:
                print(f"    [Warning] Error on page {page_num + 1}: {e}")
        
        full_html += "</body></html>"
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        
        if not kwargs.get('no_html_cleanup', False):
            clean_html_for_ebook(html_path)
        
        print(f"[Success] Extracted with pypdf to '{html_path}'")
        return html_path, "html"
        
    except Exception as e:
        print(f"[Failed] pypdf extraction failed: {e}")
        return None, None

def extract_text_with_pdfminer(pdf_path, html_path, **kwargs):
    """Extract using pdfminer - good for complex layouts"""
    if not HAS_PDFMINER:
        print("[Skip] pdfminer not installed")
        return None, None
    
    print(f"[*] Attempting (pdfminer) extraction...")
    try:
        with open(pdf_path, 'rb') as pdf_file:
            with open(html_path, 'w', encoding='utf-8') as html_file:
                # pdfminer can output HTML directly
                laparams = LAParams()
                extract_text_to_fp(
                    pdf_file, 
                    html_file, 
                    output_type='html',
                    laparams=laparams,
                    codec='utf-8'
                )
        
        if not kwargs.get('no_html_cleanup', False):
            clean_html_for_ebook(html_path)
        
        print(f"[Success] Extracted with pdfminer to '{html_path}'")
        return html_path, "html"
        
    except Exception as e:
        print(f"[Failed] pdfminer extraction failed: {e}")
        return None, None

def extract_with_calibre_ebook_convert(pdf_path, html_path, **kwargs):
    """Use Calibre's ebook-convert directly for extraction"""
    print(f"[*] Attempting (calibre ebook-convert) extraction...")
    
    command = ["ebook-convert", pdf_path, html_path]
    
    try:
        print(f"    [Exec] {' '.join(shlex.quote(arg) for arg in command)}")
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0 and os.path.exists(html_path):
            if not kwargs.get('no_html_cleanup', False):
                clean_html_for_ebook(html_path)
            print(f"[Success] Extracted with calibre to '{html_path}'")
            return html_path, "html"
        else:
            print(f"[Failed] calibre ebook-convert failed")
            return None, None
            
    except FileNotFoundError:
        print(f"[Failed] calibre ebook-convert not found")
        return None, None
    except Exception as e:
        print(f"[Failed] calibre extraction error: {e}")
        return None, None

# --- OCR SUPPORT FOR SCANNED PDFS ---

def extract_with_ocr(pdf_path, html_path, **kwargs):
    """Use OCR for scanned PDFs - requires ocrmypdf or tesseract"""
    print(f"[*] Attempting (OCR) extraction...")
    
    # First try ocrmypdf to create a searchable PDF
    temp_pdf = pdf_path + "_ocr.pdf"
    command = ["ocrmypdf", "--force-ocr", pdf_path, temp_pdf]
    
    try:
        print(f"    [Exec] Running OCR...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Now extract text from OCR'd PDF using our best method
            extracted = extract_text_with_pymupdf(temp_pdf, html_path, **kwargs)
            os.remove(temp_pdf)
            return extracted
        else:
            print(f"[Failed] OCR failed")
            return None, None
            
    except FileNotFoundError:
        print(f"[Failed] ocrmypdf not found - install with: pip install ocrmypdf")
        return None, None
    except Exception as e:
        print(f"[Failed] OCR error: {e}")
        return None, None

# --- EBOOK GENERATORS ---

def generate_mobi_with_ebook_converter(input_path, output_path, **kwargs):
    """
    Tries to generate a MOBI using the user's 'ebook-converter' fork.
    """
    print(f"[*] Attempting (ebook-converter) MOBI generation...")
    
    command = [
        "ebook-converter", 
        input_path,
        output_path,
    ]
    try:
        print(f"    [Exec] {' '.join(shlex.quote(arg) for arg in command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        if result.stdout:
            print("    --- ebook-converter stdout ---", result.stdout, "    --- end stdout ---", sep="\n")
        if result.stderr:
            print("    --- ebook-converter stderr ---", result.stderr, "    --- end stderr ---", sep="\n")
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        
        if not os.path.exists(output_path):
            print(f"[Failed] 'ebook-converter' ran successfully (code 0) but did NOT create the output file.")
            print(f"    Expected file: {output_path}")
            return None
            
        print(f"[Success] Successfully created '{output_path}'")
        return output_path
        
    except FileNotFoundError:
        print(f"[Failed] 'ebook-converter' command not found.")
        print("    Please ensure the gryf/ebook-converter fork is installed and in your PATH.")
        return None
    except subprocess.CalledProcessError:
        print(f"[Failed] 'ebook-converter' failed with a non-zero return code.")
        return None
    except Exception as e:
        print(f"[Failed] An unexpected error occurred with ebook-converter: {e}")
        return None

def generate_epub_with_ebooklib(input_path, input_type, output_path, **kwargs):
    """
    Fallback method: generates an EPUB using pure Python.
    """
    print(f"[*] Attempting (ebooklib) EPUB generation...")
    try:
        book = epub.EpubBook()
        book.set_title(os.path.basename(output_path).replace(".epub", ""))
        book.set_language("en")

        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')
        if soup.body:
            chapter_content = str(soup.body)
        else:
            chapter_content = content 

        c1 = epub.EpubHtml(title='Content', file_name='chap_1.xhtml', lang='en')
        c1.content = chapter_content
        book.add_item(c1)

        book.toc = (c1,)
        book.spine = ['nav', c1]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(output_path, book, {})
        print(f"[Success] Successfully created '{output_path}'")
        return output_path

    except Exception as e:
        print(f"[Failed] ebooklib EPUB generation failed: {e}")
        return None

def generate_with_pandoc(input_path, output_path, output_format="epub", **kwargs):
    """Use pandoc for conversion - very versatile"""
    print(f"[*] Attempting (pandoc) {output_format.upper()} generation...")
    
    command = ["pandoc", "-f", "html", "-t", output_format, "-o", output_path, input_path]
    
    # Add metadata if available
    if kwargs.get('title'):
        command.extend(["--metadata", f"title={kwargs['title']}"])
    
    try:
        print(f"    [Exec] {' '.join(shlex.quote(arg) for arg in command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"[Success] Generated with pandoc: '{output_path}'")
            return output_path
        else:
            print(f"[Failed] pandoc conversion failed")
            return None
            
    except FileNotFoundError:
        print(f"[Failed] pandoc not found")
        return None
    except Exception as e:
        print(f"[Failed] pandoc error: {e}")
        return None

def generate_epub_with_pandoc(input_path, input_type, output_path, **kwargs):
    """Wrapper for pandoc EPUB generation"""
    return generate_with_pandoc(input_path, output_path, "epub", **kwargs)

def generate_with_calibre_original(input_path, output_path, **kwargs):
    """Use original Calibre ebook-convert"""
    print(f"[*] Attempting (calibre ebook-convert) generation...")
    
    command = ["ebook-convert", input_path, output_path]
    
    # Add useful Calibre options
    if output_path.endswith('.mobi'):
        command.extend(["--output-profile", "kindle"])
    
    try:
        print(f"    [Exec] {' '.join(shlex.quote(arg) for arg in command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"[Success] Generated with calibre: '{output_path}'")
            return output_path
        else:
            print(f"[Failed] calibre conversion failed")
            return None
            
    except Exception as e:
        print(f"[Failed] calibre error: {e}")
        return None

# --- ENHANCED REGISTRIES ---

EXTRACTORS = {
    "pymupdf": extract_text_with_pymupdf,
    "pdftohtml": extract_text_with_pdftohtml,
    "pdfplumber": extract_text_with_pdfplumber,
    "pypdf": extract_text_with_pypdf,
    "pdfminer": extract_text_with_pdfminer,
    "calibre": extract_with_calibre_ebook_convert,
    "ocr": extract_with_ocr,
}

# Order matters - best quality first
DEFAULT_EXTRACTOR_ORDER = [
    "pymupdf",      # Best for formatted text
    "pdfplumber",   # Good for tables
    "pdfminer",     # Good for complex layouts
    "pdftohtml",    # Reliable fallback
    "calibre",      # If calibre is installed
    "pypdf",        # Fast but basic
]

GENERATORS = {
    "ebook_converter": generate_mobi_with_ebook_converter,
    "ebooklib": generate_epub_with_ebooklib,
    "pandoc": generate_epub_with_pandoc,
    "calibre": generate_with_calibre_original,
}

# --- CONFIGURATION AND PROFILES ---

PROFILES = {
    "quality": {
        "extractors": ["pymupdf", "pdfminer", "calibre"],
        "description": "Focus on quality over speed"
    },
    "fast": {
        "extractors": ["pypdf", "pdftohtml"],
        "description": "Fast extraction, basic formatting"
    },
    "scanned": {
        "extractors": ["ocr", "pymupdf"],
        "description": "For scanned PDFs"
    },
    "tables": {
        "extractors": ["pdfplumber", "pdfminer"],
        "description": "Optimized for documents with tables"
    }
}

# --- MAIN EXECUTION ---

def detect_pdf_type(pdf_path):
    """Try to detect if PDF is scanned or has selectable text"""
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return "empty"
            
            # Check first few pages for text
            text_found = False
            for i in range(min(3, len(doc))):
                text = doc[i].get_text().strip()
                if len(text) > 50:  # More than 50 chars suggests real text
                    text_found = True
                    break
            
            if not text_found:
                # Check if it has images (might be scanned)
                for i in range(min(3, len(doc))):
                    if doc[i].get_images():
                        return "scanned"
                return "empty"
            
            return "text"
    except:
        return "unknown"

def main():
    parser = argparse.ArgumentParser(
        description="Robust PDF to Ebook Converter with multiple fallback methods.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pdf_file", help="Path to the source PDF file.")
    parser.add_argument(
        "--profile",
        choices=PROFILES.keys(),
        help="Use a predefined extraction profile:\n" + 
             "\n".join(f"  {k}: {v['description']}" for k, v in PROFILES.items())
    )
    parser.add_argument(
        "--pdfextractor",
        choices=EXTRACTORS.keys(),
        help="Force a specific PDF extractor"
    )
    parser.add_argument(
        "--generator",
        choices=GENERATORS.keys(),
        help="Force a specific ebook generator"
    )
    parser.add_argument(
        "--list-extractors",
        action="store_true",
        help="List all available extractors and their status"
    )
    parser.add_argument(
        "--output-format",
        choices=["mobi", "epub", "azw3"],
        default="mobi",
        help="Output format (default: mobi)"
    )
    parser.add_argument(
        "--include-imgs",
        action="store_true",
        help="Include images in the extraction"
    )
    parser.add_argument(
        "--no-html-cleanup",
        action="store_true",
        help="Skip HTML cleanup step"
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files for debugging"
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect PDF type and choose best extractor"
    )
    args = parser.parse_args()

    # List extractors if requested
    if args.list_extractors:
        print("Available PDF extractors:")
        for name, func in EXTRACTORS.items():
            status = "✓" if name in ["pymupdf", "pdftohtml"] else "?"
            if name == "pdfplumber":
                status = "✓" if HAS_PDFPLUMBER else "✗"
            elif name == "pypdf":
                status = "✓" if HAS_PYPDF else "✗"
            elif name == "pdfminer":
                status = "✓" if HAS_PDFMINER else "✗"
            print(f"  {status} {name}")
        sys.exit(0)

    # Process PDF file
    pdf_file = os.path.abspath(args.pdf_file)
    
    if not os.path.exists(pdf_file) or not pdf_file.lower().endswith(".pdf"):
        print(f"Error: File not found or is not a PDF: '{pdf_file}'")
        sys.exit(1)

    # Check for DRM
    if detect_drm(pdf_file):
        print("Error: This PDF appears to be DRM-protected.")
        print("Please remove DRM before conversion.")
        sys.exit(1)

    # Auto-detect PDF type if requested
    if args.auto_detect:
        pdf_type = detect_pdf_type(pdf_file)
        print(f"[Info] Detected PDF type: {pdf_type}")
        if pdf_type == "scanned":
            args.profile = "scanned"
        elif pdf_type == "empty":
            print("Error: PDF appears to be empty")
            sys.exit(1)

    # Setup filenames
    base_name = os.path.splitext(pdf_file)[0]
    html_file = base_name + "_intermediate.html"
    
    # Determine output file based on format
    output_extensions = {
        "mobi": "_text_only.mobi",
        "epub": "_text_only.epub",
        "azw3": "_text_only.azw3"
    }
    output_file = base_name + output_extensions.get(args.output_format, "_output.ebook")
    
    intermediate_files = [html_file]
    
    print(f"--- Starting conversion for: {pdf_file} ---")
    print(f"Target format: {args.output_format.upper()}")

    # Determine extractor order
    if args.pdfextractor:
        extractor_order = [args.pdfextractor]
    elif args.profile:
        extractor_order = PROFILES[args.profile]["extractors"]
    else:
        extractor_order = DEFAULT_EXTRACTOR_ORDER

    # Stage 1: Extract text
    extracted_file, extracted_type = None, None
    
    extractor_args = {
        "pdf_path": pdf_file,
        "html_path": html_file,
        "include_images": args.include_imgs,
        "no_html_cleanup": args.no_html_cleanup
    }
    
    print(f"[Info] Trying extractors in order: {', '.join(extractor_order)}")
    
    for extractor_name in extractor_order:
        if extractor_name in EXTRACTORS:
            extractor_func = EXTRACTORS[extractor_name]
            extracted_file, extracted_type = extractor_func(**extractor_args)
            if extracted_file:
                print(f"[Info] Successfully extracted with: {extractor_name}")
                break
    
    if not extracted_file:
        print("\n[FATAL] All PDF text extraction methods failed.")
        print("Consider trying --profile=scanned if this is a scanned PDF")
        sys.exit(1)

    # Stage 2: Generate ebook
    final_file = None
    
    # Map format to appropriate generators
    if args.generator:
        generator_order = [args.generator]
    else:
        if args.output_format == "mobi":
            generator_order = ["ebook_converter", "calibre"]
        else:
            generator_order = ["pandoc", "ebooklib", "calibre"]
    
    print(f"[Info] Trying generators in order: {', '.join(generator_order)}")
    
    for generator_name in generator_order:
        if generator_name in GENERATORS:
            gen_func = GENERATORS[generator_name]
            
            # Adjust output path based on generator capabilities
            if generator_name == "ebooklib" and args.output_format != "epub":
                print(f"    [Skip] ebooklib only supports EPUB")
                continue
            
            if generator_name == "ebook_converter" and args.output_format != "mobi":
                print(f"    [Skip] ebook_converter only tested with MOBI")
                continue
            
            final_file = gen_func(
                input_path=extracted_file,
                input_type=extracted_type,
                output_path=output_file,
                title=os.path.basename(base_name)
            )
            
            if final_file:
                print(f"[Info] Successfully generated with: {generator_name}")
                break

    if not final_file:
        print("\n[FATAL] All ebook generation methods failed.")
        print(f"Intermediate HTML file kept at: '{extracted_file}'")
        sys.exit(1)

    # Stage 3: Cleanup
    if args.keep_intermediate:
        print(f"[*] Keeping intermediate files: {', '.join(intermediate_files)}")
    else:
        for f in intermediate_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"[*] Cleaned up: '{f}'")
                except Exception as e:
                    print(f"    [Warning] Could not remove '{f}': {e}")
    
    print("\n--- Conversion complete! ✨ ---")
    print(f"Output file: {final_file}")
    print(f"Format: {args.output_format.upper()}")
    print(f"Size: {os.path.getsize(final_file) / 1024:.1f} KB")

if __name__ == "__main__":
    main()