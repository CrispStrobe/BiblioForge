#!/usr/bin/env python3
"""
Versatile document processor with proper fallback handling and response parsing
"""

from docstrange import DocumentExtractor
import json
import re
import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class UniversalDocumentProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the document processor"""
        self.extractor = DocumentExtractor(api_key=api_key)
        self.supported_formats = [
            '.pdf', '.docx', '.doc', '.txt', '.epub', '.mobi', '.azw3', 
            '.fb2', '.html', '.htm', '.odt', '.rtf', '.pdb', '.lrf'
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def convert_with_ebook_converter(self, input_path: str, output_format: str = 'txt') -> Optional[str]:
        """Convert document using ebook-converter"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp:
                output_path = tmp.name
            
            # Run ebook-converter
            cmd = ['ebook-converter', input_path, output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                os.unlink(output_path)
                return content
            else:
                print(f"⚠️  ebook-converter failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⚠️  ebook-converter timed out")
            return None
        except Exception as e:
            print(f"⚠️  ebook-converter error: {e}")
            return None
    
    def extract_text_with_fallback(self, file_path: str, start_page: int = 1, 
                                   end_page: Optional[int] = None) -> Tuple[str, bool, Optional[str]]:
        """Extract text with ebook-converter fallback if needed
        Returns: (text, used_fallback, temp_file_path)
        """
        
        # First, try docstrange
        try:
            result = self.extractor.extract(file_path)
            markdown = result.extract_markdown()
            
            # Check if extraction was successful
            if markdown and len(markdown.strip()) > 50:  # Reasonable content
                # Handle page range for PDFs
                if Path(file_path).suffix.lower() == '.pdf' and (start_page > 1 or end_page):
                    markdown = self._filter_pages(markdown, start_page, end_page)
                return markdown, False, None
        except Exception as e:
            print(f"⚠️  Docstrange extraction failed: {e}")
        
        # Fallback to ebook-converter
        print("📄 Using ebook-converter fallback...")
        text_content = self.convert_with_ebook_converter(file_path, 'txt')
        
        if text_content:
            # Save to temporary file for API processing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(text_content)
                temp_file_path = tmp.name
            
            # Format as markdown
            lines = text_content.split('\n')
            formatted = []
            for line in lines:
                if line.strip():
                    formatted.append(line.strip())
            return '\n\n'.join(formatted), True, temp_file_path
        
        return "", True, None
    
    def _filter_pages(self, markdown: str, start_page: int, end_page: Optional[int]) -> str:
        """Filter markdown to include only specified page range"""
        pages = markdown.split('## Page ')
        output_parts = []
        
        if pages[0].strip():
            output_parts.append(pages[0].strip())
        
        for page_content in pages[1:]:
            if not page_content.strip():
                continue
            
            page_num_match = re.match(r'^(\d+)', page_content)
            if page_num_match:
                page_num = int(page_num_match.group(1))
                if page_num >= start_page and (end_page is None or page_num <= end_page):
                    output_parts.append(f"## Page {page_content}")
        
        return '\n\n'.join(output_parts)
    
    def parse_api_response(self, raw_result: Dict) -> Dict:
        """Parse various API response formats"""
        extracted_fields = {}
        
        # Check for direct extracted_fields response
        if 'extracted_fields' in raw_result:
            return raw_result['extracted_fields']
        
        # Check for raw_content with JSON
        if 'document' in raw_result and 'raw_content' in raw_result['document']:
            raw = raw_result['document']['raw_content']
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, raw)
            
            for match in matches:
                try:
                    data = json.loads(match)
                    extracted_fields.update(data)
                except:
                    pass
        
        return extracted_fields
    
    def extract_fields_from_file(self, file_path: str, fields: List[str]) -> Dict:
        """Extract fields from a file (works with text files in cloud mode)"""
        try:
            result = self.extractor.extract(file_path)
            raw_result = result.extract_data(specified_fields=fields)
            return self.parse_api_response(raw_result)
            
        except Exception as e:
            print(f"⚠️  Field extraction failed: {e}")
            return {}
    
    def process_document(self, file_path: str, mode: str = "all", **kwargs):
        """Main processing function with different modes"""
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        if not self.is_supported_format(file_path):
            print(f"❌ Unsupported format: {Path(file_path).suffix}")
            print(f"Supported formats: {', '.join(self.supported_formats)}")
            return
        
        print(f"📄 Processing: {file_path}")
        print(f"📋 Format: {Path(file_path).suffix}")
        print("=" * 70)
        
        # Track temp file for cleanup
        temp_text_file = None
        
        try:
            if mode in ["text", "all"]:
                start_page = kwargs.get("start_page", 1)
                end_page = kwargs.get("end_page", None)
                
                page_info = f"Pages {start_page}-{end_page or 'all'}" if Path(file_path).suffix.lower() == '.pdf' else "All content"
                print(f"\n📝 Text Extraction ({page_info}):")
                print("-" * 50)
                
                text, used_fallback, temp_text_file = self.extract_text_with_fallback(file_path, start_page, end_page)
                
                if used_fallback:
                    print("ℹ️  Used ebook-converter for extraction")
                
                # Limit output for display
                if len(text) > 1000 and not kwargs.get("full_output", False):
                    print(text[:1000] + "\n... (truncated, use --full-output to see all)")
                else:
                    print(text)
                    
                # Store temp file path for other modes
                kwargs['_temp_text_file'] = temp_text_file
                kwargs['_used_fallback'] = used_fallback
            
            if mode in ["raw", "all"]:
                fields = kwargs.get("fields", ["author", "title", "year"])
                
                print(f"\n🔧 Raw API Response for fields: {fields}")
                print("-" * 50)
                
                # If we used fallback, process the temp text file
                if kwargs.get('_used_fallback') and kwargs.get('_temp_text_file'):
                    print("ℹ️  Using text file from fallback extraction")
                    raw = self.extractor.extract(kwargs['_temp_text_file']).extract_data(specified_fields=fields)
                else:
                    result = self.extractor.extract(file_path)
                    raw = result.extract_data(specified_fields=fields)
                
                print(json.dumps(raw, indent=2))
            
            if mode in ["metadata", "all"]:
                fields = kwargs.get("metadata_fields", None) or [
                    "author_full_name",
                    "document_title",
                    "publication_year",
                    "language",
                    "publisher"
                ]
                
                print("\n📚 Structured Metadata:")
                print("-" * 50)
                
                # If we used fallback, process the temp text file
                if kwargs.get('_used_fallback') and kwargs.get('_temp_text_file'):
                    print("ℹ️  Using text file from fallback extraction")
                    metadata = self.extract_fields_from_file(kwargs['_temp_text_file'], fields)
                else:
                    metadata = self.extract_fields_from_file(file_path, fields)
                
                # Clean up metadata
                metadata = {k: v for k, v in metadata.items() 
                           if v and v not in ["None", "null", None, ""]}
                
                if metadata:
                    for field, value in metadata.items():
                        print(f"{field}: {value}")
                else:
                    print("⚠️  No metadata extracted")
                
                # Save if requested
                if kwargs.get("save_metadata", False) and metadata:
                    filename = kwargs.get("output_file", "metadata.json")
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    print(f"\n💾 Metadata saved to {filename}")
                    
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup temp file
            if temp_text_file and os.path.exists(temp_text_file):
                try:
                    os.unlink(temp_text_file)
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(
        description="Universal document processor with docstrange and ebook-converter fallback"
    )
    
    parser.add_argument("file_path", help="Path to the document file")
    
    parser.add_argument("--mode", choices=["text", "raw", "metadata", "all"], 
                       default="all", help="Processing mode")
    
    # Text extraction options
    parser.add_argument("--start-page", type=int, default=1, 
                       help="Start page for text extraction (PDF only)")
    parser.add_argument("--end-page", type=int, 
                       help="End page for text extraction (PDF only)")
    parser.add_argument("--full-output", action="store_true",
                       help="Show full text output without truncation")
    
    # Field extraction options
    parser.add_argument("--fields", nargs="+", 
                       default=["author", "title", "year"],
                       help="Fields to extract for raw mode")
    parser.add_argument("--metadata-fields", nargs="+",
                       help="Custom metadata fields to extract")
    
    # Output options
    parser.add_argument("--save-metadata", action="store_true",
                       help="Save metadata to JSON file")
    parser.add_argument("--output-file", default="metadata.json",
                       help="Output filename for metadata")
    
    # API options
    parser.add_argument("--api-key", help="Nanonets API key")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = UniversalDocumentProcessor(api_key=args.api_key)
    
    # Process document
    processor.process_document(
        args.file_path,
        mode=args.mode,
        start_page=args.start_page,
        end_page=args.end_page,
        full_output=args.full_output,
        fields=args.fields,
        metadata_fields=args.metadata_fields,
        save_metadata=args.save_metadata,
        output_file=args.output_file
    )

if __name__ == "__main__":
    # If run directly without arguments, use example
    import sys
    
    if len(sys.argv) == 1:
        print("Usage: python universal_processor.py <file_path> [options]")
        print("\nSupported formats:", ', '.join(UniversalDocumentProcessor().supported_formats))
        print("\nExamples:")
        print("  python universal_processor.py book.pdf --mode text --start-page 1 --end-page 10")
        print("  python universal_processor.py document.epub --mode metadata --save-metadata")
        print("  python universal_processor.py article.mobi --mode all")
    else:
        main()