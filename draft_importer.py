import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import textstat

class DraftImporter:
    """Core document importer and validator (backend only)"""
    
    def __init__(self, input_dir: str = "input_drafts"):
        self.input_dir = input_dir
        Path(input_dir).mkdir(exist_ok=True)
    
    def get_available_files(self) -> List[Dict]:
        """List available draft files with metadata"""
        files = []
        for filepath in Path(self.input_dir).glob("*"):
            if filepath.is_file() and filepath.suffix.lower() in ('.txt', '.docx', '.pdf'):
                files.append({
                    'filename': filepath.name,
                    'path': str(filepath),
                    'size': f"{filepath.stat().st_size / 1024:.1f} KB",
                    'type': filepath.suffix[1:].upper()
                })
        return files

    def import_draft(self, filepath: str) -> Dict:
        """Core import logic - returns raw structured data"""
        path = Path(filepath)
        content = self._read_file_content(path)
        
        is_valid, validation_msg = self._validate_structure(content)
        articles = self._parse_articles(content)
        
        return {
            'filename': path.name,
            'content': content,
            'is_valid': is_valid,
            'validation_msg': validation_msg,
            'articles': articles,
            'imported_at': datetime.now().isoformat()
        }

    def _read_file_content(self, path: Path) -> str:
        """Universal file content reader"""
        if path.suffix.lower() == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif path.suffix.lower() == '.docx':
            return self._read_docx(path)
        elif path.suffix.lower() == '.pdf':
            return self._read_pdf(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _read_docx(self, path: Path) -> str:
        """DOCX specific reader"""
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            raise ImportError("python-docx required for Word files")

    def _read_pdf(self, path: Path) -> str:
        """PDF specific reader"""
        try:
            from PyPDF2 import PdfReader
            with open(path, 'rb') as f:
                return "\n".join(page.extract_text() for page in PdfReader(f).pages)
        except ImportError:
            raise ImportError("PyPDF2 required for PDF files")

    def _validate_structure(self, text: str) -> Tuple[bool, str]:
        """Validate legal document structure"""
        if not text.strip():
            return False, "Empty document"
        if not re.search(r'^(Article|Section)\s+\w+', text, re.MULTILINE | re.IGNORECASE):
            return False, "No valid article/section headers found"
        return True, "Valid structure"

    def _parse_articles(self, text: str) -> Dict[str, str]:
        """Extract articles/sections and subprovisions from text"""
        articles = {}
        current_article = None
        current_content = []
        current_subprovision = None
        current_subcontent = []
        
        def save_subprovision():
            if current_subprovision:
                if current_article not in articles:
                    articles[current_article] = {}
                articles[current_article][current_subprovision] = '\n'.join(current_subcontent).strip()
        
        def save_article():
            if current_article:
                if current_subprovision:
                    save_subprovision()
                else:
                    articles[current_article] = '\n'.join(current_content).strip()
        
        for line in text.split('\n'):
            line = line.strip()
            # Detect article headers
            if re.match(r'^(Article|Section|PART)\s+.*', line, re.IGNORECASE) or re.match(r'^\d+\.', line):
                save_article()
                current_article = line
                current_content = []
                current_subprovision = None
                current_subcontent = []
            # Detect subprovisions like (1), (a), (i)
            elif re.match(r'^\(\w+\)', line):
                if current_subprovision:
                    save_subprovision()
                current_subprovision = line
                current_subcontent = []
            else:
                if current_subprovision:
                    current_subcontent.append(line)
                elif current_article:
                    current_content.append(line)
                else:
                    # Lines before first article
                    if None not in articles:
                        articles[None] = []
                    articles[None].append(line)
        
        save_article()
        if None in articles:
            articles['Introduction'] = '\n'.join(articles.pop(None)).strip()
        
        return articles
