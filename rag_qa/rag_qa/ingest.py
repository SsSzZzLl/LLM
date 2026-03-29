from __future__ import annotations

import re
from pathlib import Path
from typing import List

from rag_qa.chunking import Chunk, chunk_document


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text_with_ocr(pdf_path: Path, lang: str = "chi_sim+eng") -> str:
    """使用 OCR 从扫描件 PDF 中提取文本（使用 PyMuPDF，无需 Poppler）"""
    try:
        import pytesseract
        from PIL import Image
        import fitz  # PyMuPDF
        import io
        from pathlib import Path
        
        # 设置 Tesseract 路径（如果不在 PATH 中）
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"D:\apps\tesseract.exe",
            r"D:\Tesseract-OCR\tesseract.exe",
        ]
        for tess_path in tesseract_paths:
            if Path(tess_path).exists():
                pytesseract.pytesseract.tesseract_cmd = tess_path
                break
    except ImportError:
        raise ImportError("请安装 OCR 依赖: pip install pytesseract pillow pymupdf")
    
    text_parts = []
    
    print(f"  - Converting PDF to images for OCR...")
    
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        
        for i, page in enumerate(doc, 1):
            print(f"  - OCR processing page {i}/{total_pages}...")
            
            # 使用 PyMuPDF 将页面转为图片（2x 缩放提高 OCR 精度）
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # 转换为 PIL Image
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # 使用 Tesseract 进行 OCR
            page_text = pytesseract.image_to_string(img, lang=lang)
            cleaned_text = clean_text(page_text)
            
            if cleaned_text.strip():
                text_parts.append(cleaned_text)
    
    return "\n\n".join(text_parts)


def is_scanned_pdf(doc) -> bool:
    """检测 PDF 是否为扫描件（图片型）"""
    total_pages = len(doc)
    empty_text_pages = 0
    
    for page in doc:
        text = page.get_text().strip()
        if not text:
            empty_text_pages += 1
    
    # 如果超过 80% 的页面没有文本，认为是扫描件
    return empty_text_pages / total_pages > 0.8 if total_pages > 0 else False


def extract_text_from_pdf(pdf_path: Path, use_ocr: bool = True) -> str:
    """使用 PyMuPDF (fitz) 从 PDF 中提取文本并进行清洗，支持 OCR 处理扫描件"""
    import fitz  # PyMuPDF
    
    text_parts = []
    
    with fitz.open(pdf_path) as doc:
        # 检测是否为扫描件
        if use_ocr and is_scanned_pdf(doc):
            print(f"  - Detected scanned PDF, using OCR...")
            return extract_text_with_ocr(pdf_path)
        
        # 正常文本提取
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            
            # 清洗文本
            cleaned_text = clean_text(page_text)
            
            if cleaned_text.strip():
                text_parts.append(cleaned_text)
    
    return "\n\n".join(text_parts)


def clean_text(text: str) -> str:
    """清洗文本：去除多余空格、换行符，过滤页码等"""
    # 去除多余空格（多个空格合并为一个）
    text = re.sub(r' +', ' ', text)
    
    # 去除多余换行（保留段落分隔）
    text = re.sub(r'\n+', '\n', text)
    
    # 过滤简单的页码（单独一行的数字，通常是页码）
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过纯数字的页码行
        if stripped and not re.match(r'^\d+$', stripped):
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def load_pdf_files(data_dir: Path, use_ocr: bool = True) -> dict[str, str]:
    """加载指定目录下的所有 PDF 文件，返回文件名到文本内容的映射"""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    pdf_contents = {}
    
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        print(f"Loading PDF: {pdf_path.name}")
        try:
            text = extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
            pdf_contents[pdf_path.name] = text
            print(f"  - Extracted {len(text)} characters")
        except Exception as e:
            print(f"  - Error loading {pdf_path.name}: {e}")
    
    return pdf_contents


def load_corpus(corpus_dir: Path, max_chars: int, overlap_chars: int) -> List[Chunk]:
    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_chunks: List[Chunk] = []
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            text = read_text_file(p)
            source_id = str(p.relative_to(corpus_dir)).replace("\\", "/")
            all_chunks.extend(chunk_document(text, source_id, max_chars, overlap_chars))
    return all_chunks


if __name__ == "__main__":
    """测试入口：加载 PDF 文件并进行分块，打印前两个分块"""
    import sys
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 数据目录路径（项目根目录下的 data/）
    data_dir = project_root.parent / "data"
    
    print("=" * 60)
    print("RAG QA PDF Ingest & Chunking Test")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print()
    
    # 检查数据目录是否存在
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        print("Please create the directory and add BBQ.pdf and ImplicitBBQ.pdf")
        sys.exit(1)
    
    # 加载所有 PDF 文件（启用 OCR）
    print("Step 1: Loading PDF files...")
    print("(OCR enabled for scanned documents)")
    print()
    pdf_contents = load_pdf_files(data_dir, use_ocr=True)
    
    if not pdf_contents:
        print("\nNo PDF files found in the data directory.")
        print("Please ensure BBQ.pdf and ImplicitBBQ.pdf are placed in:")
        print(f"  {data_dir}")
        sys.exit(1)
    
    print(f"\nSuccessfully loaded {len(pdf_contents)} PDF file(s)")
    print()
    
    # 对每个 PDF 进行分块
    print("Step 2: Chunking documents...")
    print(f"Parameters: chunk_size=1000, chunk_overlap=150")
    print()
    
    all_chunks = []
    for filename, text in pdf_contents.items():
        print(f"Processing: {filename}")
        chunks = chunk_document(text, filename, max_chars=1000, overlap_chars=150)
        all_chunks.extend(chunks)
        print(f"  - Generated {len(chunks)} chunks")
    
    print()
    print("=" * 60)
    print(f"Total chunks generated: {len(all_chunks)}")
    print("=" * 60)
    print()
    
    # 打印前两个分块
    if len(all_chunks) >= 1:
        print("\n" + "=" * 60)
        print("FIRST CHUNK:")
        print("=" * 60)
        print(f"Source: {all_chunks[0].source_id}")
        print(f"Index: {all_chunks[0].chunk_index}")
        print(f"Length: {len(all_chunks[0].text)} characters")
        print("-" * 60)
        print(all_chunks[0].text[:500] + "..." if len(all_chunks[0].text) > 500 else all_chunks[0].text)
        print()
    
    if len(all_chunks) >= 2:
        print("\n" + "=" * 60)
        print("SECOND CHUNK:")
        print("=" * 60)
        print(f"Source: {all_chunks[1].source_id}")
        print(f"Index: {all_chunks[1].chunk_index}")
        print(f"Length: {len(all_chunks[1].text)} characters")
        print("-" * 60)
        print(all_chunks[1].text[:500] + "..." if len(all_chunks[1].text) > 500 else all_chunks[1].text)
        print()
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
