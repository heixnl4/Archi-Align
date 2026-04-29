import docx
import re

def clean_and_chunk_docx(file_path, chunk_size=500, overlap=50):
    """
    读取 Word 文档，清洗无用字符，并按字数进行滑动窗口切片
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        text = para.text.strip()
        # 简单清洗：去除连续的空格和特殊符号
        text = re.sub(r'\s+', ' ', text)
        if len(text) > 5:  # 过滤掉太短的无意义段落
            full_text.append(text)
            
    content = " ".join(full_text)
    
    chunks = []
    # 滑动窗口切片，overlap 保证上下文不被生硬截断
    for i in range(0, len(content), chunk_size - overlap):
        chunk = content[i:i + chunk_size]
        if len(chunk) > 100: # 过滤掉末尾太短的切片
            chunks.append(chunk)
            
    return chunks