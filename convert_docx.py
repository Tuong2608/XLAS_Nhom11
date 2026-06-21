"""
Script đọc và chuyển đổi file Word (.docx) thành Markdown
"""

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

def read_docx(docx_path):
    with zipfile.ZipFile(docx_path, 'r') as z:
        xml_content = z.read('word/document.xml')
    
    tree = ET.fromstring(xml_content)
    paragraphs = []
    
    for para in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        texts = []
        for text_elem in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
            if text_elem.text:
                texts.append(text_elem.text)
        
        pPr = para.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')
        style = None
        if pPr is not None:
            pStyle = pPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle')
            if pStyle is not None:
                style = pStyle.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
        
        text = ''.join(texts)
        if text.strip():
            paragraphs.append({'text': text, 'style': style})
    
    return paragraphs

def convert_to_markdown(paragraphs):
    markdown_lines = []
    
    for para in paragraphs:
        text = para['text'].strip()
        style = para['style']
        
        if not text:
            markdown_lines.append('')
            continue
        
        if style and 'Heading' in str(style):
            level = style.replace('Heading', '')
            if level.isdigit():
                level = int(level)
                if level == 1:
                    markdown_lines.append(f"# {text}")
                elif level == 2:
                    markdown_lines.append(f"## {text}")
                elif level == 3:
                    markdown_lines.append(f"### {text}")
                else:
                    markdown_lines.append(f"#### {text}")
            else:
                markdown_lines.append(f"## {text}")
        else:
            markdown_lines.append(text)
    
    return '\n'.join(markdown_lines)

if __name__ == "__main__":
    docx_path = Path(r"C:\Users\Acer\Downloads\BaoCao_PhatHienChayRung_Nhom11_FINAL.docx")
    output_path = Path(r"D:\Project_CK_XLAS_Nhom11_ForGithub\BAO_CAO.md")
    
    if not docx_path.exists():
        print("File not found:", docx_path)
        exit(1)
    
    print("Reading:", docx_path)
    paragraphs = read_docx(docx_path)
    print("Found", len(paragraphs), "paragraphs")
    
    markdown_content = convert_to_markdown(paragraphs)
    output_path.write_text(markdown_content, encoding='utf-8')
    print("Saved to:", output_path)
    print("\nPreview (first 5000 chars):")
    print(markdown_content[:5000])
