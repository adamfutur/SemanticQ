import sys
import codecs
import re

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        sys.exit(1)

pdf_path = r'c:\Users\MSI\Desktop\SemanticQ\11- Semantic-BERT and semantic-FastText models for  education question classification.pdf'

try:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                text += t + "\n"
        except:
            pass
    
    with open('fasttext_details.txt', 'w', encoding='utf-8') as f:
        # Locate "FastText"
        indices = [m.start() for m in re.finditer('FastText', text, re.IGNORECASE)]
        
        f.write(f"Found {len(indices)} occurrences of FastText\n")
        
        for idx in indices: 
            start = max(0, idx - 1000)
            end = min(len(text), idx + 1000)
            chunk = text[start:end].replace('\n', ' ')
            
            # Filter for relevant chunks only
            if any(x in chunk.lower() for x in ['sentence vector', 'algorithm', 'equation', 'average', 'mean', 'embedding', 'model', 'training']):
                f.write(f"\n--- CONTEXT {idx} ---\n")
                f.write(chunk)
                f.write("\n")

except Exception as e:
    with open('fasttext_details.txt', 'w', encoding='utf-8') as f:
        f.write(str(e))
