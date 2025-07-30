import re

def extract_sizes(text):
    text = text.lower()

    # 600x300, 600/300
    size_patterns = re.findall(r'(\d{2,4})[xх*/\-\\](\d{2,4})', text)
    sizes = set((int(a), int(b)) for a, b in size_patterns)
    return sizes