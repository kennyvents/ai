import re

def extract_sizes(text):
    type_pattern = re.search(r'-(\d)-', text)

    if type_pattern is None:
        sizes_pattern = re.search(r'-(\d{2,4})/(\d{2,4})', text)
        sizes_pattern = [sizes_pattern.group(1), sizes_pattern.group(2)]
        form = 'круглый'

    else:
        type_pattern = type_pattern.group(1)
        sizes_pattern = re.findall(r'(\d{2,4}[*]\d{2,4})', text)
        form = 'прямоугольный'

        if len(sizes_pattern) != 2:
            sizes_pattern = re.search(r'(\d{2,4}[*]\d{2,4})/(\d{2,4})', text)
            sizes_pattern = [sizes_pattern.group(1), sizes_pattern.group(2)]
            form = 'прямоугольный на круглый'

    l_pattern = re.search(r'-(\d{2,4})-', text).group(1)
    tire_pattern = re.search(r'-\d{2,4}-(\d{2}-\d{2})-[а-я]', text).group(1)

    size_dict = {
        'type': type_pattern,
        'size': sizes_pattern,
        'l': l_pattern,
        'tire': tire_pattern,
        'form': form
    }
    return size_dict