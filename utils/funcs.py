from utils.filters import extract_sizes

def add_size_filter(price_items, query_item):
    sizes = {}

    for i in price_items:
        query_sizes = extract_sizes(query_item)
        price_item_sizes = extract_sizes(i)

        sizes.setdefault(query_sizes, []).append(price_item_sizes)

    return sizes
