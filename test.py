from utils.filters import extract_sizes


def test1():
    text = 'Переход-1-150*350/150-300-20-20-оц.-0.7-т2.ш2'
    print(extract_sizes(text))

def test2():
    text = 'Переход-100/95-64-50-50-оц.-0.5-з.з'
    print(extract_sizes(text))

test2()
