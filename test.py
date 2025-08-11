from utils.ai import extract_entities



def test2():
    text = 'Воздуховод-400-3000-оц.-0.6-г.г'
    text = text.lower()
    print(extract_entities(text))

test2()
