from utils.ent_ai import extract_entities

def test1():
    text = 'Воздуховод-0.6-400-3000-оц.-г.г'
    print(extract_entities(text))

def test2():
    text = 'Воздуховод-d-400-3000-оц.-0.6-г.г'
    print(extract_entities(text))

def test3():
    text = 'воздуховод круглый-400-3000-оц.-0.7-г.г'
    print(extract_entities(text))


test3()