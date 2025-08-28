from utils.ent_ai import extract_entities
from utils.excel_log import get_top_ent

def test1():
    text = 'Воздуховод-0.6-400-3000-оц.-г.г'
    print(extract_entities(text))

def test2():
    text = 'Воздуховод-d-400-3000-оц.-0.6-г.г'
    print(extract_entities(text))

def test3():
    text = 'Воздуховод круглый Ø100'
    extract_entities(text)

def test4():
    get_top_ent('files/results/emb_results', 'files/results/finally_results')



test4()