from json import dump
from random import randrange

obj_list = []

with open('test.json','w+', encoding='utf-8') as f:
    for _ in range(100):
        obj_list.append({'tokens':
                             ['Воздуховод', 'круглый',  'Ø', str(randrange(50, 500, 10))],
                         'ner_tags':
                            ['B-TYPE', 'O', 'O', 'B-FSIZE']
                         })
        obj_list.append({'tokens':
                         ['Воздуховод', 'прямоугольный',  str(randrange(50, 500, 10)), 'x', str(randrange(50, 500, 10))],
                         'ner_tags':
                         ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-SSIZE']
                         })
    print(obj_list)
    dump(obj_list, f, ensure_ascii=False, indent=4)
