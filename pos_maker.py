from random import randint, shuffle, randrange, choice
from json import load, dump

flange_types = ['т2', 'т3', 'ш2', 'ш3', 'н', 'крф', 'пф', 'г']
material_types = ['оц', 'AISI 304', 'AISI 316', 'AISI 430', 'нерж', 'ч.мет', 'Al']

def make_voz():
    with open('training/voz_pos.json', 'w+', encoding='utf-8') as f:

        pos_list = []
        for _ in range(200):
            size1 = str(randrange(100, 1000, 10))
            size2 = str(randrange(100, 1000, 10))

            l_size = str(randrange(50, 1000, 10))
            thickness = str(randrange(1, 10, 1) / 10)

            type1 = {'tokens': ['воздуховод', '-', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type2 = {'tokens': ['воздуховод', '-', size1, '*', size2, '-' ,l_size ,'-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-SSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            text_t1 = type1
            text_t2 = type2

            pos_list.append(text_t1)
            pos_list.append(text_t2)

        dump(pos_list, f, ensure_ascii=False, indent=4)

make_voz()
