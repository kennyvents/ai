from random import randint, shuffle, randrange, choice
from json import load, dump

flange_types = ['т2', 'т3', 'ш2', 'ш3', 'н', 'крф', 'пф', 'г']
material_types = ['оц', 'AISI 304', 'AISI 316', 'AISI 430', 'нерж', 'ч.мет', 'Al']

def make_voz():
    with open('voz_pos.json', 'w+', encoding='utf-8') as f:

        pos_list = []
        for _ in range(20):
            size1 = str(randrange(100, 1000, 10))
            size2 = str(randrange(100, 1000, 10))

            l_size = str(randrange(50, 1000, 10))
            thickness = str(randrange(1, 10, 1) / 10)

            type1 = {'tokens': ['воздуховод', '-', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type2 = {'tokens': ['воздуховод', '-', size1, '*', size2, '-' ,l_size ,'-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-SSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            # d
            type3 = {'tokens': ['воздуховод', 'Ø', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type4 = {'tokens': ['воздуховод', 'φ', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type5 = {'tokens': ['воздуховод', 'D', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type6 = {'tokens': ['воздуховод', 'd', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type7 = {'tokens': ['воздуховод', '⌀', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            #-d-
            type8 = {'tokens': ['воздуховод', '-', 'Ø', '-', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'O', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type9 = {'tokens': ['воздуховод', '-', 'φ', '-', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'O', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type10 = {'tokens': ['воздуховод','-', 'D', '-', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE','O', 'O', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type11 = {'tokens': ['воздуховод','-', 'd','-', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'O', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}

            type12 = {'tokens': ['воздуховод','-', '⌀', '-', size1, '-', l_size, '-', choice(material_types), '.', '-', thickness, '-', f'{choice(flange_types)}.{choice(flange_types)}'],
                     'ner_tags': ['B-TYPE', 'O', 'O', 'O', 'B-FSIZE', 'O', 'B-LENGTH', 'O', 'B-MATERIAL', 'O', 'O', 'B-THICKNESS', 'O', 'B-FLANGE']}


            pos_list.append(type3)
            pos_list.append(type4)
            pos_list.append(type5)
            pos_list.append(type6)
            pos_list.append(type7)
            pos_list.append(type8)
            pos_list.append(type9)
            pos_list.append(type10)
            pos_list.append(type11)
            pos_list.append(type12)

        dump(pos_list, f, ensure_ascii=False, indent=4)

make_voz()
