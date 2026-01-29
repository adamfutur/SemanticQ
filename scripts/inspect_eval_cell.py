import json

nb_path = r'c:\Users\MSI\Desktop\SemanticQ\notebooks\semantic_bloom_full-update.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('cells_dump.txt', 'w', encoding='utf-8') as f_out:
    for i, cell in enumerate(nb['cells']):
        if i > 10 and cell['cell_type'] == 'code':
            f_out.write(f"\n--- CELL {i} SOURCE ---\n")
            f_out.write("".join(cell['source']))
