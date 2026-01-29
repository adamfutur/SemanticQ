import json
import os

nb_path = r'c:\Users\MSI\Desktop\SemanticQ\notebooks\semantic_bloom_full-update.ipynb'

if not os.path.exists(nb_path):
    print(f"Error: File not found at {nb_path}")
    exit(1)

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Iterate through cells to find the config
    changes_made = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "'path': '../data/raw/bloom_questions.csv'" in line:
                    line = line.replace("'path': '../data/raw/bloom_questions.csv'", "'path': '../data/raw/bloom_questions_augmented.csv'")
                    changes_made = True
                    print("Found and replaced path configuration.")
                new_source.append(line)
            cell['source'] = new_source

    if changes_made:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully.")
    else:
        print("No changes needed or target string not found.")

except Exception as e:
    print(f"An error occurred: {e}")
