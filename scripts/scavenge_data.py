import pandas as pd
import os
import re

def standardize_level(level):
    if not isinstance(level, str):
        return None
    level = level.strip().lower()
    
    mapping = {
        'remember': 'Remember',
        'remembering': 'Remember',
        'knowledge': 'Remember',
        'understand': 'Understand',
        'understanding': 'Understand',
        'comprehension': 'Understand',
        'apply': 'Apply',
        'applying': 'Apply',
        'application': 'Apply',
        'analyze': 'Analyze',
        'analyzing': 'Analyze',
        'analysis': 'Analyze',
        'evaluate': 'Evaluate',
        'evaluating': 'Evaluate',
        'evaluation': 'Evaluate',
        'create': 'Create',
        'creating': 'Create',
        'synthesis': 'Create'
    }
    
    return mapping.get(level, None)

def process_format_2(path):
    """Format: taxonomy,word - each word is a keyword for that level"""
    data = []
    try:
        df = pd.read_csv(path, encoding='latin1')
        for _, row in df.iterrows():
            lvl = standardize_level(str(row['taxonomy']))
            word = str(row['word']).strip()
            if lvl and word and len(word) > 2:
                # Create question-like prompts from keywords
                prompts = [
                    f"{word} the main concepts.",
                    f"Can you {word.lower()} this topic?",
                    f"Please {word.lower()} the key points."
                ]
                for p in prompts:
                    data.append({'question': p, 'level': lvl})
        print(f"  ✓ blooms_taxonomy_format_2.csv: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ blooms_taxonomy_format_2.csv: {e}")
    return data

def process_format_3(path):
    """Format: taxonomy,word (comma-separated keywords)"""
    data = []
    try:
        df = pd.read_csv(path, encoding='latin1')
        for _, row in df.iterrows():
            lvl = standardize_level(str(row['taxonomy']))
            words = str(row['word']).split(',')
            for word in words:
                word = word.strip()
                if lvl and word and len(word) > 2:
                    prompts = [
                        f"{word} the main ideas.",
                        f"How would you {word.lower()} this?",
                    ]
                    for p in prompts:
                        data.append({'question': p, 'level': lvl})
        print(f"  ✓ blooms_taxonomy_format_3.csv: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ blooms_taxonomy_format_3.csv: {e}")
    return data

def process_keywords(path):
    """Format: columns are level names, rows are keywords"""
    data = []
    try:
        df = pd.read_csv(path, encoding='latin1')
        for col in df.columns:
            lvl = standardize_level(col)
            if lvl:
                for word in df[col].dropna():
                    word = str(word).strip()
                    if word and len(word) > 2:
                        prompts = [
                            f"{word} the concept.",
                            f"Can you {word.lower()} this problem?",
                        ]
                        for p in prompts:
                            data.append({'question': p, 'level': lvl})
        print(f"  ✓ Blooms_Taxonomy_keywords.csv: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ Blooms_Taxonomy_keywords.csv: {e}")
    return data

def process_format_4(path):
    """Format: columns are level names with keywords"""
    data = []
    try:
        df = pd.read_csv(path, encoding='latin1')
        for col in df.columns:
            lvl = standardize_level(col)
            if lvl:
                for word in df[col].dropna():
                    word = str(word).strip()
                    if word and len(word) > 2:
                        data.append({'question': f"{word} the material.", 'level': lvl})
        print(f"  ✓ blooms_taxonomy_format_4.csv: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ blooms_taxonomy_format_4.csv: {e}")
    return data

def process_corpus(path):
    """Format: text,label - labels are like __label__1, __label__2, etc."""
    data = []
    try:
        df = pd.read_csv(path, encoding='latin1', nrows=2000)  # Limit for speed
        label_map = {
            '__label__1': 'Remember',
            '__label__2': 'Understand', 
            '__label__3': 'Apply',
            '__label__4': 'Analyze',
            '__label__5': 'Evaluate',
            '__label__6': 'Create'
        }
        for _, row in df.iterrows():
            label = str(row.get('label', '')).strip()
            text = str(row.get('text', '')).strip()
            lvl = label_map.get(label)
            if lvl and text and len(text) > 20:
                # Truncate very long texts
                if len(text) > 200:
                    text = text[:200] + "..."
                data.append({'question': text, 'level': lvl})
        print(f"  ✓ corpus.csv: {len(data)} samples (limited to 2000 rows)")
    except Exception as e:
        print(f"  ✗ corpus.csv: {e}")
    return data

def process_data_question_levels(path):
    """Format: taxonomy,question"""
    data = []
    try:
        df = pd.read_csv(path, encoding='latin1')
        for _, row in df.iterrows():
            lvl = standardize_level(str(row['taxonomy']))
            txt = str(row['question']).strip()
            if lvl and txt and len(txt) > 10:
                data.append({'question': txt, 'level': lvl})
        print(f"  ✓ data_question_levels.csv: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ data_question_levels.csv: {e}")
    return data

def process_dataEx(path):
    """Format: tutorial_sheet, tutorial_question, ..., Blooms_level_P"""
    data = []
    try:
        df = pd.read_csv(path, encoding='latin1')
        for _, row in df.iterrows():
            lvl = standardize_level(str(row.get('Blooms_level_P', '')))
            txt = str(row.get('tutorial_question', '')).strip()
            if lvl and txt and len(txt) > 10:
                data.append({'question': txt, 'level': lvl})
        print(f"  ✓ {os.path.basename(path)}: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ {os.path.basename(path)}: {e}")
    return data

def process_excel(path):
    """Process Excel file"""
    data = []
    try:
        df = pd.read_excel(path)
        for _, row in df.iterrows():
            lvl = standardize_level(str(row.get('Blooms_level_P', '')))
            txt = str(row.get('tutorial_question', '')).strip()
            if lvl and txt and len(txt) > 10:
                data.append({'question': txt, 'level': lvl})
        print(f"  ✓ dataEx.xlsx: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ dataEx.xlsx: {e}")
    return data

def merge_all():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'data', 'dataset')
    all_data = []
    
    print("="*60)
    print("EXTRACTING DATA FROM ALL FILES")
    print("="*60)
    
    # Process each file with its specific format
    all_data.extend(process_format_2(os.path.join(dataset_dir, 'blooms_taxonomy_format_2.csv')))
    all_data.extend(process_format_3(os.path.join(dataset_dir, 'blooms_taxonomy_format_3.csv')))
    all_data.extend(process_keywords(os.path.join(dataset_dir, 'Blooms_Taxonomy_keywords.csv')))
    all_data.extend(process_format_4(os.path.join(dataset_dir, 'blooms_taxonomy_format_4.csv')))
    all_data.extend(process_corpus(os.path.join(dataset_dir, 'corpus.csv')))
    all_data.extend(process_data_question_levels(os.path.join(dataset_dir, 'data_question_levels.csv')))
    all_data.extend(process_dataEx(os.path.join(dataset_dir, 'dataEx.csv')))
    all_data.extend(process_dataEx(os.path.join(dataset_dir, 'dataExCsv.csv')))
    all_data.extend(process_excel(os.path.join(dataset_dir, 'dataEx.xlsx')))
    
    # Add base questions to ensure coverage
    base_questions = [
        ("What is the capital of France?", "Remember"),
        ("Define the term photosynthesis.", "Remember"),
        ("Explain how the water cycle works.", "Understand"),
        ("Compare photosynthesis and cellular respiration.", "Understand"),
        ("Calculate the area of a triangle.", "Apply"),
        ("Use the Pythagorean theorem.", "Apply"),
        ("What are the parts of a computer?", "Analyze"),
        ("Compare mitosis and meiosis.", "Analyze"),
        ("Assess the effectiveness of vaccines.", "Evaluate"),
        ("Critique this argument.", "Evaluate"),
        ("Design an experiment to test gravity.", "Create"),
        ("Develop a new solution to reduce waste.", "Create"),
    ]
    for q, l in base_questions:
        all_data.append({'question': q, 'level': l})
    
    df_final = pd.DataFrame(all_data)
    df_final = df_final.dropna()
    df_final = df_final.drop_duplicates(subset=['question'])
    
    print(f"\n{'='*60}")
    print(f"TOTAL UNIQUE SAMPLES: {len(df_final)}")
    print("="*60)
    print(df_final['level'].value_counts().to_string())
    
    output_path = os.path.join(current_dir, 'data', 'raw', 'bloom_questions.csv')
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    merge_all()
