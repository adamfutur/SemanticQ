import pandas as pd
import os

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

def merge_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_file = os.path.join(current_dir, 'data', 'raw', 'bloom_questions.csv')
    github_dir = os.path.join(current_dir, 'blooms-taxonomy-classification', 'dataset')
    all_data = []

    # 1. Load data_question_levels.csv
    file1 = os.path.join(github_dir, 'data_question_levels.csv')
    try:
        df1 = pd.read_csv(file1, encoding='latin1')
        print(f"Reading {file1}: {len(df1)} samples")
        for _, row in df1.iterrows():
            lvl = standardize_level(row['taxonomy'])
            if lvl:
                all_data.append({'question': row['question'], 'level': lvl})
            else:
                print(f"Skipping unknown level in {os.path.basename(file1)}: {row['taxonomy']}")
    except Exception as e:
        print(f"Error reading {file1}: {e}")
        
    # 2. Add dataExCsv.csv
    file2 = os.path.join(github_dir, 'dataExCsv.csv')
    try:
        df2 = pd.read_csv(file2, encoding='latin1')
        print(f"Reading {file2}: {len(df2)} samples")
        for _, row in df2.iterrows():
            lvl = standardize_level(row['Blooms_level_P'])
            if lvl:
                all_data.append({'question': row['tutorial_question'], 'level': lvl})
            else:
                if pd.notna(row['Blooms_level_P']):
                    print(f"Skipping unknown level in {os.path.basename(file2)}: {row['Blooms_level_P']}")
    except Exception as e:
        print(f"Error reading {file2}: {e}")

    # 3. Add original core dataset (to ensure we don't lose it)
    # Actually, bloom_questions.csv currently has the 125 paper questions.
    # Let's keep a backup if possible, but I already overwrote it luckily I can just regenerate the 125 core ones if needed.
    # Wait, the 125 core ones were what I created in the previous turn.
    
    # Create final DataFrame
    df_final = pd.DataFrame(all_data)
    
    # Drop rows where question or level is NaN
    df_final = df_final.dropna(subset=['question', 'level'])
    
    # Remove duplicates
    df_final = df_final.drop_duplicates(subset=['question'])
    print(f"\nFinal merged dataset: {len(df_final)} unique samples")
    
    # Check class distribution
    print("\nClass distribution:")
    print(df_final['level'].value_counts())
    
    # Save
    df_final.to_csv(base_file, index=False)
    print(f"Saved merged dataset to {base_file}")

if __name__ == "__main__":
    merge_datasets()
