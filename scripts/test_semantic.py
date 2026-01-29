import sys
sys.path.append('src')
from preprocessing import TextPreprocessor, load_config

config = load_config('config/config.yaml')
prep = TextPreprocessor(**config['preprocessing'])

text = "What is the capital of France?"
cleaned = prep.clean_text(text)
print(f"Original: {text}")
print(f"Enriched: {cleaned}")
