import stanza
import json
from tqdm import tqdm

print("Starting Download")
stanza.download('la')
print("Download finished")

with open('D:\\EDR\\data\\parsed_results_with_id_and_dating.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

nlp_gpu = stanza.Pipeline('la', processors='tokenize,pos', use_gpu=True)

results = []
for record in tqdm(data, desc="Processing records"):
    parsed_field = record['parsed_field']
    record_number = record['record_number']
    
    cleaned_text = parsed_field.replace('<', '').replace('>', '')
    
    doc = nlp_gpu(cleaned_text)
    simple_tags = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.lower() == "unk":
                pos_tag = "X"
                word_text = "<unk>"
            else:
                pos_tag = word.pos
                word_text = word.text
            simple_tags.append((word_text, pos_tag))
    
    result = {
        'record_number': record_number,
        'parsed_field': parsed_field,
        'pos_tags': simple_tags
    }
    results.append(result)

with open('processed_results.json', 'w', encoding='utf-8') as output_file:
    json.dump(results, output_file, indent=2, ensure_ascii=False)

print(f"Processed {len(results)} records and saved to 'processed_results.json'")

total_words = sum(len(result['pos_tags']) for result in results)
print(f"Total words processed: {total_words}")
