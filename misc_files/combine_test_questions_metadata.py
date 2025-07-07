import json

# File paths
TEST_QUESTIONS_PATH = 'TEST Questions 1750975249515.json'
METADATA_PATH = 'metadata.jsonl'  # Use the standard JSONL file
OUTPUT_PATH = 'TEST Questions Metadata.json'

# Load test questions
with open(TEST_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    test_questions = json.load(f)

# Load metadata (JSONL)
metadata_by_id = {}
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            metadata_by_id[entry['task_id']] = entry

# Merge
combined = []
for q in test_questions:
    task_id = q.get('task_id')
    meta = metadata_by_id.get(task_id, {})
    merged = dict(q)  # start with question fields
    # Add reference answer and annotator metadata if available
    if meta:
        merged['reference_answer'] = meta.get('Final answer')
        merged['annotator_metadata'] = meta.get('Annotator Metadata')
    else:
        merged['reference_answer'] = None
        merged['annotator_metadata'] = None
    combined.append(merged)

# Write output
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"Combined file written to {OUTPUT_PATH} with {len(combined)} questions.") 