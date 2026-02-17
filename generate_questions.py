import json
import random
from datasets import load_dataset

print("Downloading SQuAD 2.0 dataset... (this might take a minute)")
# Load SQuAD 2.0 (The gold standard for unanswerable questions)
dataset = load_dataset("squad_v2", split="validation")

# 1. Filter for TRUTH (Questions with answers)
truths = [
    item["question"] 
    for item in dataset 
    if len(item["answers"]["text"]) > 0
]

# 2. Filter for LIES (Questions designed to be impossible)
# These look like normal questions but rely on false premises or missing info.
lies = [
    item["question"] 
    for item in dataset 
    if len(item["answers"]["text"]) == 0
]

# Shuffle and pick 1,000 of each
random.shuffle(truths)
random.shuffle(lies)
selected_truths = truths[:1000]
selected_lies = lies[:1000]

# Save to a single file for Phase 2
output_data = {
    "truth_prompts": selected_truths,
    "hallucination_prompts": selected_lies
}

with open("research_data.json", "w") as f:
    json.dump(output_data, f, indent=4)

print(f"✅ Success! Saved 1,000 Truths and 1,000 Lie Triggers to 'research_data.json'")
print("Example Lie Trigger:", selected_lies[0])