#nlp1 - venv
import json
import spacy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Load the "Brain" (Grammar Parser)
nlp = spacy.load("en_core_web_sm")

def get_structural_metrics(text):
    """Calculates tree depth and passive voice usage."""
    if not text or len(text) < 10: return None # Skip empty/short answers
    doc = nlp(text)
    
    # 1. Max Tree Depth
    depths = []
    for token in doc:
        current_depth = 0
        head = token
        while head.head != head:
            current_depth += 1
            head = head.head
        depths.append(current_depth)
    max_depth = max(depths) if depths else 0
    
    return max_depth

# --- EXECUTION ---
print("📊 Loading your experiment data...")
with open("final_cloud_results.json", "r") as f:
    data = json.load(f)

final_stats = []

for model_name, responses in data.items():
    print(f"  > Analyzing {model_name}...")
    
    # Calculate scores
    truth_depths = [get_structural_metrics(t) for t in responses["truth"] if get_structural_metrics(t)]
    lie_depths = [get_structural_metrics(t) for t in responses["lie"] if get_structural_metrics(t)]
    
    # Statistical Test (The P-Value)
    t_stat, p_val = stats.ttest_ind(truth_depths, lie_depths, equal_var=False)
    
    # Store results
    final_stats.append({
        "Model": model_name,
        "Avg Truth Depth": round(np.mean(truth_depths), 2),
        "Avg Lie Depth": round(np.mean(lie_depths), 2),
        "Difference": round(np.mean(lie_depths) - np.mean(truth_depths), 2),
        "Significant?": "✅ YES" if p_val < 0.05 else "❌ NO",
        "P-Value": "{:.5f}".format(p_val)
    })

    # Generate Graph
    plt.figure(figsize=(10, 6))
    plt.hist(truth_depths, bins=20, alpha=0.5, label='Truth', color='green', density=True)
    plt.hist(lie_depths, bins=20, alpha=0.5, label='Hallucination', color='red', density=True)
    plt.title(f"Syntactic Fingerprint: {model_name}")
    plt.xlabel("Grammar Tree Depth")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"graph_{model_name.replace('/','_')}.png")
    plt.close()

# Show the Table
df = pd.DataFrame(final_stats)
print("\n" + "="*60)
print("FINAL SCIENTIFIC RESULTS")
print("="*60)
print(df.to_string(index=False))
print("\ngraphs saved to your folder.")