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

#new script that takes into accoubt 4 dimensions instead of jus tvertical height 
"""
import json
import spacy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Load the "Brain" (Grammar Parser)
nlp = spacy.load("en_core_web_sm")

def get_structural_metrics(text):
    # Calculates a multi-dimensional syntactic fingerprint for structural divergence.
    if not text or len(text) < 10: return None 
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
    
    # 2. Average Branching Factor
    children_counts = [len(list(token.children)) for token in doc]
    avg_branching = sum(children_counts) / len(children_counts) if children_counts else 0
    
    # 3. Modifier Density 
    modifiers = [token for token in doc if token.pos_ in ['ADJ', 'ADV']]
    modifier_density = len(modifiers) / len(doc) if len(doc) > 0 else 0
    
    # 4. Passive Voice Count
    passive_count = len([token for token in doc if token.dep_ in ['nsubjpass', 'auxpass']])
    
    return {
        "depth": max_depth,
        "branching": avg_branching,
        "modifier_density": modifier_density,
        "passive_count": passive_count
    }

# --- EXECUTION ---
print("📊 Loading your experiment data...")
with open("final_cloud_results.json", "r") as f:
    data = json.load(f)

final_stats = []
# The 4 metrics we are testing
metrics_list = ["depth", "branching", "modifier_density", "passive_count"]

for model_name, responses in data.items():
    print(f"  > Analyzing {model_name}...")
    
    # 1. Parse all texts and drop the None values
    truth_fingerprints = [get_structural_metrics(t) for t in responses["truth"]]
    truth_fingerprints = [f for f in truth_fingerprints if f]
    
    lie_fingerprints = [get_structural_metrics(t) for t in responses["lie"]]
    lie_fingerprints = [f for f in lie_fingerprints if f]
    
    # 2. Set up a 2x2 grid for the graphs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Syntactic Fingerprint: {model_name}", fontsize=16)
    axes = axes.flatten()
    
    # 3. Run the stats and plot for each metric individually
    for i, metric in enumerate(metrics_list):
        truth_vals = [f[metric] for f in truth_fingerprints]
        lie_vals = [f[metric] for f in lie_fingerprints]
        
        # Calculate the T-Test (Welch's t-test assuming unequal variance)
        t_stat, p_val = stats.ttest_ind(truth_vals, lie_vals, equal_var=False)
        
        # Store results for the table
        final_stats.append({
            "Model": model_name,
            "Metric": metric.replace("_", " ").title(),
            "Avg Truth": round(np.mean(truth_vals), 3),
            "Avg Lie": round(np.mean(lie_vals), 3),
            "Difference": round(np.mean(lie_vals) - np.mean(truth_vals), 3),
            "Significant?": "✅ YES" if p_val < 0.05 else "❌ NO",
            "P-Value": "{:.5f}".format(p_val)
        })
        
        # Plot the histograms in the correct subplot
        axes[i].hist(truth_vals, bins=20, alpha=0.5, label='Truth', color='green', density=True)
        axes[i].hist(lie_vals, bins=20, alpha=0.5, label='Hallucination', color='red', density=True)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"graph_{model_name.replace('/','_')}_4D.png")
    plt.close()

# 4. Show the Table
df = pd.DataFrame(final_stats)
print("\n" + "="*80)
print("FINAL SCIENTIFIC RESULTS (4-DIMENSIONAL)")
print("="*80)
print(df.to_string(index=False))
print("\nGraphs saved to your folder.")
"""