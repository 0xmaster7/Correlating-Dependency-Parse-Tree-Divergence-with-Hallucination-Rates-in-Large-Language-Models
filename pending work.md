# Pending Work

## Critical gaps before submission

1. **Only 2 models with usable data**  
   Llama and Qwen. 3 models got wiped by refusal filter. Reviewers will immediately say "findings don't generalize." You need at least 3-4 models with actual results. Fix the prompts for Mistral/Phi so they don't refuse, or swap in different models like GPT-3.5, Gemma-2, or Falcon.

2. **Dataset is self-generated**  
   Your `research_data.json` was LLM-generated. Reviewers will question validity. You should either validate it manually (sample 100 prompts, human-annotate) or supplement with an established benchmark like TruthfulQA or HaluEval, even partially.

3. **AUC of 0.56 needs context**  
   You need to compare against a baseline. Even a simple word-count or perplexity baseline. If your syntactic features match or beat that, it's meaningful. Right now there's nothing to compare against.

4. **No qualitative error analysis**  
   Pick 20-30 cases where the classifier was wrong. Why did it fail? This section is what separates a rejected paper from an accepted one.

## Note to self

Using conda env.
