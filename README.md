# Structural Signals of Hallucination in LLM Outputs

This project studies whether lightweight linguistic structure signals can help distinguish truthful model responses from hallucinated ones. Instead of relying on only one feature, the analysis now uses a 10-feature syntactic and stylistic fingerprint built from spaCy parses, then compares truthful vs hallucinated outputs with statistical tests and simple classifiers.

The current workflow uses locally hosted Ollama models and benchmark-backed prompts rather than the older self-generated prompt pipeline.

## Current Models

- `Llama-3.2-3B` via Ollama model `llama3.2`
- `Qwen-2.5-7B` via Ollama model `qwen2.5:7b`
- `Gemma-2-2B` via Ollama model `gemma2:2b`

## Dataset

The active prompt file is [`benchmark_data.json`](/Users/keshavnanda/NLP%20research/benchmark_data.json).

It contains two prompt lists:

- `truth_prompts`: answerable benchmark questions from `TruthfulQA` (`truthful_qa`, `generation`, `validation`)
- `hallucination_prompts`: hallucination-trigger questions from `HaluEval` (`flowaicom/HaluEval`, `test`, `FAIL` rows)

This replaced the older `research_data.json` workflow, which used self-generated prompts.

## Main Files

- [`ollama_models.py`](/Users/keshavnanda/NLP%20research/ollama_models.py): sends prompts to locally running Ollama models and writes their responses into `final_cloud_results.json`
- [`final_cloud_results.json`](/Users/keshavnanda/NLP%20research/final_cloud_results.json): stores the generated responses for each model under `truth` and `lie`
- [`analyze_results.py`](/Users/keshavnanda/NLP%20research/analyze_results.py): computes structural features, filters refusal-style responses, runs statistical tests, classifier comparisons, PCA plots, and feature-importance plots
- [`pending work.md`](/Users/keshavnanda/NLP%20research/pending%20work.md): research gaps and next-step notes

## Response JSON Format

`final_cloud_results.json` is expected to look like this:

```json
{
  "Model-Name": {
    "truth": ["response1", "response2"],
    "lie": ["response1", "response2"]
  }
}
```

`truth` means responses to answerable prompts.  
`lie` means responses to hallucination-trigger prompts.

## Analysis Pipeline

[`analyze_results.py`](/Users/keshavnanda/NLP%20research/analyze_results.py) currently extracts these 10 features:

1. tree depth
2. branching
3. modifier density
4. passive count
5. modal / hedge count
6. subordinate clause density
7. negation count
8. entity density
9. average sentence length
10. coordination density

The script also:

- filters refusal-style answers
- stores `response_length` as a baseline feature
- runs Welch’s t-tests per metric
- runs logistic regression with ROC AUC
- compares against a baseline using response length only
- reports a random baseline of `0.5000`
- prints improvement over the response-length baseline
- fits a random forest for feature comparison
- saves histogram grids, PCA plots, and feature-importance charts

## Setup

```bash
cd "/Users/keshavnanda/NLP research"
python3 -m venv nlp_env
source nlp_env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.2
ollama pull qwen2.5:7b
ollama pull gemma2:2b
```

If you are using conda instead of `venv`, activate your conda environment first, install the Python dependencies from `requirements.txt`, and make sure the Ollama models above are pulled locally.

## Running the Project

Make sure Ollama is installed and the three local models are available:

```bash
ollama list
```

You should see:

```text
llama3.2
qwen2.5:7b
gemma2:2b
```

Then run:

```bash
cd "/Users/keshavnanda/NLP research"
python ollama_models.py
python analyze_results.py
```

## Generated Outputs

Running the analysis recreates image outputs like:

- `graph_<model>.png`
- `pca_<model>.png`
- `feature_importance_<model>.png`
- combined versions such as `graph_combined.png`

These are generated from the current contents of `final_cloud_results.json`.
