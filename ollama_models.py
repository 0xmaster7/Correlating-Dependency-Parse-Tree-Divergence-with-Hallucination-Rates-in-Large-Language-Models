import json
import time

import requests
from tqdm import tqdm


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODELS = {
    "Llama-3.2-3B": "llama3.2",
    "Qwen-2.5-7B": "qwen2.5:7b",
    "Gemma-2-2B": "gemma2:2b",
}
TARGET_SAMPLES_PER_CLASS = 100


def load_prompt_data():
    print("Loading data...")
    try:
        with open("benchmark_data.json", "r") as file:
            data = json.load(file)
        print("Using benchmark_data.json")
        return data
    except FileNotFoundError:
        try:
            with open("research_data.json", "r") as file:
                data = json.load(file)
            print("Using research_data.json")
            return data
        except FileNotFoundError:
            raise SystemExit("Error: neither 'benchmark_data.json' nor 'research_data.json' was found.")


def load_results():
    return {}


def save_final_results(results):
    with open("final_cloud_results.json", "w") as file:
        json.dump(results, file)


def ensure_ollama_is_running():
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            "Could not reach Ollama at http://127.0.0.1:11434.\n"
            "Start Ollama first, then rerun this script.\n"
            f"Underlying error: {exc}"
        )


def query_ollama_robust(model_id, prompt):
    user_prompt = (
        f"Question: {prompt}\n"
        "Answer with a detailed explanation. Do not mention that you are an AI assistant."
    )

    payload = {
        "model": model_id,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200,
        },
    }

    while True:
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
            body = response.json()
            answer = body.get("response", "").strip()
            return answer if answer else None
        except requests.Timeout:
            print("    Request timed out. Retrying in 10s...")
            time.sleep(10)
        except requests.RequestException as exc:
            error_text = str(exc).lower()
            if "404" in error_text:
                print(f"    Model {model_id} is not available in Ollama. Skipping...")
                return None
            print(f"    Ollama error: {exc}")
            retry = input("    Press Enter to retry, or type 'skip' to skip this prompt: ").strip().lower()
            if retry == "skip":
                return None


def run_model_on_split(results, friendly_name, model_id, split_name, prompts):
    current_done = len(results[friendly_name][split_name])
    if current_done >= TARGET_SAMPLES_PER_CLASS:
        return

    print(f"\n{friendly_name}: Running {split_name} set ({TARGET_SAMPLES_PER_CLASS} prompts)...")
    to_process = prompts[current_done:TARGET_SAMPLES_PER_CLASS]

    for prompt in tqdm(to_process):
        answer = query_ollama_robust(model_id, prompt)

        if answer:
            results[friendly_name][split_name].append(answer)


def main():
    ensure_ollama_is_running()
    data = load_prompt_data()
    results = load_results()

    for friendly_name, model_id in MODELS.items():
        if friendly_name not in results:
            results[friendly_name] = {"truth": [], "lie": []}

        run_model_on_split(results, friendly_name, model_id, "truth", data["truth_prompts"])
        run_model_on_split(results, friendly_name, model_id, "lie", data["hallucination_prompts"])

    save_final_results(results)
    print("\nDone. All models processed.")


if __name__ == "__main__":
    main()
