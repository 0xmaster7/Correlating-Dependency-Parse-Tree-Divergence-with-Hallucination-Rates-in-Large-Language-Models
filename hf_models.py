import json
import time
from tqdm import tqdm
from huggingface_hub import InferenceClient

# --- CONFIGURATION ---
CURRENT_TOKEN = "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" # <--- PASTE YOUR WORKING TOKEN

MODELS = {
    # ✅ Llama-3.2-3B (Keep this, you already have data for it)
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct", 
    
    # 🚀 NEW: Qwen 2.5 7B (The current "King" of open source. Very reliable API)
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct"
    
    
}
 # "Gemma-2-9B": "google/gemma-2-9b-it"
# ---------------------

def get_client(token):
    return InferenceClient(api_key=token)

def query_huggingface_robust(client, model_id, prompt, current_token):
    messages = [{"role": "user", "content": f"Question: {prompt}\nAnswer (provide detailed explanation):"}]
    
    while True: 
        try:
            response = client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip(), current_token

        except Exception as e:
            error_msg = str(e).lower()
            
            # CASE 1: QUOTA DEAD (402)
            if "402" in error_msg or "payment" in error_msg:
                print(f"\n\n🚨 QUOTA DEAD for token: {current_token[:10]}...")
                new_key = input("Paste New Token (or type 'skip'): ").strip()
                if new_key.lower() == "skip": return None, current_token
                
                current_token = new_key
                client = get_client(current_token)
                print("🔄 Key updated! Retrying...")
                continue 

            # CASE 2: MODEL BROKEN / NOT SUPPORTED (The Zephyr Error)
            elif "model_not_supported" in error_msg or "404" in error_msg or "400" in error_msg:
                print(f"    ❌ Model {model_id} is unavailable on Free Tier. Skipping...")
                return None, current_token 
            
            # CASE 3: Rate Limit
            elif "429" in error_msg:
                print(f"    🛑 Rate limit. Sleeping 30s...")
                time.sleep(30)
                continue
            
            # CASE 4: Loading
            elif "loading" in error_msg or "503" in error_msg:
                print(f"    ❄️ Model loading... waiting 20s")
                time.sleep(20)
                continue
                
            else:
                print(f"    ⚠️ Unknown error: {e}")
                return None, current_token

# --- MAIN EXECUTION ---

print("Loading data...")
try:
    with open("research_data.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Error: 'research_data.json' not found.")
    exit()

# Load Resume Data
try:
    with open("cloud_results_partial.json", "r") as f:
        results = json.load(f)
    print("🔄 Resuming from last save...")
except FileNotFoundError:
    results = {}

client = get_client(CURRENT_TOKEN)

for friendly_name, model_path in MODELS.items():
    if friendly_name not in results:
        results[friendly_name] = {"truth": [], "lie": []}

    # 1. Run Truths
    current_done = len(results[friendly_name]["truth"])
    if current_done < 200:
        print(f"\n🧪 {friendly_name}: Resuming Truths from {current_done}/200...")
        to_process = data["truth_prompts"][current_done:200]
        
        for i, q in enumerate(tqdm(to_process)):
            ans, CURRENT_TOKEN = query_huggingface_robust(client, model_path, q, CURRENT_TOKEN)
            
            # Update client if token changed
            if client.headers.get("Authorization") != f"Bearer {CURRENT_TOKEN}":
                client = get_client(CURRENT_TOKEN)

            if ans: results[friendly_name]["truth"].append(ans)
            
            if i % 5 == 0:
                with open("cloud_results_partial.json", "w") as f:
                    json.dump(results, f)
                time.sleep(1)

    # 2. Run Lies
    current_done = len(results[friendly_name]["lie"])
    if current_done < 200:
        print(f"\n🧪 {friendly_name}: Resuming Lies from {current_done}/200...")
        to_process = data["hallucination_prompts"][current_done:200]
        
        for i, q in enumerate(tqdm(to_process)):
            ans, CURRENT_TOKEN = query_huggingface_robust(client, model_path, q, CURRENT_TOKEN)
            
            if client.headers.get("Authorization") != f"Bearer {CURRENT_TOKEN}":
                client = get_client(CURRENT_TOKEN)

            if ans: results[friendly_name]["lie"].append(ans)
            
            if i % 5 == 0:
                with open("cloud_results_partial.json", "w") as f:
                    json.dump(results, f)
                time.sleep(1)

# Final Save
with open("final_cloud_results.json", "w") as f:
    json.dump(results, f)

print("\n🎉 DONE! All models processed.")