"""
batch_scorebugs_openai.py
Creates a Batch-API job that extracts scoreboard data
from ~5 000 images in one asynchronous run.
"""

import os, glob, json, base64, time
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()  # requires openai-python ≥1.12

BASE_DIR = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebug_chunks"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TASKS_DIR = os.path.join(BASE_DIR, "tasks")
MODEL = "gpt-4.1"
BATCH_WINDOW = "24h"  # only 24h or 48h allowed

PROMPT_TEXT = """\
Return a JSON object with exactly these keys:
• player1_name       (string)
• player2_name       (string)
• server             ("player1" | "player2")
• set_number : int 1–5 (the set currently being played; to determine this:
    1. Count the set columns: Each set usually has its own column for games won by each player.
    2. The current set is the rightmost column where the number of games won by at least one player is less than the number required to win a set (usually 6, unless a tiebreak is in progress).
    3. If the current game scores indicate a tiebreak (e.g., both players have 6 games and the points are numeric like 5–4), that column is the current set.
    4. If only one set column exists, set_number is 1.
    5. If all sets are complete, use the last set column as the current set.
    6. If you cannot determine the current set, set set_number to null.
    Example: If three set columns are visible, and the third column shows games less than 6 for both players, set_number is 3. If only one set column is visible, set_number is 1.)
• current_set        (object):
    - games : { player1:int 0–6, player2:int 0–6 } (the number of games won by each player in the current set)
• current_game       (object):
    - player1_points : "0","15","30","40","Ad" (the current point score for player 1 in the ongoing game; if unavailable, use '0')
    - player2_points : "0","15","30","40","Ad" (the current point score for player 2 in the ongoing game; if unavailable, use '0')

Guidelines:
• The 'set_number' is the number of completed sets plus one, or the number of set columns shown (excluding the game score column, if present). If set_number cannot be determined, set to null.
• The 'current_set' only contains the number of games won by each player in the current set.
• The 'current_game' refers to the ongoing game within the current set. Only include the current point scores for both players in this game. \n• In the scoreboard, the last column is usually the game score (current points in the game), and the second to last column is typically the current set score.\n• The set number must be between 1 and 5. If outside this range or unreadable, set to null.\n• Game scores must be between 1 and 6 for both players. If outside this range or unreadable, set to null.\n• Determine the server exclusively from the yellow indicator; if ambiguous, set to null.\n• Enforce these constraints and output STRICT JSON only.\n"""

def encode_image(path:str)->str:
    ext  = os.path.splitext(path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg","jpeg") else "png"
    with open(path,"rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"

def run_batch_for_chunk(chunk_num):
    chunk_str = f"{chunk_num:03d}"
    input_dir = os.path.join(BASE_DIR, f"chunk_{chunk_str}")
    tasks_file = os.path.join(TASKS_DIR, f"tasks_scorebug_chunk_{chunk_str}.jsonl")
    results_file = os.path.join(RESULTS_DIR, f"results_scorebug_chunk_{chunk_str}.jsonl")

    # 1. Build tasks list
    tasks = []
    for img in sorted(glob.glob(os.path.join(input_dir,"*.jp*g"))+glob.glob(os.path.join(input_dir,"*.png"))):
        tasks.append({
            "custom_id": os.path.basename(img),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "temperature": 0.7,
                "top_p": 1.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system",
                     "content": "You are ChatGPT, a large language model trained by OpenAI. Respond in markdown. You are a vision model that extracts tennis scoreboard data and returns STRICT JSON."},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": PROMPT_TEXT},
                         {"type": "image_url", "image_url": {"url": encode_image(img)}}
                     ]}
                ]
            }
        })
    os.makedirs(os.path.dirname(tasks_file), exist_ok=True)
    with open(tasks_file,"w") as f:
        for t in tasks:
            f.write(json.dumps(t)+"\n")

    # 3. Upload tasks file
    file_obj = client.files.create(
        file=open(tasks_file,"rb"),
        purpose="batch"
    )

    # 4. Kick off batch job
    batch = client.batches.create(
        input_file_id = file_obj.id,
        endpoint = "/v1/chat/completions",
        completion_window = BATCH_WINDOW
    )
    print(f"Chunk {chunk_str}: Batch ID: {batch.id}")

    # 5. Poll status until 'completed'
    while True:
        batch = client.batches.retrieve(batch.id)
        #print(f"Chunk {chunk_str}: {batch.status} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if batch.status in ("completed","failed","expired","canceled"):
            break
        time.sleep(30)

    if batch.status=="completed":
        out_file = client.files.content(batch.output_file_id).content
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file,"wb") as f:
            f.write(out_file)
        print(f"✓ Chunk {chunk_str}: Results written to {results_file}")
    else:
        print(f"⚠ Chunk {chunk_str}: Batch ended with status: {batch.status}")

if __name__ == "__main__":
    while True:
        for chunk_num in tqdm(range(1, 108), desc="Processing Chunks"):
            chunk_str = f"{chunk_num:03d}"
            results_file = os.path.join(RESULTS_DIR, f"results_scorebug_chunk_{chunk_str}.jsonl")
            if os.path.exists(results_file):
                print(f"✓ Chunk {chunk_str}: Results already exist, skipping.")
                continue
            try:
                run_batch_for_chunk(chunk_num)
            except Exception as e:
                print(f"Error processing chunk {chunk_num:03d}: {e}")
                break‹›