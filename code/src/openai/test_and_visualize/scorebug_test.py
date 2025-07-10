#!/usr/bin/env python3
import os
import glob
import random
import base64
import json
from openai import OpenAI

# ---- Configuration ----
INPUT_DIR   = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_test_onethirdres"         # folder with your images
NUM_SAMPLES = 5                   # how many random images to test
MODEL       = "gpt-4.1-mini"        # or another vision-capable model
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
• The 'current_game' refers to the ongoing game within the current set. Only include the current point scores for both players in this game. 
• In the scoreboard, the last column is usually the game score (current points in the game), and the second to last column is typically the current set score.
• The set number must be between 1 and 5. If outside this range or unreadable, set to null.
• Game scores must be between 1 and 6 for both players. If outside this range or unreadable, set to null.
• Determine the server exclusively from the yellow indicator; if ambiguous, set to null.
• Enforce these constraints and output STRICT JSON only.
"""

# ---- Helpers ----
def pick_random_images(folder, n):
    """Return up to n random image paths from folder."""
    patterns = [os.path.join(folder, "*.jpg"),
                os.path.join(folder, "*.jpeg"),
                os.path.join(folder, "*.png")]
    all_imgs = []
    for p in patterns:
        all_imgs.extend(glob.glob(p))
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {folder}")
    return random.sample(all_imgs, min(n, len(all_imgs)))

def encode_image_to_dataurl(path):
    """Read file and return a data URL string."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg","jpeg") else "png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"

# ---- Main ----
def main():
    # initialize client
    client = OpenAI()
    
    # pick images
    random.seed(42)
    images = pick_random_images(INPUT_DIR, NUM_SAMPLES)
    # Save images used
    output_dir = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_test_onethirdres"  
    os.makedirs(output_dir, exist_ok=True)
    images_used_path = os.path.join(output_dir, 'images_used.txt')
    with open(images_used_path, 'w') as f:
        for img in images:
            f.write(f"{img}\n")
    print(f"Images used written to: {images_used_path}")
    print(f"Selected {len(images)} images for testing:\n" +
          "\n".join(os.path.basename(p) for p in images) + "\n")

    # process each image
    for img_path in images:
        print(f"🖼️  Processing {os.path.basename(img_path)}...")
        data_url = encode_image_to_dataurl(img_path)
        
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.7,  # closer to web UI default
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "You are ChatGPT, a large language model trained by OpenAI. Respond in markdown. You are a vision model that extracts tennis scoreboard data and returns STRICT JSON."},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": PROMPT_TEXT},
                         {"type": "image_url", "image_url": {"url": data_url}}
                     ]}
                ]
            )
            result = resp.choices[0].message.content

            # --- Cost Calculation ---
            usage = getattr(resp, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                # Adjust these costs for your model if needed
                input_token_cost = 0.02 / 1000  # $0.01 per 1K input tokens
                output_token_cost = 0.08 / 1000 # $0.03 per 1K output tokens
                image_cost = 0.01               # $0.01 per image (gpt-4-vision-preview default)
                cost = (prompt_tokens * input_token_cost) + (completion_tokens * output_token_cost) + image_cost
                print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Estimated cost: ${cost:.4f}")
            else:
                print("[Warning] No usage/cost info available for this response.")

        except Exception as e:
            result = {
                "error": str(e)
            }
        
        # Save result as JSON
        def sanitize_filename(name):
            return ''.join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in name)
        img_base = os.path.basename(img_path)
        json_filename = sanitize_filename(os.path.splitext(img_base)[0]) + '.json'
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as jf:
            if isinstance(result, str):
                # Try to pretty print JSON string
                try:
                    jf.write(json.dumps(json.loads(result), indent=2))
                except Exception:
                    jf.write(result)
            else:
                jf.write(json.dumps(result, indent=2))
        print(f"Output written to: {json_path}")
        print("-" * 60)

if __name__ == "__main__":
    main()