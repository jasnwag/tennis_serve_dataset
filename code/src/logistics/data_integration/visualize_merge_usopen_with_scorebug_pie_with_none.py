import matplotlib.pyplot as plt
import pandas as pd
import json

# Use the numbers from your merge script output and recalculate with a 'None' category
scorebug_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebug_chunks/results/combined_renamed.jsonl"
MERGE_COLS = [
    "player1", "player2", "SetNo", "P1GamesWon", "P2GamesWon", "P1Score", "P2Score"
]

def get_lastname(name):
    if pd.isnull(name):
        return None
    s = str(name).strip()
    if not s:
        return None
    parts = s.split()
    if not parts:
        return None
    return parts[-1].lower()

# Parse scorebug jsonl as DataFrame
scorebug_records = []
with open(scorebug_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        if "response" in obj and "body" in obj["response"] and "choices" in obj["response"]["body"]:
            message_content = obj["response"]["body"]["choices"][0]["message"]["content"]
            try:
                content_dict = json.loads(message_content)
            except Exception:
                continue
            games = content_dict.get("games") or content_dict.get("current_set", {}).get("games", {})
            current_game = content_dict.get("current_game", {})
            record = {
                "player1": get_lastname(content_dict.get("player1", content_dict.get("player1_name"))),
                "player2": get_lastname(content_dict.get("player2", content_dict.get("player2_name"))),
                "SetNo": content_dict.get("SetNo", content_dict.get("set_number")),
                "P1GamesWon": games.get("P1GamesWon", games.get("player1")) if games else None,
                "P2GamesWon": games.get("P2GamesWon", games.get("player2")) if games else None,
                "P1Score": current_game.get("P1Score", current_game.get("player1_points")),
                "P2Score": current_game.get("P2Score", current_game.get("player2_points")),
            }
            scorebug_records.append(record)
scorebug_df = pd.DataFrame(scorebug_records)

# Category: any None in merge columns
is_none = scorebug_df[MERGE_COLS].isnull().any(axis=1)
none_count = is_none.sum()
not_none_df = scorebug_df[~is_none]

group_cols = MERGE_COLS
dupes = not_none_df.duplicated(subset=group_cols, keep=False)
unique_scorebug = not_none_df[~dupes]
duplicate_scorebug = not_none_df[dupes]

# Simulate merge counts (estimates)
matched = len(unique_scorebug) - (unique_scorebug.shape[0] - 6694 - 1378)  # Use previous matched estimate
no_match = 1378  # From previous
duplicates = len(duplicate_scorebug)

labels = ['Matched', 'No Match', 'Duplicates', 'Any None']
counts = [matched, no_match, duplicates, none_count]
colors = ['#4F81BD', '#C0504D', '#9BBB59', '#AAAAAA']

plt.figure(figsize=(8,8))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.title("Merge Results", fontsize=16)
plt.tight_layout()
plt.savefig("annotation_merge_usopen_with_scorebug_pie_with_none.png")
plt.close()
print("Saved annotation merge outcome pie chart with 'None' category as annotation_merge_usopen_with_scorebug_pie_with_none.png")
