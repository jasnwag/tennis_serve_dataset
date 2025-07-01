import pandas as pd
import matplotlib.pyplot as plt
import json

# --- CONFIG ---
USOPEN_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/us_open_data/us_open.csv"
SCOREBUG_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebug_chunks/results/combined_renamed.jsonl"
MERGE_COLS = [
    "player1", "player2", "SetNo",
    "P1GamesWon", "P2GamesWon", "P1Score", "P2Score"
]

# --- LOAD US OPEN DATA ---
usopen_df = pd.read_csv(USOPEN_PATH)

# --- LOAD AND PARSE SCOREBUG DATA ---
scorebug_records = []
with open(SCOREBUG_PATH, "r") as f:
    for line in f:
        d = json.loads(line)
        try:
            content = d["response"]["body"]["choices"][0]["message"]["content"]
            content_json = json.loads(content)
            record = {
                "player1": content_json.get("player1_name", "").lower(),
                "player2": content_json.get("player2_name", "").lower(),
                "SetNo": content_json.get("set_number"),
                "P1GamesWon": content_json.get("current_set", {}).get("games", {}).get("player1"),
                "P2GamesWon": content_json.get("current_set", {}).get("games", {}).get("player2"),
                "P1Score": content_json.get("current_game", {}).get("player1_points"),
                "P2Score": content_json.get("current_game", {}).get("player2_points"),
                "video_name": d.get("custom_id", d.get("id"))
            }
            scorebug_records.append(record)
        except Exception as e:
            print(f"Error parsing line: {e}")
scorebug_df = pd.DataFrame(scorebug_records)

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

scorebug_df['player1'] = scorebug_df['player1'].apply(get_lastname)
scorebug_df['player2'] = scorebug_df['player2'].apply(get_lastname)

# Only keep scorebug rows with unique (player1, player2, SetNo, ...)
group_cols = MERGE_COLS
dupes = scorebug_df.duplicated(subset=group_cols, keep=False)
unique_scorebug = scorebug_df[~dupes]
duplicate_scorebug = scorebug_df[dupes]

# Find rows in unique_scorebug with no match in usopen_df
merged = pd.merge(usopen_df, unique_scorebug, on=group_cols, how="right", indicator=True)
no_match = merged[merged['_merge'] == 'right_only']

# Now do the actual inner merge for output
matched = pd.merge(usopen_df, unique_scorebug, on=group_cols, how="inner")

# --- PIE CHART ---
total = len(scorebug_df)
num_no_match = len(no_match)
num_too_many_matches = len(duplicate_scorebug)
num_matched = len(matched)

labels = ['Matched', 'No Match', 'Too Many Matches']
counts = [num_matched, num_no_match, num_too_many_matches]
colors = ['#4F81BD', '#C0504D', '#9BBB59']
plt.figure(figsize=(7,7))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.title("Annotation Merge Outcome (Main Script Logic)", fontsize=16)
plt.tight_layout()
plt.savefig("annotation_merge_usopen_with_scorebug_pie.png")
plt.close()
print("Saved annotation merge outcome pie chart as annotation_merge_usopen_with_scorebug_pie.png")
