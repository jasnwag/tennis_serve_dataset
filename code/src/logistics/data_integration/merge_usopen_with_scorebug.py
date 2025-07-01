import pandas as pd
import json

# Paths
usopen_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/us_open_data/us_open.csv"
scorebug_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebug_chunks/results/combined_renamed.jsonl"
output_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/usopen_points_with_scorebug_V2.csv"

# Load US Open data
usopen_df = pd.read_csv(usopen_path)

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

# Normalize player1 and player2 in usopen
usopen_df['player1'] = usopen_df['player1'].apply(get_lastname)
usopen_df['player2'] = usopen_df['player2'].apply(get_lastname)

# Load scorebug jsonl as DataFrame, extracting player1, player2, SetNo, custom_id
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
            # Extract nested fields for games and current_game
            games = content_dict.get("games") or content_dict.get("current_set", {}).get("games", {})
            current_game = content_dict.get("current_game", {})
            record = {
                "player1": content_dict.get("player1", content_dict.get("player1_name")),
                "player2": content_dict.get("player2", content_dict.get("player2_name")),
                "SetNo": content_dict.get("SetNo", content_dict.get("set_number")),
                "P1GamesWon": games.get("P1GamesWon", games.get("player1")) if games else None,
                "P2GamesWon": games.get("P2GamesWon", games.get("player2")) if games else None,
                "P1Score": current_game.get("P1Score", current_game.get("player1_points")),
                "P2Score": current_game.get("P2Score", current_game.get("player2_points")),
                "video_name": obj.get("custom_id")
            }
            scorebug_records.append(record)
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

# Normalize player1 and player2 in scorebug
scorebug_df['player1'] = scorebug_df['player1'].apply(get_lastname)
scorebug_df['player2'] = scorebug_df['player2'].apply(get_lastname)

# Only keep scorebug rows with unique (player1, player2, SetNo, ...)
group_cols = ["player1", "player2", "SetNo", "P1GamesWon", "P2GamesWon", "P1Score", "P2Score"]
print(f"Total rows in scorebug_df: {len(scorebug_df)}")
dupes = scorebug_df.duplicated(subset=group_cols, keep=False)
unique_scorebug = scorebug_df[~dupes]
print(f"Rows with unique {group_cols} in scorebug_df: {len(unique_scorebug)}")

# Show examples of dropped duplicates
if dupes.any():
    print("Examples of dropped duplicate rows in scorebug_df:")
    print(scorebug_df[dupes].sample(20, random_state=42))
else:
    print("No duplicates found in scorebug_df on merge columns.")

# Find rows in unique_scorebug with no match in usopen_df
merged = pd.merge(usopen_df, unique_scorebug, on=group_cols, how="right", indicator=True)
no_match = merged[merged['_merge'] == 'right_only']
print(f"Rows in scorebug_df with no match in usopen_df: {len(no_match)}")
if len(no_match) > 0:
    print("Examples of scorebug rows with no match in usopen_df:")
    print(no_match[group_cols + ["video_name"]].head(3))

# Now do the actual inner merge for output
merged = pd.merge(usopen_df, unique_scorebug, on=group_cols, how="inner")
print(f"Rows in final merged file: {len(merged)}")

# Save merged file
merged.to_csv(output_path, index=False)
print(f"Merged file saved as {output_path}")
print(f"Merged file saved as {output_path}")
