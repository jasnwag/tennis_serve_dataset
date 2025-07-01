import matplotlib.pyplot as plt

# Use the numbers from your merge script output
# Total rows in scorebug_df: 10663
# Rows with unique ...: 7925
# Rows in scorebug_df with no match in usopen_df: 1378
# Rows in final merged file: 6694
# Duplicates: total - unique = 2738

matched = 6694
no_match = 1378
duplicates = 2738

labels = ['Matched', 'No Match', 'Duplicates']
counts = [matched, no_match, duplicates]
colors = ['#4F81BD', '#C0504D', '#9BBB59']

plt.figure(figsize=(7,7))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.title("Annotation Merge Outcome (Main Script)", fontsize=16)
plt.tight_layout()
plt.savefig("annotation_merge_usopen_with_scorebug_pie_from_counts.png")
plt.close()
print("Saved annotation merge outcome pie chart as annotation_merge_usopen_with_scorebug_pie_from_counts.png")
