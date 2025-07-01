from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fetch specific batch
batch_id = "batch_684b1e5d2c00819082918452cc152eab"
batch = client.batches.retrieve(batch_id)

print(f"\nBatch Details:")
print(f"ID: {batch.id}")
print(f"Status: {batch.status}")
print(f"Created at: {batch.created_at}")
print(f"Request counts: {batch.request_counts}")

# Print all available batch attributes
print("\nAll available batch information:")
for attr in dir(batch):
    if not attr.startswith('_'):  # Skip private attributes
        try:
            value = getattr(batch, attr)
            print(f"{attr}: {value}")
        except:
            pass

# Try to get the results if available
if batch.output_file_id:
    print("\nFetching results...")
    results = client.files.content(batch.output_file_id).content
    print("\nResults:")
    print(results.decode('utf-8'))
else:
    print("\nNo output file available for this batch.")

# Fetch error details if available
if batch.error_file_id:
    print("\nFetching error details...")
    error_content = client.files.content(batch.error_file_id).content
    print("\nError Details:")
    print(error_content.decode('utf-8'))
else:
    print("\nNo error file available for this batch.") 