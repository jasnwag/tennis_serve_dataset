from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List recent batches
batches = client.batches.list(limit=5)  # Get last 5 batches
print("\nRecent batches details:")
for batch in batches:
    print(f"\nBatch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    print(f"Request counts: {batch.request_counts}")
    
    # Get the input file to check the model
    if batch.input_file_id:
        try:
            input_content = client.files.content(batch.input_file_id).content
            input_data = input_content.decode('utf-8')
            # Print first line of input to see the model
            print("First line of input:")
            print(input_data.split('\n')[0])
        except Exception as e:
            print(f"Could not read input file: {e}")
    print("-" * 50) 