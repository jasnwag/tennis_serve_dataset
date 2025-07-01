import numpy as np
import os

# Path to the .npy file
data_path = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/raw/all_matches_data.npy"
)

# Load the data
try:
    data = np.load(data_path, allow_pickle=True)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

print(f"Type of loaded data: {type(data)}")

# If it's a structured numpy array or object array, try to explore columns/fields
if isinstance(data, np.ndarray):
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    if data.dtype.names:
        print(f"Column names: {data.dtype.names}")
    elif data.dtype == object:
        # Check for 0-dimensional object array
        if data.shape == ():
            obj = data.item()
            print(f"Loaded a single object of type: {type(obj)}")
            if hasattr(obj, 'keys'):
                print(f"Keys: {list(obj.keys())}")
            elif hasattr(obj, 'columns'):
                print(f"Columns: {list(obj.columns)}")
            elif isinstance(obj, (list, tuple)):
                print(f"Object is a {type(obj)} of length {len(obj)}")
                for i, item in enumerate(obj[:5]):
                    print(f"\nElement {i}:")
                    print(f"  Type: {type(item)}")
                    if hasattr(item, 'keys'):
                        print(f"  Keys: {list(item.keys())}")
                    elif hasattr(item, 'columns'):
                        print(f"  Columns: {list(item.columns)}")
                    else:
                        print(f"  Value: {item}")
            else:
                print("Object does not have 'keys', 'columns', or is not a list/tuple.")
        else:
            print("Object array is not 0-dimensional. Further inspection needed.")
    else:
        print("Standard ndarray, no column names.")
else:
    print("Loaded object is not a numpy ndarray.")
