import pickle
import sys

def inspect_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            print(f"Successfully loaded {file_path}")
            print(f"Number of items: {len(data)}")
            if data:
                first_key = next(iter(data.keys()))
                print(f"Type of first key: {type(first_key)}")
                print(f"First key: {first_key}")
        else:
            print(f"Loaded data is not a dictionary, but a {type(data)}")
    except Exception as e:
        print(f"Error loading or inspecting pickle file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_pickle(sys.argv[1])
    else:
        print("Please provide the path to the pickle file.")
