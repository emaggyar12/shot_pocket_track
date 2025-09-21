# utils/stubs_utils.py
import os, pickle

def read_stub(stub_path):
    if not stub_path or not os.path.exists(stub_path):
        return None
    with open(stub_path, "rb") as f:
        return pickle.load(f)

def save_stub(stub_path, data):
    # If no path provided, silently do nothing
    if not stub_path:
        return
    dirpath = os.path.dirname(stub_path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(data, f)
