import os

def list_files_recursive(directory):
    ret = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            ret = ret + list_files_recursive(full_path)
        elif os.path.isfile(full_path):
            ret.append(full_path)
    return ret

