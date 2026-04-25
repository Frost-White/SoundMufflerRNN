import concurrent.futures
import os
import sys

# Compute the path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "data", "chunk_train")
NUM_WORKERS = os.cpu_count() or 1


def print_progress(current, total, width=40):
    if total == 0:
        return
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"Removing |{bar}| {current}/{total}"
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


def remove_file(path):
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            return True
    except OSError:
        pass
    return False


def remove_tree_with_progress(path):
    files = []
    dirs = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
        for dirname in dirnames:
            dirs.append(os.path.join(root, dirname))

    dirs.sort(reverse=True)
    total = len(files) + len(dirs) + 1
    current = 0

    if files:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(remove_file, file_path) for file_path in files]
            for _ in concurrent.futures.as_completed(futures):
                current += 1
                print_progress(current, total)

    for dir_path in dirs:
        try:
            os.rmdir(dir_path)
        except OSError:
            pass
        current += 1
        print_progress(current, total)

    try:
        os.rmdir(path)
    except OSError:
        pass
    current += 1
    print_progress(current, total)
    sys.stdout.write("\n")


if os.path.isdir(TARGET_DIR):
    print(f"Removing directory: {TARGET_DIR}")
    remove_tree_with_progress(TARGET_DIR)
    print("Done. data/chunk_train has been deleted.")
else:
    print(f"Directory not found: {TARGET_DIR}")
