import concurrent.futures
import os
import sys

from console_bar import bar_line

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "chunk_train"))
NUM_WORKERS = os.cpu_count() or 1


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
                sys.stdout.write("\r" + bar_line("Removing", current, total))
                sys.stdout.flush()

    for dir_path in dirs:
        try:
            os.rmdir(dir_path)
        except OSError:
            pass
        current += 1
        sys.stdout.write("\r" + bar_line("Removing", current, total))
        sys.stdout.flush()

    try:
        os.rmdir(path)
    except OSError:
        pass
    current += 1
    sys.stdout.write("\r" + bar_line("Removing", current, total))
    sys.stdout.flush()
    sys.stdout.write("\n")


if os.path.isdir(TARGET_DIR):
    print(f"Removing directory: {TARGET_DIR}")
    remove_tree_with_progress(TARGET_DIR)
    print("Done. data/chunk_train has been deleted.")
else:
    print(f"Directory not found: {TARGET_DIR}")
