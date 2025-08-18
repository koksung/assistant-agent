import os

IGNORE_DIRS = {'__pycache__', '.venv', '.git', '.idea', '.mypy_cache', 'pdfs'}
IGNORE_FILES = set()
HIDDEN_PREFIX = '.'  # skip all hidden files/folders

def print_tree(startpath='.', prefix=''):
    entries = sorted(os.listdir(startpath))
    entries = [
        e for e in entries
        if not e.startswith(HIDDEN_PREFIX)
        and e not in IGNORE_DIRS
        and e not in IGNORE_FILES
    ]

    for index, entry in enumerate(entries):
        path = os.path.join(startpath, entry)
        connector = '└── ' if index == len(entries) - 1 else '├── '
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = '    ' if index == len(entries) - 1 else '│   '
            print_tree(path, prefix + extension)

if __name__ == '__main__':
    print_tree('../../.')  # Always start from current directory
