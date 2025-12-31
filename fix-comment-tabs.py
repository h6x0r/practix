#!/usr/bin/env python3
"""
Fix comment tabulation in Go task files.
Replaces multiple spaces before // with a single TAB character.
"""

import os
import re
from pathlib import Path

def fix_file(filepath):
    """Fix comment tabulation in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace 2 or more spaces before // with a single tab
    # This pattern matches any sequence of 2+ spaces followed by //
    fixed_content = re.sub(r'  +//', '\t//', content)

    if content != fixed_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    return False

def main():
    base_path = Path('/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go')

    directories = [
        'caching',
        'channels',
        'circuit-breaker',
        'concurrency-patterns',
    ]

    files_fixed = 0
    files_checked = 0

    for directory in directories:
        dir_path = base_path / directory / 'topics'
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        # Find all .ts files in tasks subdirectories
        for task_file in dir_path.rglob('tasks/*.ts'):
            files_checked += 1
            if fix_file(task_file):
                files_fixed += 1
                print(f"Fixed: {task_file.relative_to(base_path)}")

    print(f"\nSummary:")
    print(f"  Files checked: {files_checked}")
    print(f"  Files fixed: {files_fixed}")
    print(f"  Files unchanged: {files_checked - files_fixed}")

if __name__ == '__main__':
    main()
