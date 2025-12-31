#!/usr/bin/env python3
"""
Fix comment tabulation in Java Design Pattern task files.
Replaces multiple spaces before // with a single TAB character (inline comments only).
"""

import re
import os
from pathlib import Path

def fix_file(filepath):
    """Fix comment tabulation in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace 2 or more spaces before // with a single tab
    # Pattern matches: (any non-whitespace character) + (2 or more spaces) + (//)
    # This ensures we only fix inline comments, not comments at start of lines
    fixed_content = re.sub(r'([^ \t])  +//', r'\1\t//', content)

    if content != fixed_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    return False

def main():
    base_path = Path('/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/courses/java-design-patterns/modules')

    directories = [
        'behavioral/topics/patterns/tasks',
        'creational/topics/patterns/tasks',
        'structural/topics/patterns/tasks',
    ]

    files_fixed = 0
    files_checked = 0

    print("Fixing comment tabulation in Java Design Pattern task files...")
    print()

    for directory in directories:
        dir_path = base_path / directory
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        # Find all .ts files (excluding index.ts)
        for task_file in sorted(dir_path.glob('*.ts')):
            if task_file.name == 'index.ts':
                continue

            files_checked += 1
            if fix_file(task_file):
                files_fixed += 1
                print(f"✓ Fixed: {task_file.name}")
            else:
                print(f"  Skipped: {task_file.name} (no changes needed)")

    print()
    print(f"Summary:")
    print(f"  Files checked: {files_checked}")
    print(f"  Files fixed: {files_fixed}")
    print(f"  Files unchanged: {files_checked - files_fixed}")

if __name__ == '__main__':
    main()
