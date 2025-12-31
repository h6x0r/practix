#!/usr/bin/env python3

import os
import glob

# Base directory
base_dir = "/Users/isyahex/Desktop/prjct/kodla-starter"

# Find all task files
pattern = os.path.join(base_dir, "server/prisma/seeds/**/tasks/*.ts")
all_files = glob.glob(pattern, recursive=True)

# Filter files that have "export const task: Task"
task_files = []
for file_path in all_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'export const task: Task' in content:
                task_files.append(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print(f"Found {len(task_files)} task files")

# Check which files already have export default
files_with_export = []
files_needing_export = []

for file_path in task_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if '\nexport default task;' in content or content.endswith('export default task;'):
                files_with_export.append(file_path)
            else:
                files_needing_export.append(file_path)
    except Exception as e:
        print(f"Error checking {file_path}: {e}")

print(f"Files already having export default: {len(files_with_export)}")
print(f"Files needing export default: {len(files_needing_export)}")

# Add export default to files that need it
success_count = 0
error_count = 0
errors = []

for file_path in files_needing_export:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Ensure file ends with proper formatting
        content = content.rstrip()

        # Add export default
        content += '\n\nexport default task;\n'

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        success_count += 1
        if success_count % 50 == 0:
            print(f"✓ Processed {success_count} files...")

    except Exception as e:
        error_count += 1
        errors.append((file_path, str(e)))
        print(f"✗ Error processing {file_path}: {e}")

print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"Total files checked: {len(task_files)}")
print(f"Files that already had export default: {len(files_with_export)}")
print(f"Files needing export default: {len(files_needing_export)}")
print(f"Successfully added export default: {success_count}")
print(f"Errors: {error_count}")

if errors:
    print("\n" + "="*60)
    print("ERRORS:")
    print("="*60)
    for file_path, error in errors:
        print(f"- {file_path}: {error}")

print("\n✓ Done!\n")
