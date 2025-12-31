#!/usr/bin/env python3
import os
import re
import glob

# Directories to check
dirs = [
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/courses/go-design-patterns/modules/behavioral/topics/patterns/tasks",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/courses/go-design-patterns/modules/creational/topics/patterns/tasks",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/courses/go-design-patterns/modules/structural/topics/patterns/tasks"
]

files_with_issues = []

for directory in dirs:
    pattern = os.path.join(directory, "*.ts")
    for filepath in glob.glob(pattern):
        if filepath.endswith("index.ts"):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find solutionCode sections - look for both main and translations
        solution_pattern = r'solutionCode:\s*`([^`]*)`'
        matches = re.findall(solution_pattern, content, re.DOTALL)

        has_issue = False
        for match in matches:
            # Look for lines with code followed by 2+ spaces then //
            # Pattern: non-whitespace, then 2+ spaces, then //
            lines = match.split('\n')
            for i, line in enumerate(lines, 1):
                # Skip lines that start with // (line-beginning comments)
                if line.lstrip().startswith('//'):
                    continue
                # Look for inline comments with multiple spaces
                if re.search(r'[^\s\t]  +//', line):
                    has_issue = True
                    print(f"{filepath}:{i}")
                    print(f"  Issue: {repr(line[:80])}")
                    break
            if has_issue:
                break

        if has_issue:
            files_with_issues.append(filepath)

print(f"\n\nTotal files with issues: {len(files_with_issues)}")
for f in files_with_issues:
    print(f"  - {f}")
