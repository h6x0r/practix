#!/bin/bash

# ==========================================================================
# AUTOMATED FIX: Go Task File Comment Tabulation
# ==========================================================================
# This script fixes inline comments in Go solutionCode sections by replacing
# multiple spaces before // with a single TAB character.
#
# Usage: chmod +x RUNME_FIX_COMMENTS.sh && ./RUNME_FIX_COMMENTS.sh
# ==========================================================================

set -e  # Exit on error

echo "=========================================="
echo "Fixing Go Task File Comment Tabulation"
echo "=========================================="
echo ""

# Change to the Go modules directory
cd "$(dirname "$0")/server/prisma/seeds/shared/modules/go" || exit 1

# Counter for modified files
count=0

# Process all task files in the specified directories
for dir in logging/topics/implementation/tasks metrics/topics/implementation/tasks panic-recovery/topics/implementation/tasks pointersx/topics/fundamentals/tasks; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"

        for file in "$dir"/*.ts; do
            if [ -f "$file" ]; then
                # Create a backup
                cp "$file" "$file.bak"

                # Use perl to replace 2+ spaces before // with a single tab
                # The -i flag edits in-place, -p prints each line
                perl -pi -e 's/ {2,}(\/\/)/\t$1/g' "$file"

                # Check if file was modified
                if ! cmp -s "$file" "$file.bak"; then
                    ((count++))
                    echo "  ✓ Fixed: $(basename "$file")"
                    rm "$file.bak"  # Remove backup
                else
                    echo "  - No changes: $(basename "$file")"
                    rm "$file.bak"  # Remove backup
                fi
            fi
        done
    else
        echo "Directory not found: $dir"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total files modified: $count"
echo ""
echo "Done! All comment tabulation has been fixed."
echo "Changes:"
echo "  WRONG: code    // comment  (multiple spaces)"
echo "  RIGHT: code\t// comment  (single tab)"
echo ""
