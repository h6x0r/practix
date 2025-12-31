#!/bin/bash
# Find all numbered task files and add export default task if missing

count=0
for file in $(find /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds -name "[0-9]*.ts" -path "*/tasks/*"); do
    if ! grep -q "export default task" "$file"; then
        # Check if file ends with newline
        if [ -n "$(tail -c 1 "$file")" ]; then
            echo "" >> "$file"
        fi
        echo "" >> "$file"
        echo "export default task;" >> "$file"
        ((count++))
    fi
done

echo "Added export default to $count files"
