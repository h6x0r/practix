#!/bin/bash

# Script to fix comment tabulation in Go task files
# Replaces multiple spaces before // comments with a single tab

files=(
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/01-request-id-context.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/02-structured-fields.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/03-log-level-filter.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/04-async-buffered-logger.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/01-prometheus-endpoint.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/02-thread-safe-counter.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/03-histogram-latency.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/04-gauge-metrics.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/05-metrics-registry.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/01-safe-calls.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/02-safe-goroutines.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/03-panic-to-error.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/04-retry-on-panic.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/01-pointer-operations.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/02-nil-safe-access.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/03-pointer-swap-advanced.ts"
  "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/04-optional-pattern.ts"
)

count=0

for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    # Use sed to replace 2 or more spaces followed by // with a tab and //
    # This specifically targets inline comments after code
    sed -i.bak -E 's/( {2,})(\/\/)/\t\2/g' "$file"

    # Remove backup file
    rm "${file}.bak"

    count=$((count + 1))
    echo "Fixed: $(basename "$file")"
  else
    echo "File not found: $file"
  fi
done

echo ""
echo "Total files modified: $count"
