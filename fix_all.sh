#!/bin/bash

# Comprehensive fix for all Go task files
# Replace multiple spaces before // with a single tab

cd /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go

# Find all .ts files in the specified directories and fix them
find logging/topics/implementation/tasks metrics/topics/implementation/tasks panic-recovery/topics/implementation/tasks pointersx/topics/fundamentals/tasks \
  -name "*.ts" -type f \
  -exec perl -pi -e 's/ {2,}(\/\/)/\t$1/g' {} \;

echo "All files have been processed!"
echo "Fixed comment tabulation in all Go task files."
