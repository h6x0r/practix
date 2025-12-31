#!/bin/sh
# Conditional seeding - only seed if DB is empty

set -e

# Check if Course table has any records
COURSE_COUNT=$(npx prisma db execute --stdin <<EOF
SELECT COUNT(*) as count FROM "Course";
EOF
)

echo "Checking database state..."

# If count is 0, run seed
if echo "$COURSE_COUNT" | grep -q '"count":"0"' || echo "$COURSE_COUNT" | grep -q "0 rows"; then
  echo "Database is empty. Running seed..."
  npm run seed
else
  echo "Database already has data. Skipping seed."
fi
