const fs = require('fs');
const path = require('path');

const files = [
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/01-request-id-context.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/02-structured-fields.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/03-log-level-filter.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/04-async-buffered-logger.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/01-prometheus-endpoint.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/02-thread-safe-counter.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/03-histogram-latency.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/04-gauge-metrics.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/05-metrics-registry.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/01-safe-calls.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/02-safe-goroutines.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/03-panic-to-error.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/04-retry-on-panic.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/01-pointer-operations.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/02-nil-safe-access.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/03-pointer-swap-advanced.ts",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/04-optional-pattern.ts",
];

let count = 0;

files.forEach(filePath => {
    if (fs.existsSync(filePath)) {
        let content = fs.readFileSync(filePath, 'utf-8');

        // Replace 2 or more spaces before // with a single tab
        const modified = content.replace(/ {2,}(\/\/)/gm, '\t$1');

        if (content !== modified) {
            fs.writeFileSync(filePath, modified, 'utf-8');
            count++;
            console.log(`Fixed: ${path.basename(filePath)}`);
        } else {
            console.log(`No changes needed: ${path.basename(filePath)}`);
        }
    } else {
        console.log(`File not found: ${filePath}`);
    }
});

console.log(`\nTotal files modified: ${count}`);
