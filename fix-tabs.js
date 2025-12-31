const fs = require('fs');
const path = require('path');
const { glob } = require('glob');

// Find all task files in the specified directories
const basePath = '/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go';
const patterns = [
  `${basePath}/caching/topics/*/tasks/*.ts`,
  `${basePath}/channels/topics/*/tasks/*.ts`,
  `${basePath}/circuit-breaker/topics/*/tasks/*.ts`,
  `${basePath}/concurrency-patterns/topics/*/tasks/*.ts`,
];

let filesFixed = 0;

for (const pattern of patterns) {
  const files = glob.sync(pattern);

  for (const file of files) {
    const content = fs.readFileSync(file, 'utf8');

    // Replace multiple spaces before // with a single tab
    // This pattern matches 2 or more spaces followed by //
    const fixed = content.replace(/  +\/\//g, '\t//');

    if (content !== fixed) {
      fs.writeFileSync(file, fixed, 'utf8');
      filesFixed++;
      console.log(`Fixed: ${path.relative(basePath, file)}`);
    }
  }
}

console.log(`\nTotal files fixed: ${filesFixed}`);
