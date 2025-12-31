const fs = require('fs');
const path = require('path');

const files = [
  'encodingx/topics/json-validation/tasks/01-strict-json-validation.ts',
  'encodingx/topics/json-validation/tasks/02-custom-json-marshaler.ts',
  'encodingx/topics/json-validation/tasks/03-json-stream-parser.ts',
  'encodingx/topics/json-validation/tasks/04-json-tag-control.ts',
  'error-handling/topics/fundamentals/tasks/01-sentinel-errors.ts',
  'error-handling/topics/fundamentals/tasks/02-custom-error-type.ts',
  'error-handling/topics/fundamentals/tasks/03-error-unwrap.ts',
  'error-handling/topics/fundamentals/tasks/04-error-wrap.ts',
  'error-handling/topics/fundamentals/tasks/05-error-e.ts',
  'error-handling/topics/fundamentals/tasks/06-is-not-found.ts',
  'error-handling/topics/fundamentals/tasks/07-format-not-found.ts',
  'fundamentals/topics/constructors/tasks/01-functional-options.ts',
  'fundamentals/topics/constructors/tasks/02-factory-method.ts',
  'fundamentals/topics/constructors/tasks/03-constructor-validation.ts',
  'fundamentals/topics/constructors/tasks/04-builder-pattern.ts',
  'fundamentals/topics/data-structures/tasks/01-flatten-nested.ts',
  'fundamentals/topics/data-structures/tasks/02-safe-delete.ts',
  'fundamentals/topics/data-structures/tasks/03-unique.ts',
  'fundamentals/topics/data-structures/tasks/04-reverse-in-place.ts',
  'fundamentals/topics/data-structures/tasks/05-batch.ts',
  'fundamentals/topics/data-structures/tasks/06-join-efficient.ts',
  'fundamentals/topics/io-interfaces/tasks/01-copy-n.ts',
  'fundamentals/topics/io-interfaces/tasks/02-tee-reader.ts',
  'fundamentals/topics/io-interfaces/tasks/03-multi-writer.ts',
  'fundamentals/topics/io-interfaces/tasks/04-limited-reader.ts',
  'generics/topics/fundamentals/tasks/01-generic-function.ts',
  'generics/topics/fundamentals/tasks/02-type-constraints.ts',
  'generics/topics/fundamentals/tasks/03-generic-struct.ts',
  'generics/topics/fundamentals/tasks/04-comparable-constraint.ts',
  'generics/topics/fundamentals/tasks/05-custom-constraint.ts',
  'generics/topics/fundamentals/tasks/06-generic-slice-operations.ts',
  'generics/topics/fundamentals/tasks/07-generic-map-operations.ts',
  'generics/topics/fundamentals/tasks/08-generic-result-type.ts'
];

const basePath = '/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go';
let modifiedCount = 0;

files.forEach(file => {
  const filePath = path.join(basePath, file);
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;

    // Replace multiple spaces (2 or more) before // with a single tab
    // This pattern looks for: (any non-whitespace character) + (2 or more spaces) + (//)
    // We need to be careful to only match trailing comments, not comments at start of line
    content = content.replace(/([^\s\t])  +\/\//g, '$1\t//');

    if (content !== originalContent) {
      fs.writeFileSync(filePath, content, 'utf8');
      modifiedCount++;
      console.log(`Fixed: ${file}`);
    }
  } catch (error) {
    console.error(`Error processing ${file}:`, error.message);
  }
});

console.log(`\nTotal files modified: ${modifiedCount} out of ${files.length}`);
