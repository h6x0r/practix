#!/usr/bin/env python3
import re
import os

files = [
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
]

base_path = '/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go'
modified_count = 0

for file in files:
    file_path = os.path.join(base_path, file)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace multiple spaces (2 or more) before // with a single tab
        # This pattern looks for: (any non-whitespace character) + (2 or more spaces) + (//)
        content = re.sub(r'([^\s\t])  +//', r'\1\t//', content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            modified_count += 1
            print(f'Fixed: {file}')
    except Exception as e:
        print(f'Error processing {file}: {str(e)}')

print(f'\nTotal files modified: {modified_count} out of {len(files)}')
