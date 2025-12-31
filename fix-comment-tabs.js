#!/usr/bin/env node

/**
 * This script fixes inline comment tabulation in Go design pattern task files.
 * It replaces multiple spaces before // comments with a single tab character,
 * but only within solutionCode fields (not in descriptions or examples).
 */

const fs = require('fs');
const path = require('path');

const dirs = [
    "server/prisma/seeds/courses/go-design-patterns/modules/behavioral/topics/patterns/tasks",
    "server/prisma/seeds/courses/go-design-patterns/modules/creational/topics/patterns/tasks",
    "server/prisma/seeds/courses/go-design-patterns/modules/structural/topics/patterns/tasks"
];

let filesProcessed = 0;
let filesModified = 0;
const modifiedFiles = [];

function fixCommentTabulation(content) {
    let modified = false;

    // Process solutionCode fields - match backtick-delimited strings
    const result = content.replace(
        /(solutionCode:\s*`)([^`]*?)(`)/gs,
        (match, prefix, code, suffix) => {
            const lines = code.split('\n');
            const fixedLines = lines.map(line => {
                // Skip lines that are pure comments (start with //)
                const trimmed = line.trimStart();
                if (trimmed.startsWith('//')) {
                    return line;
                }

                // Replace 2+ spaces before // with a single tab
                // Pattern: non-whitespace character, followed by 2+ spaces, then //
                const fixed = line.replace(/([^\s\t])  +\/\//g, '$1\t//');

                if (fixed !== line) {
                    modified = true;
                }

                return fixed;
            });

            return prefix + fixedLines.join('\n') + suffix;
        }
    );

    return { content: result, modified };
}

function processFile(filepath) {
    try {
        const originalContent = fs.readFileSync(filepath, 'utf-8');
        const { content: fixedContent, modified } = fixCommentTabulation(originalContent);

        if (modified) {
            fs.writeFileSync(filepath, fixedContent, 'utf-8');
            filesModified++;
            modifiedFiles.push(filepath);
            console.log(`✓ Fixed: ${path.basename(filepath)}`);
            return true;
        }
        return false;
    } catch (error) {
        console.error(`✗ Error processing ${filepath}:`, error.message);
        return false;
    }
}

function main() {
    console.log('Starting comment tabulation fix...\n');

    dirs.forEach(directory => {
        const fullPath = path.resolve(directory);

        if (!fs.existsSync(fullPath)) {
            console.log(`⚠ Directory not found: ${directory}`);
            return;
        }

        console.log(`Processing: ${directory}`);

        const files = fs.readdirSync(fullPath)
            .filter(f => f.endsWith('.ts') && f !== 'index.ts')
            .sort();

        files.forEach(file => {
            const filepath = path.join(fullPath, file);
            filesProcessed++;
            processFile(filepath);
        });
    });

    console.log('\n' + '='.repeat(70));
    console.log(`Summary:`);
    console.log(`  Files processed: ${filesProcessed}`);
    console.log(`  Files modified: ${filesModified}`);
    console.log(`  Files unchanged: ${filesProcessed - filesModified}`);
    console.log('='.repeat(70));

    if (modifiedFiles.length > 0) {
        console.log('\nModified files:');
        modifiedFiles.forEach((f, idx) => {
            console.log(`  ${idx + 1}. ${path.basename(f)}`);
        });
    } else {
        console.log('\n✓ No files needed modification - all inline comments already use tabs!');
    }
}

main();
