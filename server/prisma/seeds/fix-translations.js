/**
 * Script to fix Uzbek translation issues:
 * 1. Replace single quotes around code with backticks
 * 2. Ensure proper line breaks in numbered lists
 */

const fs = require('fs');
const path = require('path');

// Find all .ts files in tasks directories
function findTaskFiles(dir, files = []) {
    const items = fs.readdirSync(dir);
    for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        if (stat.isDirectory()) {
            findTaskFiles(fullPath, files);
        } else if (item.endsWith('.ts') && fullPath.includes('/tasks/')) {
            files.push(fullPath);
        }
    }
    return files;
}

// Fix single quotes in Uzbek translations
function fixSingleQuotes(content) {
    // Pattern: '...' in Uzbek descriptions where ... contains code-like text
    // Replace 1. 'code' with 1. \`code\`
    // This handles patterns like: 1. 'FunctionName[T any](...) ...' funksiyasini

    let fixed = content;

    // Find Uzbek sections and fix single quotes around function signatures
    // Pattern: number. 'code' text
    fixed = fixed.replace(/(\d+)\. '([^']+)' (funksiya|metod|struktura|interfeys)/g,
        (match, num, code, word) => `${num}. \\\`${code}\\\` ${word}`);

    // Pattern: '...' funksiyasini at start of line or after newline
    fixed = fixed.replace(/(\n\d+)\. '([^']+)'/g,
        (match, prefix, code) => `${prefix}. \\\`${code}\\\``);

    return fixed;
}

// Process all files
const seedsDir = '/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules';
const files = findTaskFiles(seedsDir);

console.log(`Found ${files.length} task files`);

let fixedCount = 0;
let issuesFound = [];

for (const file of files) {
    const content = fs.readFileSync(file, 'utf8');

    // Check for single quotes pattern in Uzbek
    if (content.includes("1. '") && content.includes('uz:')) {
        const fixed = fixSingleQuotes(content);
        if (fixed !== content) {
            fs.writeFileSync(file, fixed, 'utf8');
            fixedCount++;
            console.log(`Fixed: ${path.basename(file)}`);
        } else {
            issuesFound.push(file);
        }
    }
}

console.log(`\nFixed ${fixedCount} files`);
if (issuesFound.length > 0) {
    console.log(`\nFiles that may need manual review:`);
    issuesFound.forEach(f => console.log(`  - ${path.relative(seedsDir, f)}`));
}
