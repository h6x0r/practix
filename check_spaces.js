const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Directories to check
const dirs = [
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/courses/go-design-patterns/modules/behavioral/topics/patterns/tasks",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/courses/go-design-patterns/modules/creational/topics/patterns/tasks",
    "/Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/courses/go-design-patterns/modules/structural/topics/patterns/tasks"
];

const filesWithIssues = [];

dirs.forEach(directory => {
    const files = fs.readdirSync(directory).filter(f => f.endsWith('.ts') && f !== 'index.ts');

    files.forEach(file => {
        const filepath = path.join(directory, file);
        const content = fs.readFileSync(filepath, 'utf-8');

        // Find solutionCode sections
        const solutionPattern = /solutionCode:\s*`([^`]*)`/gs;
        let match;
        let hasIssue = false;

        while ((match = solutionPattern.exec(content)) !== null) {
            const solutionCode = match[1];
            const lines = solutionCode.split('\n');

            lines.forEach((line, idx) => {
                // Skip lines that start with // (line-beginning comments)
                if (line.trim().startsWith('//')) return;

                // Look for inline comments with 2+ spaces before //
                if (/[^\s\t]  +\/\//.test(line)) {
                    if (!hasIssue) {
                        console.log(`\n${filepath}:`);
                        hasIssue = true;
                    }
                    console.log(`  Line ${idx + 1}: ${line.substring(0, 80)}`);
                }
            });
        }

        if (hasIssue) {
            filesWithIssues.push(filepath);
        }
    });
});

console.log(`\n\nTotal files with issues: ${filesWithIssues.length}`);
filesWithIssues.forEach(f => console.log(`  - ${f}`));
