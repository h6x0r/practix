import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { glob } from 'glob';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  const baseDir = __dirname;

  // Find all task files
  const pattern = 'server/prisma/seeds/**/tasks/*.ts';
  const allFiles = await glob(pattern, { cwd: baseDir, absolute: true });

  console.log(`Found ${allFiles.length} .ts files in tasks directories`);

  // Filter files that have "export const task: Task"
  const taskFiles = [];
  for (const filePath of allFiles) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      if (content.includes('export const task: Task')) {
        taskFiles.append(filePath);
      }
    } catch (err) {
      console.error(`Error reading ${filePath}:`, err.message);
    }
  }

  console.log(`Found ${taskFiles.length} task files with "export const task: Task"`);

  // Check which files already have export default
  const filesWithExport = [];
  const filesNeedingExport = [];

  for (const filePath of taskFiles) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      if (content.includes('\nexport default task;') || content.endsWith('export default task;')) {
        filesWithExport.push(filePath);
      } else {
        filesNeedingExport.push(filePath);
      }
    } catch (err) {
      console.error(`Error checking ${filePath}:`, err.message);
    }
  }

  console.log(`Files already having export default: ${filesWithExport.length}`);
  console.log(`Files needing export default: ${filesNeedingExport.length}`);

  // Add export default to files that need it
  let successCount = 0;
  let errorCount = 0;
  const errors = [];

  for (const filePath of filesNeedingExport) {
    try {
      let content = fs.readFileSync(filePath, 'utf-8');

      // Ensure file ends with proper formatting
      content = content.trimEnd();

      // Add export default
      content += '\n\nexport default task;\n';

      // Write back to file
      fs.writeFileSync(filePath, content, 'utf-8');

      successCount++;
      if (successCount % 50 === 0) {
        console.log(`✓ Processed ${successCount} files...`);
      }
    } catch (err) {
      errorCount++;
      errors.push({ file: filePath, error: err.message });
      console.error(`✗ Error processing ${filePath}:`, err.message);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY REPORT');
  console.log('='.repeat(60));
  console.log(`Total files checked: ${taskFiles.length}`);
  console.log(`Files that already had export default: ${filesWithExport.length}`);
  console.log(`Files needing export default: ${filesNeedingExport.length}`);
  console.log(`Successfully added export default: ${successCount}`);
  console.log(`Errors: ${errorCount}`);

  if (errors.length > 0) {
    console.log('\n' + '='.repeat(60));
    console.log('ERRORS:');
    console.log('='.repeat(60));
    errors.forEach(({ file, error }) => {
      console.log(`- ${file}: ${error}`);
    });
  }

  console.log('\n✓ Done!\n');
}

main().catch(console.error);
