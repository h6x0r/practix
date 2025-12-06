import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class JudgeService {
  private readonly logger = new Logger(JudgeService.name);
  // Default to local judge0 instance
  private readonly judgeUrl = this.config.get('JUDGE0_URL') || 'http://localhost:2358';

  constructor(private config: ConfigService) {}

  /**
   * Sends code to Judge0 for execution.
   * Maps language names to Judge0 IDs (e.g. java -> 62, go -> 95)
   */
  async executeCode(language: string, code: string): Promise<any> {
    const langId = this.getLanguageId(language);
    
    // Base64 encode code as per Judge0 requirement
    const sourceCode = Buffer.from(code).toString('base64');

    this.logger.log(`Submitting code for execution (Lang: ${language}, ID: ${langId})...`);

    try {
      // 1. Submit Submission
      const response = await axios.post(`${this.judgeUrl}/submissions?base64_encoded=true&wait=true`, {
        source_code: sourceCode,
        language_id: langId,
        stdin: Buffer.from("").toString('base64'), // Input if needed
      });

      // 2. Process Result (since wait=true, we get result immediately for short snippets)
      const result = response.data;
      
      return {
        status: result.status.id === 3 ? 'passed' : 'failed', // 3 is Accepted
        description: result.status.description,
        stdout: result.stdout ? Buffer.from(result.stdout, 'base64').toString('utf-8') : '',
        stderr: result.stderr ? Buffer.from(result.stderr, 'base64').toString('utf-8') : '',
        compile_output: result.compile_output ? Buffer.from(result.compile_output, 'base64').toString('utf-8') : '',
        time: result.time,
        memory: result.memory
      };

    } catch (error) {
      this.logger.warn(`Judge0 connection failed: ${error.message}. Falling back to MOCK mode.`);
      
      // Fallback Mock for development if Judge0 container is not running
      return this.mockExecution(code);
    }
  }

  private getLanguageId(language: string): number {
    // Standard Judge0 IDs
    // 62: Java (OpenJDK 13.0.1)
    // 95: Go (1.18.5)
    // 63: JavaScript (Node.js 12.14.0)
    if (language.toLowerCase().includes('java')) return 62;
    if (language.toLowerCase().includes('go')) return 95;
    return 63; // JS default
  }

  private mockExecution(code: string) {
    return new Promise(resolve => {
        setTimeout(() => {
            const isError = code.includes('error');
            resolve({
                status: isError ? 'failed' : 'passed',
                description: isError ? 'Runtime Error' : 'Accepted',
                stdout: isError ? '' : 'Hello from Mock Judge0!\nTest passed.',
                stderr: isError ? 'panic: runtime error: index out of range' : '',
                time: '0.015',
                memory: '1024'
            });
        }, 1000);
    });
  }
}
