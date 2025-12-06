import { Injectable, ForbiddenException, ServiceUnavailableException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { UsersService } from '../users/users.service';
import { ConfigService } from '@nestjs/config';
import { GoogleGenAI } from "@google/genai";

@Injectable()
export class AiService {
  private genAI: GoogleGenAI;
  
  // Rate Limits
  private readonly LIMIT_FREE = 3;
  private readonly LIMIT_PREMIUM = 15;

  private readonly PROMPT_TEMPLATE = `
      You are an expert programming tutor for \${language}.
      The student is working on the task: "\${taskTitle}".
      
      Here is their current code:
      \`\`\`\${language}
      \${userCode}
      \`\`\`
      
      The student asks: "\${question}"
      
      Provide a helpful, concise hint or explanation. Do not give the full solution code directly. 
      Focus on guiding them to the answer. Use Markdown formatting.
  `;

  constructor(
    private prisma: PrismaService,
    private usersService: UsersService,
    private configService: ConfigService
  ) {
    const apiKey = this.configService.get<string>('GEMINI_API_KEY') || this.configService.get<string>('API_KEY');
    if (apiKey) {
        this.genAI = new GoogleGenAI({ apiKey });
    }
  }

  async askTutor(userId: string, taskTitle: string, userCode: string, question: string, language: string) {
    if (!this.genAI) {
        throw new ServiceUnavailableException('AI Service is not configured (Missing API Key).');
    }

    // 1. Check Rate Limit
    const user = await this.usersService.findById(userId);
    const today = new Date().toISOString().split('T')[0];
    
    // Upsert ensures we have a record for today
    let usage = await (this.prisma as any).aiUsage.findUnique({
        where: {
            userId_date: { userId, date: today }
        }
    });

    if (!usage) {
        usage = await (this.prisma as any).aiUsage.create({
            data: { userId, date: today, count: 0 }
        });
    }

    const limit = user.isPremium ? this.LIMIT_PREMIUM : this.LIMIT_FREE;

    if (usage.count >= limit) {
        throw new ForbiddenException(
            `Daily AI limit reached (${usage.count}/${limit}). Upgrade to Premium for more.`
        );
    }

    // 2. Prepare Prompt
    const prompt = this.PROMPT_TEMPLATE
        .replace('${language}', language || 'General')
        .replace('${taskTitle}', taskTitle || 'Unknown Task')
        .replace('${userCode}', userCode || '')
        .replace('${question}', question);

    try {
        // 3. Call Gemini
        const response = await this.genAI.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        const text = response.text;

        // 4. Increment Usage only if successful
        await (this.prisma as any).aiUsage.update({
            where: { id: usage.id },
            data: { count: { increment: 1 } }
        });

        return { answer: text, remaining: limit - (usage.count + 1) };

    } catch (error) {
        console.error("Gemini Error:", error);
        throw new ServiceUnavailableException("AI is currently overloaded. Please try again.");
    }
  }
}