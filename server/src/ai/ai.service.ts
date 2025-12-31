import { Injectable, ForbiddenException, ServiceUnavailableException, Logger } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { UsersService } from '../users/users.service';
import { ConfigService } from '@nestjs/config';
import { GoogleGenAI } from "@google/genai";
import { AccessControlService } from '../subscriptions/access-control.service';

@Injectable()
export class AiService {
  private readonly logger = new Logger(AiService.name);
  private genAI: GoogleGenAI;
  
  // Rate Limits
  private readonly LIMIT_FREE = 3;
  private readonly LIMIT_PREMIUM = 15;

  private readonly PROMPT_TEMPLATE = `
You are KODLA AI Tutor - an expert programming mentor for \${language}.

## CRITICAL RULES:
1. **NEVER give the complete solution code** - this is the most important rule
2. **ALWAYS respond in \${uiLanguage}** (the user's interface language)
3. Guide students to discover answers themselves through hints and explanations
4. Be encouraging but honest about mistakes

## CONTEXT:
- Task: "\${taskTitle}"
- Programming Language: \${language}
- User's UI Language: \${uiLanguage}

## Student's Current Code:
\`\`\`\${language}
\${userCode}
\`\`\`

## Student's Question:
"\${question}"

## YOUR RESPONSE APPROACH:
1. First, acknowledge what the student is trying to do
2. If there's a bug: explain WHY it's wrong conceptually, don't just fix it
3. Give a small hint or pseudo-code, never the full implementation
4. Ask a guiding question to make them think
5. If they're stuck on syntax, you can show a small 1-2 line example (not the solution)

## LANGUAGE MAPPING:
- "en" = English
- "ru" = Russian (respond entirely in Russian)
- "uz" = Uzbek (respond entirely in Uzbek)

Remember: Your goal is to TEACH, not to solve. A good tutor asks questions that lead to understanding.
Use Markdown formatting for clarity (code blocks, bullet points, etc.).
  `;

  constructor(
    private prisma: PrismaService,
    private usersService: UsersService,
    private configService: ConfigService,
    private accessControlService: AccessControlService
  ) {
    const apiKey = this.configService.get<string>('GEMINI_API_KEY') || this.configService.get<string>('API_KEY');
    if (apiKey) {
        this.genAI = new GoogleGenAI({ apiKey });
    }
  }

  async askTutor(userId: string, taskId: string, taskTitle: string, userCode: string, question: string, language: string, uiLanguage: string = 'en') {
    if (!this.genAI) {
        throw new ServiceUnavailableException('AI Service is not configured (Missing API Key).');
    }

    // 1. Check subscription access (AI Tutor is premium-only)
    const canUseAiTutor = await this.accessControlService.canUseAiTutor(userId, taskId);
    if (!canUseAiTutor) {
      throw new ForbiddenException('AI Tutor requires a subscription. Upgrade to Premium to access this feature.');
    }

    // 2. Check Rate Limit with atomic transaction to prevent race conditions
    const user = await this.usersService.findById(userId);
    const today = new Date().toISOString().split('T')[0];
    const limit = user.isPremium ? this.LIMIT_PREMIUM : this.LIMIT_FREE;

    // Atomic upsert and check within transaction
    const usageResult = await this.prisma.$transaction(async (tx) => {
      // Create or get usage record
      const usage = await tx.aiUsage.upsert({
        where: { userId_date: { userId, date: today } },
        create: { userId, date: today, count: 0 },
        update: {}, // No update, just ensure exists
      });

      // Check limit BEFORE incrementing
      if (usage.count >= limit) {
        return { exceeded: true, count: usage.count };
      }

      // Atomic increment - only succeeds if count still < limit
      const updated = await tx.aiUsage.updateMany({
        where: {
          userId,
          date: today,
          count: { lt: limit }, // Only increment if still under limit
        },
        data: { count: { increment: 1 } },
      });

      // If no rows were updated, another request beat us
      if (updated.count === 0) {
        return { exceeded: true, count: limit };
      }

      return { exceeded: false, count: usage.count + 1 };
    });

    if (usageResult.exceeded) {
      throw new ForbiddenException(
        `Daily AI limit reached (${usageResult.count}/${limit}). Upgrade to Premium for more.`
      );
    }

    const currentCount = usageResult.count;

    // 2. Prepare Prompt with language mapping
    const languageNames: Record<string, string> = {
      en: 'English',
      ru: 'Russian',
      uz: 'Uzbek'
    };
    const uiLangFull = languageNames[uiLanguage] || 'English';

    const prompt = this.PROMPT_TEMPLATE
        .replace(/\${language}/g, language || 'General')
        .replace(/\${taskTitle}/g, taskTitle || 'Unknown Task')
        .replace(/\${userCode}/g, userCode || '')
        .replace(/\${question}/g, question)
        .replace(/\${uiLanguage}/g, uiLangFull);

    try {
        // 3. Call Gemini
        const response = await this.genAI.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        const text = response.text;

        // Usage was already incremented in the transaction above
        return { answer: text, remaining: limit - currentCount };

    } catch (error) {
        this.logger.error('Gemini API call failed', error instanceof Error ? error.stack : String(error));

        // Rollback the usage increment on failure
        await this.prisma.aiUsage.updateMany({
          where: { userId, date: today, count: { gt: 0 } },
          data: { count: { decrement: 1 } },
        }).catch(() => {}); // Ignore rollback errors

        throw new ServiceUnavailableException("AI is currently overloaded. Please try again.");
    }
  }
}