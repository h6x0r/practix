export default {
  slug: 'pe-meta-prompting',
  title: 'Prompts About Prompts',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Prompts About Prompts

Create a meta-prompt that generates effective prompts for specific use cases.

## Your Goal

Design a prompt generator that:
1. Takes a use case description
2. Generates an optimized prompt for that use case
3. Explains why the generated prompt works

## Requirements

Your prompt must:
- Understand prompt engineering principles
- Generate task-specific prompts
- Include best practices automatically
- Include \`{{INPUT}}\` for the use case

## Example Input
\`\`\`
I need a prompt for summarizing legal documents for non-lawyers.
\`\`\`

## Expected Output
A well-crafted prompt tailored to that specific use case.
`,
  initialCode: `Generate a prompt for this use case:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'I need a prompt for helping students debug their Python code.',
        expectedCriteria: [
          'Generates a relevant prompt',
          'Prompt includes debugging best practices',
          'Tailored for student level',
          'Explains prompt design choices',
        ],
        rubric: 'Response should generate an effective, tailored prompt.',
      },
      {
        input: 'Create a prompt for writing product descriptions for an e-commerce site.',
        expectedCriteria: [
          'Generates e-commerce focused prompt',
          'Includes relevant instructions',
          'Considers conversion optimization',
          'Well-structured output',
        ],
        rubric: 'Response should demonstrate meta-prompting ability.',
      },
    ],
    judgePrompt: `Evaluate the meta-prompting quality.

Criteria:
1. Relevance - Is the generated prompt relevant to the use case?
2. Quality - Is the generated prompt well-crafted?
3. Best practices - Does it include prompt engineering best practices?
4. Explanation - Is the reasoning explained?

Score 0-10:
- 0-3: Poor or irrelevant prompt generated
- 4-6: Basic prompt but missing optimization
- 7-8: Good tailored prompt with explanation
- 9-10: Excellent optimized prompt with full reasoning`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Промпты о промптах',
      description: `
# Промпты о промптах

Создайте мета-промпт, который генерирует эффективные промпты для конкретных случаев использования.

## Ваша цель

Создайте генератор промптов, который:
1. Принимает описание случая использования
2. Генерирует оптимизированный промпт для этого случая
3. Объясняет, почему сгенерированный промпт работает
`,
    },
    tr: {
      title: 'Promptlar Hakkında Promptlar',
      description: `
# Promptlar Hakkında Promptlar

Belirli kullanım durumları için etkili promptlar üreten bir meta-prompt oluşturun.

## Hedefiniz

Şunları yapan bir prompt üreteci tasarlayın:
1. Kullanım durumu açıklamasını alır
2. Bu kullanım durumu için optimize edilmiş bir prompt üretir
3. Üretilen promptun neden işe yaradığını açıklar
`,
    },
  },
};
