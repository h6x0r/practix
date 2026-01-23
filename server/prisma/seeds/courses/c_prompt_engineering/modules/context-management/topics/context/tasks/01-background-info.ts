export default {
  slug: 'pe-background-info',
  title: 'Setting the Scene',
  difficulty: 'easy' as const,
  estimatedTime: '10m',
  taskType: 'PROMPT' as const,
  description: `
# Setting the Scene

Create a prompt that provides essential background information for accurate responses.

## Your Goal

Design a prompt for a marketing copywriter that includes:
1. Company/product background
2. Target audience details
3. Brand voice guidelines

## Requirements

Your prompt must:
- Include relevant company context
- Specify the target audience
- Define the expected tone and style
- Include \`{{INPUT}}\` for the content request

## Example Input
\`\`\`
Write a product description for our new wireless headphones.
\`\`\`

## Expected Output Style
- Content aligned with brand voice
- Relevant to target audience
- Consistent with company positioning
`,
  initialCode: `Write marketing copy for:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Write a social media post announcing our summer sale.',
        expectedCriteria: [
          'Uses appropriate brand voice/tone',
          'Targets the specified audience',
          'Aligns with any company context provided',
          'Fits the platform format (social media)',
        ],
        rubric: 'Response should demonstrate use of background context to create aligned marketing copy.',
      },
      {
        input: 'Create an email subject line for our newsletter about new features.',
        expectedCriteria: [
          'Reflects brand personality',
          'Appeals to target audience',
          'Uses appropriate tone',
          'Fits email format expectations',
        ],
        rubric: 'Response should use provided context to craft an appropriate subject line.',
      },
    ],
    judgePrompt: `Evaluate how well the response uses background context.

Criteria:
1. Brand alignment - Does it match the brand voice?
2. Audience fit - Is it appropriate for the target audience?
3. Context usage - Is the background information reflected?
4. Consistency - Is the output coherent with given context?

Score 0-10:
- 0-3: Ignores context, generic output
- 4-6: Some context used but inconsistent
- 7-8: Good use of background information
- 9-10: Excellent context integration throughout`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Создание контекста',
      description: `
# Создание контекста

Создайте промпт, который предоставляет необходимую фоновую информацию для точных ответов.

## Ваша цель

Создайте промпт для маркетингового копирайтера, который включает:
1. Информацию о компании/продукте
2. Детали целевой аудитории
3. Рекомендации по голосу бренда
`,
    },
    tr: {
      title: 'Sahneyi Hazırlamak',
      description: `
# Sahneyi Hazırlamak

Doğru yanıtlar için gerekli arka plan bilgilerini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları içeren bir pazarlama metin yazarı promptu tasarlayın:
1. Şirket/ürün arka planı
2. Hedef kitle detayları
3. Marka sesi yönergeleri
`,
    },
  },
};
