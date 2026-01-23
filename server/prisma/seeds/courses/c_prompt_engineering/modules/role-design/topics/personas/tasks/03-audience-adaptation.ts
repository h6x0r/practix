export default {
  slug: 'pe-audience-adaptation',
  title: 'Know Your Audience',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Know Your Audience

Create a prompt that adapts explanations for **non-technical stakeholders** like product managers or executives.

## Your Goal

Design a prompt for a technical translator that:
1. Avoids jargon and technical terms
2. Uses business-focused language
3. Emphasizes impact and outcomes over implementation
4. Makes complex concepts accessible

## Requirements

Your prompt must:
- Define the translator role clearly
- Specify the target audience (non-technical)
- Instruct on how to simplify without losing accuracy
- Include \`{{INPUT}}\` for the technical concept

## Example Input
\`\`\`
We need to implement caching to reduce database load and improve API response times from 500ms to 50ms.
\`\`\`

## Expected Output Style
- Business-friendly language
- Focus on benefits (faster app, better user experience)
- Avoid terms like "caching", "database load", "API"
`,
  initialCode: `Explain this technical concept in simple terms for non-technical stakeholders.

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'We need to refactor the authentication module to use OAuth 2.0 with JWT tokens for better security.',
        expectedCriteria: [
          'Avoids technical jargon like OAuth, JWT',
          'Focuses on business benefits (security, user experience)',
          'Uses simple analogies or comparisons',
          'Keeps the core message accurate',
        ],
        rubric: 'Response should translate technical auth concepts into business-friendly language a PM would understand.',
      },
      {
        input: 'The microservices architecture will allow us to scale individual components independently and deploy without downtime.',
        expectedCriteria: [
          'Explains benefits without using "microservices"',
          'Focuses on business outcomes (reliability, speed)',
          'Uses simple language and analogies',
          'Maintains accuracy of the message',
        ],
        rubric: 'Response should make architecture concepts accessible to executives without losing the key points.',
      },
    ],
    judgePrompt: `Evaluate this technical translation for non-technical audiences.

Criteria:
1. Simplicity - Is jargon avoided and replaced with accessible language?
2. Accuracy - Is the core message preserved despite simplification?
3. Business focus - Are benefits and outcomes emphasized?
4. Accessibility - Would a non-technical person understand this?

Score 0-10:
- 0-3: Still too technical, full of jargon
- 4-6: Some simplification but not fully accessible
- 7-8: Good translation with clear business focus
- 9-10: Excellent accessibility while maintaining accuracy`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Знай свою аудиторию',
      description: `
# Знай свою аудиторию

Создайте промпт, который адаптирует объяснения для **нетехнических заинтересованных сторон**, таких как продакт-менеджеры или руководители.

## Ваша цель

Создайте промпт для технического переводчика, который:
1. Избегает жаргона и технических терминов
2. Использует бизнес-ориентированный язык
3. Делает акцент на влиянии и результатах, а не на реализации
4. Делает сложные концепции доступными
`,
    },
    tr: {
      title: 'Hedef Kitlenizi Tanıyın',
      description: `
# Hedef Kitlenizi Tanıyın

**Teknik olmayan paydaşlar** için (ürün yöneticileri veya yöneticiler gibi) açıklamaları uyarlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir teknik çevirmen promptu tasarlayın:
1. Jargon ve teknik terimlerden kaçınır
2. İş odaklı dil kullanır
3. Uygulama yerine etki ve sonuçları vurgular
`,
    },
  },
};
