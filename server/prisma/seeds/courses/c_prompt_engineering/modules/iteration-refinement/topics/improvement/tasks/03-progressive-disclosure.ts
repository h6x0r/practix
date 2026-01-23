export default {
  slug: 'pe-progressive-disclosure',
  title: 'Progressive Detail',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Progressive Detail

Create a prompt that provides layered responses with increasing detail on request.

## Your Goal

Design a prompt for a technical explainer that:
1. Starts with a simple overview
2. Offers to go deeper on request
3. Provides progressively more detail

## Requirements

Your prompt must:
- Provide a concise initial answer
- Indicate that more detail is available
- Structure information in layers
- Include \`{{INPUT}}\` for the topic

## Example Flow
1. TL;DR summary
2. Basic explanation
3. Detailed technical explanation
4. Expert-level deep dive
`,
  initialCode: `Explain this topic:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'How does HTTPS encryption work?',
        expectedCriteria: [
          'Provides layered explanation',
          'Starts with simple overview',
          'Indicates more detail available',
          'Information is progressively deeper',
        ],
        rubric: 'Response should demonstrate progressive disclosure of information.',
      },
      {
        input: 'What is containerization in software?',
        expectedCriteria: [
          'Begins with high-level summary',
          'Offers deeper explanation options',
          'Structures information in layers',
          'Accessible at multiple levels',
        ],
        rubric: 'Response should show layered information architecture.',
      },
    ],
    judgePrompt: `Evaluate progressive disclosure.

Criteria:
1. Layering - Is information provided in clear layers?
2. Accessibility - Is the initial level accessible?
3. Depth - Are deeper levels genuinely more detailed?
4. Navigation - Is it clear how to get more detail?

Score 0-10:
- 0-3: No layering, all at one level
- 4-6: Some layering but unclear progression
- 7-8: Good progressive disclosure
- 9-10: Excellent layered information architecture`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Прогрессивная детализация',
      description: `
# Прогрессивная детализация

Создайте промпт, который предоставляет многоуровневые ответы с увеличивающейся детализацией по запросу.

## Ваша цель

Создайте промпт для технического объяснителя, который:
1. Начинает с простого обзора
2. Предлагает углубиться по запросу
3. Предоставляет прогрессивно больше деталей
`,
    },
    tr: {
      title: 'Aşamalı Detaylandırma',
      description: `
# Aşamalı Detaylandırma

İstek üzerine artan detaylarla katmanlı yanıtlar sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir teknik açıklayıcı promptu tasarlayın:
1. Basit bir genel bakışla başlar
2. İstek üzerine daha derine inmeyi teklif eder
3. Giderek daha fazla detay sağlar
`,
    },
  },
};
