export default {
  slug: 'pe-show-your-work',
  title: 'Show Your Work',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Show Your Work

Create a prompt that makes the AI explicitly state its assumptions and reasoning before giving conclusions.

## Your Goal

Design a prompt for a business analyst that:
1. States assumptions explicitly before analysis
2. Shows reasoning for each conclusion
3. Distinguishes facts from interpretations

## Requirements

Your prompt must:
- Ask for explicit assumption listing
- Request reasoning for conclusions
- Include \`{{INPUT}}\` for the analysis topic

## Example Input
\`\`\`
Our app's daily active users dropped from 10,000 to 8,000 over the last month. Analyze why.
\`\`\`

## Expected Output Style
- Assumptions listed clearly
- Analysis based on stated assumptions
- Conclusions tied to specific reasoning
`,
  initialCode: `Analyze this:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Our e-commerce conversion rate dropped from 3% to 2% after we changed the checkout button color from green to blue.',
        expectedCriteria: [
          'States assumptions explicitly',
          'Considers multiple possible causes',
          'Shows reasoning for conclusions',
          'Distinguishes correlation from causation',
        ],
        rubric: 'Response should explicitly state assumptions and show reasoning, not just jump to conclusions.',
      },
      {
        input: 'Our customer support tickets increased 40% after launching a new feature. Is the feature buggy?',
        expectedCriteria: [
          'Lists assumptions being made',
          'Explores multiple explanations',
          'Shows reasoning for each possibility',
          'Avoids jumping to conclusions',
        ],
        rubric: 'Response should demonstrate careful reasoning with explicit assumptions.',
      },
    ],
    judgePrompt: `Evaluate if the response shows explicit reasoning and assumptions.

Criteria:
1. Assumptions - Are assumptions clearly stated?
2. Reasoning - Is the logic chain visible?
3. Evidence-based - Are conclusions tied to evidence?
4. Transparency - Is the thought process clear?

Score 0-10:
- 0-3: Jumps to conclusions without showing reasoning
- 4-6: Some reasoning shown but assumptions unclear
- 7-8: Good explicit reasoning with clear assumptions
- 9-10: Excellent transparency with full reasoning chain`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Покажи ход решения',
      description: `
# Покажи ход решения

Создайте промпт, который заставит ИИ явно указывать свои предположения и рассуждения перед выводами.

## Ваша цель

Создайте промпт для бизнес-аналитика, который:
1. Явно указывает предположения перед анализом
2. Показывает рассуждения для каждого вывода
3. Различает факты и интерпретации
`,
    },
    tr: {
      title: 'Çalışmanı Göster',
      description: `
# Çalışmanı Göster

AI'ın sonuçlara varmadan önce varsayımlarını ve akıl yürütmesini açıkça belirtmesini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir iş analisti promptu tasarlayın:
1. Analizden önce varsayımları açıkça belirtir
2. Her sonuç için akıl yürütmeyi gösterir
3. Gerçekleri yorumlardan ayırır
`,
    },
  },
};
