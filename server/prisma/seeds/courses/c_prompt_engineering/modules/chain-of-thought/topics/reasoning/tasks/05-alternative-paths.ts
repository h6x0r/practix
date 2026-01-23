export default {
  slug: 'pe-alternative-paths',
  title: 'Explore Alternatives',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Explore Alternatives

Create a prompt that makes the AI explore multiple solution paths before choosing the best one.

## Your Goal

Design a prompt for a solution architect that:
1. Generates at least 3 different approaches
2. Evaluates pros and cons of each
3. Recommends the best option with justification

## Requirements

Your prompt must:
- Request multiple solution alternatives
- Ask for evaluation criteria
- Require justified recommendation
- Include \`{{INPUT}}\` for the problem

## Example Input
\`\`\`
We need to implement user authentication for our web application.
\`\`\`

## Expected Output Style
- 3+ distinct approaches listed
- Pros/cons for each approach
- Clear recommendation with reasoning
`,
  initialCode: `Suggest a solution for:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'We need to store user preferences that should persist across sessions and devices.',
        expectedCriteria: [
          'Presents multiple approaches (database, localStorage, cloud sync)',
          'Evaluates pros and cons of each',
          'Considers trade-offs (complexity, cost, UX)',
          'Provides justified recommendation',
        ],
        rubric: 'Response should explore multiple storage solutions with clear evaluation.',
      },
      {
        input: 'Our API is slow and needs caching. What approach should we take?',
        expectedCriteria: [
          'Lists multiple caching strategies',
          'Evaluates each approach objectively',
          'Considers trade-offs',
          'Recommends with clear justification',
        ],
        rubric: 'Response should explore caching alternatives with thoughtful comparison.',
      },
    ],
    judgePrompt: `Evaluate the exploration of alternative solutions.

Criteria:
1. Alternatives - Are multiple distinct options presented?
2. Evaluation - Are pros/cons clearly analyzed?
3. Comparison - Are trade-offs considered fairly?
4. Justification - Is the recommendation well-reasoned?

Score 0-10:
- 0-3: Single solution without alternatives
- 4-6: Some alternatives but weak evaluation
- 7-8: Good multi-option analysis with clear recommendation
- 9-10: Excellent comprehensive exploration with deep analysis`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Исследуй альтернативы',
      description: `
# Исследуй альтернативы

Создайте промпт, который заставит ИИ исследовать несколько путей решения перед выбором лучшего.

## Ваша цель

Создайте промпт для архитектора решений, который:
1. Генерирует минимум 3 разных подхода
2. Оценивает плюсы и минусы каждого
3. Рекомендует лучший вариант с обоснованием
`,
    },
    tr: {
      title: 'Alternatifleri Keşfet',
      description: `
# Alternatifleri Keşfet

AI'ın en iyisini seçmeden önce birden fazla çözüm yolunu keşfetmesini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir çözüm mimarı promptu tasarlayın:
1. En az 3 farklı yaklaşım üretir
2. Her birinin artı ve eksilerini değerlendirir
3. Gerekçeyle en iyi seçeneği önerir
`,
    },
  },
};
