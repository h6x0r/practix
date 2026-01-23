export default {
  slug: 'pe-problem-decomposition',
  title: 'Break It Down',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Break It Down

Create a prompt that makes the AI decompose complex problems into smaller, manageable sub-problems.

## Your Goal

Design a prompt for a problem-solving assistant that:
1. Identifies sub-problems within a complex question
2. Solves each sub-problem separately
3. Combines solutions for the final answer

## Requirements

Your prompt must:
- Instruct the AI to break problems into parts
- Solve each part before combining
- Include \`{{INPUT}}\` for the complex problem

## Example Input
\`\`\`
Plan a weekend trip to Paris including flights, hotel, and activities for 2 people with a $2000 budget.
\`\`\`

## Expected Output Style
- Identified sub-problems listed
- Each sub-problem addressed
- Solutions combined into final recommendation
`,
  initialCode: `Help me solve this complex problem:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Design a simple e-commerce checkout flow that handles cart review, shipping address, payment, and order confirmation.',
        expectedCriteria: [
          'Breaks down into distinct steps/sub-problems',
          'Addresses cart review component',
          'Addresses shipping and payment separately',
          'Provides a coherent combined solution',
        ],
        rubric: 'Response should clearly decompose the checkout flow into sub-problems and address each.',
      },
      {
        input: 'Calculate the total cost of painting a room that is 12ft x 15ft x 10ft high, with 2 doors and 3 windows, if paint costs $30/gallon and covers 400 sq ft.',
        expectedCriteria: [
          'Breaks into sub-problems (wall area, subtract openings, calculate gallons)',
          'Calculates total wall area',
          'Accounts for doors and windows',
          'Determines gallons needed and total cost',
        ],
        rubric: 'Response should decompose the painting cost problem systematically.',
      },
    ],
    judgePrompt: `Evaluate the problem decomposition approach.

Criteria:
1. Identification - Are sub-problems clearly identified?
2. Separation - Is each part solved independently?
3. Integration - Are solutions combined coherently?
4. Completeness - Are all parts of the problem addressed?

Score 0-10:
- 0-3: No decomposition, treats as single problem
- 4-6: Some breakdown but incomplete or poorly integrated
- 7-8: Good decomposition with clear sub-problem solving
- 9-10: Excellent systematic breakdown with perfect integration`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Разбей на части',
      description: `
# Разбей на части

Создайте промпт, который заставит ИИ разбивать сложные задачи на более мелкие, управляемые подзадачи.

## Ваша цель

Создайте промпт для помощника по решению проблем, который:
1. Выявляет подзадачи внутри сложного вопроса
2. Решает каждую подзадачу отдельно
3. Объединяет решения для финального ответа
`,
    },
    tr: {
      title: 'Parçalara Ayır',
      description: `
# Parçalara Ayır

AI'ın karmaşık problemleri daha küçük, yönetilebilir alt problemlere ayırmasını sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir problem çözme asistanı promptu tasarlayın:
1. Karmaşık soru içindeki alt problemleri belirler
2. Her alt problemi ayrı ayrı çözer
3. Çözümleri son cevap için birleştirir
`,
    },
  },
};
