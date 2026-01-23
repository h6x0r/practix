export default {
  slug: 'pe-step-by-step',
  title: 'Think Step by Step',
  difficulty: 'easy' as const,
  estimatedTime: '10m',
  taskType: 'PROMPT' as const,
  description: `
# Think Step by Step

Create a prompt that makes the AI show its reasoning process when solving problems.

## Your Goal

Design a prompt for a math tutor that:
1. Solves problems step by step
2. Explains each step clearly
3. Shows the thought process, not just the answer

## Requirements

Your prompt must:
- Instruct the AI to think step by step
- Ask for explanation of each step
- Include \`{{INPUT}}\` for the problem to solve

## Example Input
\`\`\`
A train leaves station A at 9:00 AM traveling at 60 mph. Another train leaves station B at 10:00 AM traveling toward station A at 80 mph. If stations are 280 miles apart, when do they meet?
\`\`\`

## Expected Output Style
- Numbered steps
- Clear explanation of each calculation
- Final answer with verification
`,
  initialCode: `Solve this problem:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'A store sells shirts for $25. During a sale, they offer 20% off. What is the sale price?',
        expectedCriteria: [
          'Shows step-by-step reasoning',
          'Calculates the discount amount (25 * 0.20 = 5)',
          'Subtracts to get final price (25 - 5 = 20)',
          'Clearly states the final answer ($20)',
        ],
        rubric: 'Response should show explicit step-by-step calculation, not just the answer.',
      },
      {
        input: 'If a car travels at 45 mph for 2.5 hours, how far does it go?',
        expectedCriteria: [
          'Identifies the formula (distance = speed × time)',
          'Shows the calculation explicitly',
          'Provides the answer with units (112.5 miles)',
          'Demonstrates clear reasoning process',
        ],
        rubric: 'Response should walk through the reasoning, showing the formula and calculation steps.',
      },
    ],
    judgePrompt: `Evaluate if the response demonstrates step-by-step reasoning.

Criteria:
1. Explicit steps - Are reasoning steps clearly shown?
2. Logical flow - Does each step follow logically?
3. Clarity - Is each step easy to understand?
4. Completeness - Does it reach the final answer?

Score 0-10:
- 0-3: Just gives answer without showing work
- 4-6: Some steps shown but incomplete reasoning
- 7-8: Good step-by-step explanation
- 9-10: Excellent detailed reasoning with clear logic`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Думай пошагово',
      description: `
# Думай пошагово

Создайте промпт, который заставит ИИ показывать процесс рассуждений при решении задач.

## Ваша цель

Создайте промпт для репетитора по математике, который:
1. Решает задачи шаг за шагом
2. Объясняет каждый шаг понятно
3. Показывает процесс мышления, а не только ответ
`,
    },
    tr: {
      title: 'Adım Adım Düşün',
      description: `
# Adım Adım Düşün

AI'ın problemleri çözerken akıl yürütme sürecini göstermesini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir matematik öğretmeni promptu tasarlayın:
1. Problemleri adım adım çözer
2. Her adımı açıkça açıklar
3. Sadece cevabı değil, düşünce sürecini gösterir
`,
    },
  },
};
