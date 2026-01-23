export default {
  slug: 'pe-self-reflection',
  title: 'Self-Critique',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Self-Critique

Create a prompt that makes the AI critically evaluate and improve its own output.

## Your Goal

Design a self-improving system that:
1. Generates initial content
2. Critiques its own output
3. Produces an improved version
4. Explains what was improved

## Requirements

Your prompt must:
- Generate initial output
- Apply self-critique
- Iterate to improve
- Include \`{{INPUT}}\` for the content request

## Improvement Cycle
1. Generate → 2. Critique → 3. Improve → 4. Verify
`,
  initialCode: `Create and improve content for:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Write an introduction for a technical blog post about Kubernetes.',
        expectedCriteria: [
          'Produces initial content',
          'Shows self-critique',
          'Provides improved version',
          'Explains improvements made',
        ],
        rubric: 'Response should demonstrate self-reflection and improvement.',
      },
      {
        input: 'Draft a pitch for a mobile app idea.',
        expectedCriteria: [
          'Creates initial pitch',
          'Identifies weaknesses',
          'Improves the pitch',
          'Shows reflection process',
        ],
        rubric: 'Response should show iterative self-improvement.',
      },
    ],
    judgePrompt: `Evaluate self-reflection capability.

Criteria:
1. Critique quality - Is the self-critique meaningful?
2. Improvement - Is the improved version better?
3. Transparency - Is the improvement process visible?
4. Iteration - Does it show genuine refinement?

Score 0-10:
- 0-3: No self-reflection or meaningful improvement
- 4-6: Some reflection but superficial improvements
- 7-8: Good self-critique with real improvements
- 9-10: Excellent iterative improvement with insights`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Самокритика',
      description: `
# Самокритика

Создайте промпт, который заставит ИИ критически оценивать и улучшать свой собственный вывод.

## Ваша цель

Создайте самосовершенствующуюся систему, которая:
1. Генерирует начальный контент
2. Критикует свой собственный вывод
3. Создает улучшенную версию
4. Объясняет, что было улучшено
`,
    },
    tr: {
      title: 'Öz-Eleştiri',
      description: `
# Öz-Eleştiri

AI'ın kendi çıktısını eleştirel olarak değerlendirmesini ve iyileştirmesini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir kendini geliştiren sistem tasarlayın:
1. İlk içeriği üretir
2. Kendi çıktısını eleştirir
3. Geliştirilmiş bir versiyon üretir
4. Neyin iyileştirildiğini açıklar
`,
    },
  },
};
