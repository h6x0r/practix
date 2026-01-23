export default {
  slug: 'pe-prompt-chaining',
  title: 'Chain of Prompts',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Chain of Prompts

Create a prompt that uses the output of one step as input for the next.

## Your Goal

Design a multi-stage pipeline that:
1. Breaks a complex task into stages
2. Each stage builds on the previous
3. Produces a final refined output

## Requirements

Your prompt must:
- Define clear stages
- Pass output between stages
- Show intermediate results
- Include \`{{INPUT}}\` for the initial task

## Pipeline Example
\`\`\`
Stage 1: Research → Stage 2: Outline → Stage 3: Draft → Stage 4: Polish
\`\`\`
`,
  initialCode: `Process this through multiple stages:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Create a comprehensive guide about setting up a home office.',
        expectedCriteria: [
          'Shows multi-stage approach',
          'Each stage builds on previous',
          'Intermediate outputs visible',
          'Final output is refined',
        ],
        rubric: 'Response should demonstrate clear prompt chaining.',
      },
      {
        input: 'Develop a marketing strategy for a new product launch.',
        expectedCriteria: [
          'Uses staged approach',
          'Stages are logically connected',
          'Shows pipeline progression',
          'Produces comprehensive output',
        ],
        rubric: 'Response should show multi-stage processing.',
      },
    ],
    judgePrompt: `Evaluate prompt chaining quality.

Criteria:
1. Stages - Are clear stages defined?
2. Connection - Do stages build on each other?
3. Visibility - Are intermediate results shown?
4. Quality - Is the final output refined?

Score 0-10:
- 0-3: No chaining, single-step approach
- 4-6: Some staging but weak connections
- 7-8: Good multi-stage pipeline
- 9-10: Excellent chained processing with refinement`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Цепочка промптов',
      description: `
# Цепочка промптов

Создайте промпт, который использует вывод одного шага как ввод для следующего.

## Ваша цель

Создайте многоэтапный конвейер, который:
1. Разбивает сложную задачу на этапы
2. Каждый этап строится на предыдущем
3. Производит финальный отточенный вывод
`,
    },
    tr: {
      title: 'Prompt Zinciri',
      description: `
# Prompt Zinciri

Bir adımın çıktısını bir sonrakinin girdisi olarak kullanan bir prompt oluşturun.

## Hedefiniz

Şunları yapan çok aşamalı bir boru hattı tasarlayın:
1. Karmaşık görevi aşamalara böler
2. Her aşama bir öncekinin üzerine inşa eder
3. Son rafine çıktıyı üretir
`,
    },
  },
};
