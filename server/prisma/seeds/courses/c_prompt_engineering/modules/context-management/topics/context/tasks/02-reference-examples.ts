export default {
  slug: 'pe-reference-examples',
  title: 'Lead by Example',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Lead by Example

Create a prompt that uses reference examples to guide the AI's output format and style.

## Your Goal

Design a prompt for a documentation writer that includes:
1. Example documentation to follow
2. Clear style guidelines from the example
3. Request for similar output

## Requirements

Your prompt must:
- Include a reference example
- Highlight what to mimic from the example
- Include \`{{INPUT}}\` for the function to document

## Example Input
\`\`\`
function validateEmail(email: string): boolean
\`\`\`

## Expected Output Style
- Matches the format of provided examples
- Follows the same conventions
- Maintains consistent style
`,
  initialCode: `Document this function:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'function formatCurrency(amount: number, currency: string): string',
        expectedCriteria: [
          'Follows any example format provided',
          'Includes similar sections as the example',
          'Maintains consistent style',
          'Documents parameters and return value',
        ],
        rubric: 'Response should follow the format and style of any reference examples provided.',
      },
      {
        input: 'function debounce(fn: Function, delay: number): Function',
        expectedCriteria: [
          'Matches example documentation style',
          'Includes expected sections',
          'Uses consistent terminology',
          'Provides useful documentation',
        ],
        rubric: 'Response should demonstrate learning from reference examples.',
      },
    ],
    judgePrompt: `Evaluate how well the response follows reference examples.

Criteria:
1. Format matching - Does it follow the example format?
2. Style consistency - Is the style similar to the example?
3. Convention adherence - Are the same conventions used?
4. Quality - Is the output useful and well-written?

Score 0-10:
- 0-3: Ignores example, uses different format
- 4-6: Partial format matching
- 7-8: Good example following with consistent style
- 9-10: Excellent example mimicry with high quality`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Покажи на примере',
      description: `
# Покажи на примере

Создайте промпт, использующий референсные примеры для направления формата и стиля вывода ИИ.

## Ваша цель

Создайте промпт для технического писателя, который включает:
1. Пример документации для подражания
2. Четкие стилевые рекомендации из примера
3. Запрос на аналогичный вывод
`,
    },
    tr: {
      title: 'Örnekle Göster',
      description: `
# Örnekle Göster

AI'ın çıktı formatını ve stilini yönlendirmek için referans örnekleri kullanan bir prompt oluşturun.

## Hedefiniz

Şunları içeren bir dokümantasyon yazarı promptu tasarlayın:
1. Takip edilecek örnek dokümantasyon
2. Örnekten net stil yönergeleri
3. Benzer çıktı talebi
`,
    },
  },
};
