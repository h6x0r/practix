export default {
  slug: 'pe-length-control',
  title: 'Control the Length',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Control the Length

Create a prompt that specifies and enforces output length constraints.

## Your Goal

Design a prompt for a summary generator that:
1. Produces summaries of specific length
2. Stays within word/character limits
3. Maintains quality despite constraints

## Requirements

Your prompt must:
- Specify exact length requirements
- Enforce the constraint clearly
- Include \`{{INPUT}}\` for the text to summarize

## Example Input
\`\`\`
[A long article about climate change...]
\`\`\`

## Expected Output
A summary that is exactly the specified length.
`,
  initialCode: `Summarize this text:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions. Applications include image recognition, natural language processing, and recommendation systems.',
        expectedCriteria: [
          'Summary respects length constraints if specified',
          'Captures main points despite brevity',
          'Is coherent and well-written',
          'Stays focused on key information',
        ],
        rubric: 'Response should demonstrate adherence to length constraints while maintaining quality.',
      },
      {
        input: 'Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies, ensuring consistency across environments. This solves the "it works on my machine" problem and enables efficient resource utilization compared to traditional virtual machines.',
        expectedCriteria: [
          'Appropriate length based on constraints',
          'Key concepts preserved',
          'Concise but complete',
          'No unnecessary padding',
        ],
        rubric: 'Response should be appropriately sized while capturing essential content.',
      },
    ],
    judgePrompt: `Evaluate length control and quality.

Criteria:
1. Length adherence - Does output meet specified constraints?
2. Content quality - Is essential information preserved?
3. Conciseness - Is there unnecessary padding?
4. Coherence - Is the output well-written despite constraints?

Score 0-10:
- 0-3: Wrong length or very poor quality
- 4-6: Partially meets length, some quality issues
- 7-8: Good length control with quality content
- 9-10: Perfect length with excellent quality`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Контроль длины',
      description: `
# Контроль длины

Создайте промпт, который задает и обеспечивает ограничения длины вывода.

## Ваша цель

Создайте промпт для генератора резюме, который:
1. Создает резюме определенной длины
2. Соблюдает лимиты слов/символов
3. Сохраняет качество несмотря на ограничения
`,
    },
    tr: {
      title: 'Uzunluk Kontrolü',
      description: `
# Uzunluk Kontrolü

Çıktı uzunluk kısıtlamalarını belirleyen ve uygulayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir özet üreteci promptu tasarlayın:
1. Belirli uzunlukta özetler üretir
2. Kelime/karakter sınırlarına uyar
3. Kısıtlamalara rağmen kaliteyi korur
`,
    },
  },
};
