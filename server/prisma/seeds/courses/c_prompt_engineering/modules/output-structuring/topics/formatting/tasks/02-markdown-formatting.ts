export default {
  slug: 'pe-markdown-formatting',
  title: 'Beautiful Markdown',
  difficulty: 'easy' as const,
  estimatedTime: '10m',
  taskType: 'PROMPT' as const,
  description: `
# Beautiful Markdown

Create a prompt that produces well-structured markdown output.

## Your Goal

Design a prompt for a documentation generator that:
1. Uses proper markdown headings
2. Includes code blocks with language hints
3. Uses lists and emphasis appropriately

## Requirements

Your prompt must:
- Request markdown formatting
- Specify the structure to use
- Include \`{{INPUT}}\` for the topic to document

## Example Input
\`\`\`
Document the Array.map() method in JavaScript
\`\`\`

## Expected Output
Proper markdown with headings, code examples, and structured sections.
`,
  initialCode: `Create documentation for:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Document the Python list append() method.',
        expectedCriteria: [
          'Uses markdown headings',
          'Includes code blocks with language tag',
          'Has proper structure (description, usage, examples)',
          'Uses formatting like bold/italic appropriately',
        ],
        rubric: 'Response should be well-formatted markdown documentation.',
      },
      {
        input: 'Document the concept of async/await in JavaScript.',
        expectedCriteria: [
          'Proper markdown structure',
          'Code examples with syntax highlighting hint',
          'Clear sections and organization',
          'Uses markdown features effectively',
        ],
        rubric: 'Response should demonstrate good markdown formatting practices.',
      },
    ],
    judgePrompt: `Evaluate the markdown formatting quality.

Criteria:
1. Structure - Are headings and sections well-organized?
2. Code blocks - Are code examples properly formatted?
3. Readability - Is the markdown easy to read?
4. Completeness - Are all markdown features used appropriately?

Score 0-10:
- 0-3: Poor or no markdown formatting
- 4-6: Basic markdown but missing elements
- 7-8: Good markdown structure and formatting
- 9-10: Excellent, publication-quality markdown`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Красивый Markdown',
      description: `
# Красивый Markdown

Создайте промпт, который производит хорошо структурированный markdown вывод.

## Ваша цель

Создайте промпт для генератора документации, который:
1. Использует правильные заголовки markdown
2. Включает блоки кода с указанием языка
3. Использует списки и выделение надлежащим образом
`,
    },
    tr: {
      title: 'Güzel Markdown',
      description: `
# Güzel Markdown

İyi yapılandırılmış markdown çıktısı üreten bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir dokümantasyon üreteci promptu tasarlayın:
1. Uygun markdown başlıkları kullanır
2. Dil ipuçlarıyla kod blokları içerir
3. Listeleri ve vurguyu uygun şekilde kullanır
`,
    },
  },
};
