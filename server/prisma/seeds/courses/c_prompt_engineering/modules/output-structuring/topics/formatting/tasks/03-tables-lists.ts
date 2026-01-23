export default {
  slug: 'pe-tables-lists',
  title: 'Tables and Lists',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Tables and Lists

Create a prompt that organizes information into tables or structured lists.

## Your Goal

Design a prompt for a comparison tool that:
1. Creates clear comparison tables
2. Uses aligned columns
3. Highlights key differences

## Requirements

Your prompt must:
- Request tabular format
- Specify comparison criteria
- Include \`{{INPUT}}\` for items to compare

## Example Input
\`\`\`
Compare React, Vue, and Angular for building web apps.
\`\`\`

## Expected Output
A markdown table with clear columns comparing features.
`,
  initialCode: `Compare these items:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Compare Python, JavaScript, and Java for backend development.',
        expectedCriteria: [
          'Creates a comparison table',
          'Has clear column headers',
          'Covers relevant comparison criteria',
          'Information is accurate and balanced',
        ],
        rubric: 'Response should present a clear, well-structured comparison table.',
      },
      {
        input: 'Compare REST API vs GraphQL for building APIs.',
        expectedCriteria: [
          'Uses table or structured list format',
          'Compares key aspects fairly',
          'Organized and easy to scan',
          'Provides useful comparison criteria',
        ],
        rubric: 'Response should organize comparison in a scannable format.',
      },
    ],
    judgePrompt: `Evaluate the table/list organization quality.

Criteria:
1. Format - Is data organized in a clear table or list?
2. Completeness - Are comparison criteria comprehensive?
3. Clarity - Is the comparison easy to understand?
4. Balance - Is the comparison fair and objective?

Score 0-10:
- 0-3: Unstructured, no table/list format
- 4-6: Basic structure but incomplete
- 7-8: Good organized comparison
- 9-10: Excellent, publication-ready comparison table`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Таблицы и списки',
      description: `
# Таблицы и списки

Создайте промпт, который организует информацию в таблицы или структурированные списки.

## Ваша цель

Создайте промпт для инструмента сравнения, который:
1. Создает понятные таблицы сравнения
2. Использует выровненные столбцы
3. Выделяет ключевые различия
`,
    },
    tr: {
      title: 'Tablolar ve Listeler',
      description: `
# Tablolar ve Listeler

Bilgileri tablolar veya yapılandırılmış listeler halinde organize eden bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir karşılaştırma aracı promptu tasarlayın:
1. Net karşılaştırma tabloları oluşturur
2. Hizalı sütunlar kullanır
3. Temel farklılıkları vurgular
`,
    },
  },
};
