export default {
  slug: 'pe-custom-templates',
  title: 'Custom Templates',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Custom Templates

Create a prompt that makes the AI follow a specific output template.

## Your Goal

Design a prompt for a report generator that:
1. Follows a predefined template exactly
2. Fills in sections with appropriate content
3. Maintains template structure consistently

## Requirements

Your prompt must:
- Define a clear template structure
- Specify what goes in each section
- Include \`{{INPUT}}\` for the report topic

## Example Template
\`\`\`
## Executive Summary
[2-3 sentence overview]

## Key Findings
- Finding 1
- Finding 2
- Finding 3

## Recommendations
1. [Action item]
2. [Action item]

## Next Steps
[Paragraph describing next steps]
\`\`\`

## Expected Output
A report that exactly follows the template structure.
`,
  initialCode: `Create a report about:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Analyze our website performance over the last quarter.',
        expectedCriteria: [
          'Follows template structure exactly',
          'All sections are present',
          'Content fits each section appropriately',
          'Template format is maintained',
        ],
        rubric: 'Response should strictly follow the defined template structure.',
      },
      {
        input: 'Review the onboarding process for new employees.',
        expectedCriteria: [
          'Adheres to template format',
          'Sections filled with relevant content',
          'Structure is consistent with template',
          'All required elements present',
        ],
        rubric: 'Response should demonstrate template adherence.',
      },
    ],
    judgePrompt: `Evaluate template adherence.

Criteria:
1. Structure - Does output match the template exactly?
2. Sections - Are all template sections present?
3. Content fit - Is content appropriate for each section?
4. Consistency - Is the format maintained throughout?

Score 0-10:
- 0-3: Ignores template, uses different structure
- 4-6: Partially follows template
- 7-8: Good template adherence
- 9-10: Perfect template following with quality content`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Пользовательские шаблоны',
      description: `
# Пользовательские шаблоны

Создайте промпт, который заставит ИИ следовать определенному шаблону вывода.

## Ваша цель

Создайте промпт для генератора отчетов, который:
1. Точно следует заранее определенному шаблону
2. Заполняет разделы соответствующим содержимым
3. Последовательно поддерживает структуру шаблона
`,
    },
    tr: {
      title: 'Özel Şablonlar',
      description: `
# Özel Şablonlar

AI'ın belirli bir çıktı şablonunu takip etmesini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir rapor üreteci promptu tasarlayın:
1. Önceden tanımlanmış şablonu tam olarak takip eder
2. Bölümleri uygun içerikle doldurur
3. Şablon yapısını tutarlı bir şekilde korur
`,
    },
  },
};
