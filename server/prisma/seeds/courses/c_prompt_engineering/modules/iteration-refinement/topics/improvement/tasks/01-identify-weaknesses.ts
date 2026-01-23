export default {
  slug: 'pe-identify-weaknesses',
  title: 'Find the Gaps',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Find the Gaps

Create a prompt that identifies potential weaknesses in its own output.

## Your Goal

Design a prompt for a writing assistant that:
1. Produces the requested content
2. Then lists potential weaknesses
3. Suggests improvements

## Requirements

Your prompt must:
- Generate initial output
- Self-critique the output
- Propose specific improvements
- Include \`{{INPUT}}\` for the writing request

## Example Input
\`\`\`
Write a product launch email for our new fitness app.
\`\`\`

## Expected Output
- The email content
- List of potential issues
- Suggested improvements
`,
  initialCode: `Create content for:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Write a job posting for a software developer position.',
        expectedCriteria: [
          'Produces the requested job posting',
          'Identifies potential weaknesses',
          'Suggests specific improvements',
          'Shows self-awareness about limitations',
        ],
        rubric: 'Response should demonstrate self-critique and improvement suggestions.',
      },
      {
        input: 'Draft a customer apology email for a service outage.',
        expectedCriteria: [
          'Creates the apology email',
          'Notes potential issues or gaps',
          'Proposes enhancements',
          'Shows critical self-analysis',
        ],
        rubric: 'Response should include constructive self-criticism.',
      },
    ],
    judgePrompt: `Evaluate the self-critique quality.

Criteria:
1. Self-awareness - Does it identify real weaknesses?
2. Specificity - Are critiques specific and actionable?
3. Improvement focus - Are suggestions constructive?
4. Balance - Does it acknowledge both strengths and weaknesses?

Score 0-10:
- 0-3: No self-critique or irrelevant critique
- 4-6: Some self-awareness but vague
- 7-8: Good identification of weaknesses with suggestions
- 9-10: Excellent self-critique with actionable improvements`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Найди пробелы',
      description: `
# Найди пробелы

Создайте промпт, который выявляет потенциальные слабости в собственном выводе.

## Ваша цель

Создайте промпт для помощника по написанию, который:
1. Создает запрошенный контент
2. Затем перечисляет потенциальные слабости
3. Предлагает улучшения
`,
    },
    tr: {
      title: 'Boşlukları Bul',
      description: `
# Boşlukları Bul

Kendi çıktısındaki potansiyel zayıflıkları tespit eden bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir yazma asistanı promptu tasarlayın:
1. İstenen içeriği üretir
2. Ardından potansiyel zayıflıkları listeler
3. İyileştirmeler önerir
`,
    },
  },
};
