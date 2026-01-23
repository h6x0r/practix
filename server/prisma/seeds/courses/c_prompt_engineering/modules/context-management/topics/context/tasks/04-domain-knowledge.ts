export default {
  slug: 'pe-domain-knowledge',
  title: 'Domain Expert',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Domain Expert

Create a prompt that provides domain-specific knowledge for specialized tasks.

## Your Goal

Design a prompt for a legal document assistant that includes:
1. Relevant legal terminology
2. Jurisdiction-specific rules
3. Document type conventions

## Requirements

Your prompt must:
- Provide necessary domain knowledge
- Include relevant terminology
- Specify applicable rules/standards
- Include \`{{INPUT}}\` for the document request

## Example Input
\`\`\`
Draft a non-disclosure agreement for a software consulting project.
\`\`\`

## Expected Output Style
- Uses correct legal terminology
- Follows document conventions
- Applies relevant rules
`,
  initialCode: `Create this document:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Draft a simple terms of service for a mobile app.',
        expectedCriteria: [
          'Uses appropriate legal terminology',
          'Follows document conventions',
          'Addresses key required sections',
          'Applies domain knowledge',
        ],
        rubric: 'Response should demonstrate domain-appropriate output using provided context.',
      },
      {
        input: 'Write a data processing agreement outline.',
        expectedCriteria: [
          'Uses correct terminology (GDPR, data controller, etc.)',
          'Follows standard structure',
          'Addresses required elements',
          'Shows domain expertise',
        ],
        rubric: 'Response should reflect domain knowledge and conventions.',
      },
    ],
    judgePrompt: `Evaluate the use of domain-specific knowledge.

Criteria:
1. Terminology - Is domain vocabulary used correctly?
2. Conventions - Are industry standards followed?
3. Accuracy - Is the domain knowledge applied correctly?
4. Completeness - Are key domain elements addressed?

Score 0-10:
- 0-3: Generic response ignoring domain specifics
- 4-6: Some domain awareness but inconsistent
- 7-8: Good domain knowledge application
- 9-10: Excellent domain expertise demonstrated`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Эксперт в предметной области',
      description: `
# Эксперт в предметной области

Создайте промпт, предоставляющий специализированные знания для профессиональных задач.

## Ваша цель

Создайте промпт для помощника по юридическим документам, который включает:
1. Релевантную юридическую терминологию
2. Правила, специфичные для юрисдикции
3. Конвенции типа документа
`,
    },
    tr: {
      title: 'Alan Uzmanı',
      description: `
# Alan Uzmanı

Özel görevler için alana özgü bilgi sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları içeren bir hukuki belge asistanı promptu tasarlayın:
1. İlgili hukuki terminoloji
2. Yargı yetkisine özgü kurallar
3. Belge türü kuralları
`,
    },
  },
};
