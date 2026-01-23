export default {
  slug: 'pe-clarifying-questions',
  title: 'Ask Before Acting',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Ask Before Acting

Create a prompt that asks clarifying questions when the request is ambiguous.

## Your Goal

Design a prompt for a project planner that:
1. Identifies ambiguous requests
2. Asks targeted clarifying questions
3. Proceeds only when requirements are clear

## Requirements

Your prompt must:
- Detect when more info is needed
- Ask specific, targeted questions
- Not assume or guess
- Include \`{{INPUT}}\` for the project request

## Example Input
\`\`\`
Build me a website.
\`\`\`

## Expected Behavior
Instead of guessing, ask about:
- Purpose of the website
- Target audience
- Key features needed
- Design preferences
`,
  initialCode: `Help me with this project:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Create an app for my business.',
        expectedCriteria: [
          'Identifies the ambiguity',
          'Asks clarifying questions',
          'Questions are specific and targeted',
          'Does not make assumptions',
        ],
        rubric: 'Response should ask clarifying questions rather than guessing.',
      },
      {
        input: 'I need a database.',
        expectedCriteria: [
          'Recognizes insufficient information',
          'Asks about use case and requirements',
          'Questions are relevant and helpful',
          'Guides toward better specification',
        ],
        rubric: 'Response should seek clarification on the vague request.',
      },
    ],
    judgePrompt: `Evaluate the clarification approach.

Criteria:
1. Detection - Does it identify ambiguity correctly?
2. Questions - Are clarifying questions targeted and useful?
3. Non-assumption - Does it avoid guessing?
4. Helpfulness - Do questions guide toward better requirements?

Score 0-10:
- 0-3: Makes assumptions without asking
- 4-6: Some questions but misses key ambiguities
- 7-8: Good clarifying questions
- 9-10: Excellent targeted questions that guide specification`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Спроси перед действием',
      description: `
# Спроси перед действием

Создайте промпт, который задает уточняющие вопросы, когда запрос неоднозначен.

## Ваша цель

Создайте промпт для планировщика проектов, который:
1. Выявляет неоднозначные запросы
2. Задает целевые уточняющие вопросы
3. Действует только когда требования ясны
`,
    },
    tr: {
      title: 'Harekete Geçmeden Önce Sor',
      description: `
# Harekete Geçmeden Önce Sor

Talep belirsiz olduğunda açıklayıcı sorular soran bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir proje planlayıcısı promptu tasarlayın:
1. Belirsiz talepleri tespit eder
2. Hedefli açıklayıcı sorular sorar
3. Sadece gereksinimler netleştiğinde ilerler
`,
    },
  },
};
