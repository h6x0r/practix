export default {
  slug: 'pe-context-recall',
  title: 'Remember the Context',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Remember the Context

Create a prompt that effectively references and uses information from earlier in the conversation.

## Your Goal

Design a prompt for a project assistant that:
1. Tracks project details mentioned earlier
2. References previous decisions consistently
3. Builds on earlier discussions

## Requirements

Your prompt must:
- Establish how to track conversation context
- Reference prior information appropriately
- Maintain consistency across turns
- Include \`{{INPUT}}\` for the current request

## Example Context
Turn 1: "We're building a React app with TypeScript"
Turn 2: "The target users are enterprise customers"
Turn 3: [Current request referencing above]
`,
  initialCode: `Help with this project task:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Based on our earlier discussion about the React TypeScript app for enterprise users, what testing strategy should we use?',
        expectedCriteria: [
          'Acknowledges prior context',
          'Recommendations align with mentioned tech stack',
          'Considers enterprise requirements',
          'Maintains consistency with previous decisions',
        ],
        rubric: 'Response should demonstrate awareness of conversation context.',
      },
      {
        input: 'Given what we discussed about our tech stack and target audience, what deployment approach makes sense?',
        expectedCriteria: [
          'References prior context appropriately',
          'Recommendations are consistent',
          'Builds on earlier information',
          'Shows conversation continuity',
        ],
        rubric: 'Response should use context from earlier in conversation.',
      },
    ],
    judgePrompt: `Evaluate context recall ability.

Criteria:
1. Context awareness - Does it acknowledge prior information?
2. Consistency - Are responses consistent with earlier context?
3. Building - Does it build on previous discussions?
4. Relevance - Is the context used appropriately?

Score 0-10:
- 0-3: Ignores prior context
- 4-6: Some context awareness but inconsistent
- 7-8: Good context recall and consistency
- 9-10: Excellent context integration throughout`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Помни контекст',
      description: `
# Помни контекст

Создайте промпт, который эффективно ссылается и использует информацию из более ранней части разговора.

## Ваша цель

Создайте промпт для помощника по проектам, который:
1. Отслеживает детали проекта, упомянутые ранее
2. Последовательно ссылается на предыдущие решения
3. Развивает более ранние обсуждения
`,
    },
    tr: {
      title: 'Bağlamı Hatırla',
      description: `
# Bağlamı Hatırla

Konuşmanın önceki bölümlerindeki bilgileri etkili bir şekilde referans alan ve kullanan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir proje asistanı promptu tasarlayın:
1. Daha önce bahsedilen proje detaylarını takip eder
2. Önceki kararlara tutarlı şekilde referans verir
3. Önceki tartışmalar üzerine inşa eder
`,
    },
  },
};
