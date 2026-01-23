export default {
  slug: 'pe-conversation-summary',
  title: 'Summarize Progress',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Summarize Progress

Create a prompt that can summarize conversation progress and key decisions.

## Your Goal

Design a prompt for a meeting assistant that:
1. Tracks key points discussed
2. Notes decisions made
3. Provides clear summaries on request

## Requirements

Your prompt must:
- Capture important information
- Organize it logically
- Provide concise summaries
- Include \`{{INPUT}}\` for the summary request

## Summary Elements
- Topics covered
- Decisions made
- Action items
- Open questions
`,
  initialCode: `Summarize our discussion:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Can you give me a summary of what we have discussed so far about the project architecture?',
        expectedCriteria: [
          'Provides organized summary',
          'Captures key points',
          'Notes important decisions',
          'Structured and clear',
        ],
        rubric: 'Response should provide a clear, organized summary.',
      },
      {
        input: 'What action items do we have from this conversation?',
        expectedCriteria: [
          'Extracts action items',
          'Lists them clearly',
          'Organized by priority or topic',
          'Actionable and specific',
        ],
        rubric: 'Response should summarize actionable items effectively.',
      },
    ],
    judgePrompt: `Evaluate summarization quality.

Criteria:
1. Completeness - Are key points captured?
2. Organization - Is the summary well-structured?
3. Clarity - Is it easy to understand?
4. Usefulness - Is it actionable and helpful?

Score 0-10:
- 0-3: Poor or missing summary
- 4-6: Basic summary but missing elements
- 7-8: Good comprehensive summary
- 9-10: Excellent summary with all key elements`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Подведи итоги',
      description: `
# Подведи итоги

Создайте промпт, который может резюмировать прогресс разговора и ключевые решения.

## Ваша цель

Создайте промпт для помощника по совещаниям, который:
1. Отслеживает ключевые обсужденные моменты
2. Отмечает принятые решения
3. Предоставляет четкие резюме по запросу
`,
    },
    tr: {
      title: 'İlerlemeyi Özetle',
      description: `
# İlerlemeyi Özetle

Konuşma ilerlemesini ve temel kararları özetleyebilen bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir toplantı asistanı promptu tasarlayın:
1. Tartışılan ana noktaları takip eder
2. Alınan kararları not eder
3. İstek üzerine net özetler sağlar
`,
    },
  },
};
