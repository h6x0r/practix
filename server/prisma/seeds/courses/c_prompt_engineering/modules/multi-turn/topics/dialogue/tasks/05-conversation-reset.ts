export default {
  slug: 'pe-conversation-reset',
  title: 'Start Fresh',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Start Fresh

Create a prompt that can gracefully reset or restart a conversation when needed.

## Your Goal

Design a prompt for a support assistant that:
1. Recognizes when a fresh start is needed
2. Resets context appropriately
3. Preserves useful information when requested

## Requirements

Your prompt must:
- Detect reset requests
- Clear context appropriately
- Optionally preserve key information
- Include \`{{INPUT}}\` for the reset request

## Reset Scenarios
- Complete reset: Start over entirely
- Partial reset: Keep some context
- Topic reset: New topic, same preferences
`,
  initialCode: `Handle this conversation request:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Let us start over. I want to explain my requirements from scratch.',
        expectedCriteria: [
          'Recognizes reset request',
          'Acknowledges starting fresh',
          'Clears previous context',
          'Ready for new information',
        ],
        rubric: 'Response should handle the reset request gracefully.',
      },
      {
        input: 'Keep my preferences but let us discuss a completely new project.',
        expectedCriteria: [
          'Recognizes partial reset',
          'Preserves stated preferences',
          'Clears project-specific context',
          'Ready for new project discussion',
        ],
        rubric: 'Response should handle selective context reset.',
      },
    ],
    judgePrompt: `Evaluate conversation reset handling.

Criteria:
1. Recognition - Does it detect reset requests?
2. Appropriateness - Is the reset level correct?
3. Preservation - Is useful info kept when requested?
4. Readiness - Is it prepared for new conversation?

Score 0-10:
- 0-3: Does not handle resets or loses needed info
- 4-6: Basic reset but imprecise
- 7-8: Good reset handling
- 9-10: Excellent nuanced reset management`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Начни сначала',
      description: `
# Начни сначала

Создайте промпт, который может корректно сбросить или перезапустить разговор при необходимости.

## Ваша цель

Создайте промпт для помощника поддержки, который:
1. Распознает, когда нужно начать с чистого листа
2. Сбрасывает контекст надлежащим образом
3. Сохраняет полезную информацию по запросу
`,
    },
    tr: {
      title: 'Yeni Başla',
      description: `
# Yeni Başla

Gerektiğinde bir konuşmayı zarif bir şekilde sıfırlayabilen veya yeniden başlatabilen bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir destek asistanı promptu tasarlayın:
1. Yeni başlangıç gerektiğini tanır
2. Bağlamı uygun şekilde sıfırlar
3. İstendiğinde yararlı bilgileri korur
`,
    },
  },
};
