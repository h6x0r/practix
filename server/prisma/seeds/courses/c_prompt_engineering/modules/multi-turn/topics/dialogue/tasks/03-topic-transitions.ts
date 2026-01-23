export default {
  slug: 'pe-topic-transitions',
  title: 'Smooth Transitions',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Smooth Transitions

Create a prompt that handles topic transitions gracefully in conversation.

## Your Goal

Design a prompt for a learning assistant that:
1. Recognizes when topics are changing
2. Transitions smoothly between subjects
3. Connects new topics to previous ones when relevant

## Requirements

Your prompt must:
- Acknowledge topic changes
- Transition naturally
- Maintain conversation flow
- Include \`{{INPUT}}\` for the new topic/question

## Transition Types
- Related topic shift
- Complete topic change
- Return to earlier topic
`,
  initialCode: `Continue the conversation about:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Speaking of React, can we switch to discussing backend technologies now?',
        expectedCriteria: [
          'Acknowledges the topic shift',
          'Transitions smoothly',
          'Does not abruptly change',
          'Maintains conversational flow',
        ],
        rubric: 'Response should handle the topic transition gracefully.',
      },
      {
        input: 'Going back to what we discussed about databases earlier, I have a follow-up question.',
        expectedCriteria: [
          'Recognizes return to earlier topic',
          'Connects to previous discussion',
          'Transitions naturally',
          'Shows context awareness',
        ],
        rubric: 'Response should smoothly return to earlier topic.',
      },
    ],
    judgePrompt: `Evaluate topic transition handling.

Criteria:
1. Recognition - Does it recognize topic changes?
2. Smoothness - Are transitions natural?
3. Connection - Are topics connected when relevant?
4. Flow - Is conversation flow maintained?

Score 0-10:
- 0-3: Abrupt or jarring transitions
- 4-6: Some transition awareness but awkward
- 7-8: Good smooth transitions
- 9-10: Excellent natural topic flow`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Плавные переходы',
      description: `
# Плавные переходы

Создайте промпт, который корректно обрабатывает переходы между темами в разговоре.

## Ваша цель

Создайте промпт для помощника по обучению, который:
1. Распознает смену тем
2. Плавно переходит между предметами
3. Связывает новые темы с предыдущими, когда это уместно
`,
    },
    tr: {
      title: 'Akıcı Geçişler',
      description: `
# Akıcı Geçişler

Konuşmada konu geçişlerini zarif bir şekilde ele alan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir öğrenme asistanı promptu tasarlayın:
1. Konuların ne zaman değiştiğini tanır
2. Konular arasında akıcı geçiş yapar
3. Yeni konuları ilgili olduğunda öncekilerle bağlar
`,
    },
  },
};
