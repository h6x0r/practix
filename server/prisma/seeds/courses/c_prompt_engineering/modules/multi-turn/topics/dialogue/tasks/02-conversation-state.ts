export default {
  slug: 'pe-conversation-state',
  title: 'Track the State',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Track the State

Create a prompt that maintains and tracks conversation state across multiple turns.

## Your Goal

Design a prompt for a task manager that:
1. Tracks what has been completed
2. Knows what is pending
3. Updates state as tasks progress

## Requirements

Your prompt must:
- Define a state tracking mechanism
- Update state based on conversation
- Report current state accurately
- Include \`{{INPUT}}\` for the status update

## State Example
\`\`\`
Current State:
- ✅ Requirement gathering - DONE
- 🔄 Design phase - IN PROGRESS
- ⏳ Development - PENDING
\`\`\`
`,
  initialCode: `Update task status:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'We just finished the design phase. What is our current project status?',
        expectedCriteria: [
          'Updates state appropriately',
          'Shows clear status tracking',
          'Reflects the change',
          'Provides accurate current state',
        ],
        rubric: 'Response should demonstrate state tracking and updating.',
      },
      {
        input: 'Development is now 50% complete. Please update our status.',
        expectedCriteria: [
          'Tracks progress accurately',
          'Updates relevant state',
          'Shows current overall status',
          'Maintains state consistency',
        ],
        rubric: 'Response should reflect state changes accurately.',
      },
    ],
    judgePrompt: `Evaluate state tracking ability.

Criteria:
1. Tracking - Is state tracked accurately?
2. Updates - Are state changes reflected correctly?
3. Clarity - Is the current state clear?
4. Consistency - Is state maintained properly?

Score 0-10:
- 0-3: No state tracking or wrong states
- 4-6: Some tracking but inconsistent
- 7-8: Good state management
- 9-10: Excellent stateful conversation handling`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Отслеживай состояние',
      description: `
# Отслеживай состояние

Создайте промпт, который поддерживает и отслеживает состояние разговора на протяжении нескольких ходов.

## Ваша цель

Создайте промпт для менеджера задач, который:
1. Отслеживает, что было выполнено
2. Знает, что ожидает выполнения
3. Обновляет состояние по мере продвижения задач
`,
    },
    tr: {
      title: 'Durumu Takip Et',
      description: `
# Durumu Takip Et

Birden fazla turda konuşma durumunu koruyan ve takip eden bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir görev yöneticisi promptu tasarlayın:
1. Tamamlananları takip eder
2. Bekleyenleri bilir
3. Görevler ilerledikçe durumu günceller
`,
    },
  },
};
