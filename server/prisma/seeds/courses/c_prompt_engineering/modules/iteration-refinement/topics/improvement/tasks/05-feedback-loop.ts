export default {
  slug: 'pe-feedback-loop',
  title: 'Learn from Feedback',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Learn from Feedback

Create a prompt that incorporates feedback to improve its output.

## Your Goal

Design a prompt for a writing assistant that:
1. Produces initial content
2. Accepts user feedback
3. Revises based on feedback
4. Shows what was changed and why

## Requirements

Your prompt must:
- Produce initial output
- Request feedback
- Explain how it will incorporate changes
- Include \`{{INPUT}}\` for the writing task

## Feedback Examples
- "Make it more formal"
- "Add more examples"
- "Too long, shorten it"
`,
  initialCode: `Create content for:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Write a short bio for a LinkedIn profile.',
        expectedCriteria: [
          'Produces initial content',
          'Shows willingness to accept feedback',
          'Explains revision approach',
          'Demonstrates adaptability',
        ],
        rubric: 'Response should demonstrate feedback-friendly approach.',
      },
      {
        input: 'Draft an introduction for a presentation.',
        expectedCriteria: [
          'Creates initial draft',
          'Invites or acknowledges feedback',
          'Shows how changes would be made',
          'Flexible and adaptable tone',
        ],
        rubric: 'Response should be open to iteration and improvement.',
      },
    ],
    judgePrompt: `Evaluate the feedback integration approach.

Criteria:
1. Openness - Is the response open to feedback?
2. Adaptability - Does it show how it would incorporate changes?
3. Clarity - Is the revision process clear?
4. Iteration - Does it support iterative improvement?

Score 0-10:
- 0-3: Closed to feedback, no revision support
- 4-6: Some openness but unclear process
- 7-8: Good feedback integration approach
- 9-10: Excellent iterative improvement support`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Учись на обратной связи',
      description: `
# Учись на обратной связи

Создайте промпт, который учитывает обратную связь для улучшения своего вывода.

## Ваша цель

Создайте промпт для помощника по написанию, который:
1. Создает начальный контент
2. Принимает обратную связь от пользователя
3. Вносит исправления на основе обратной связи
4. Показывает, что было изменено и почему
`,
    },
    tr: {
      title: 'Geri Bildirimden Öğren',
      description: `
# Geri Bildirimden Öğren

Çıktısını iyileştirmek için geri bildirimi dahil eden bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir yazma asistanı promptu tasarlayın:
1. İlk içeriği üretir
2. Kullanıcı geri bildirimini kabul eder
3. Geri bildirime göre revize eder
4. Neyin değiştirildiğini ve neden değiştirildiğini gösterir
`,
    },
  },
};
