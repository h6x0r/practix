export default {
  slug: 'pe-handle-edge-cases',
  title: 'Edge Case Handling',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Edge Case Handling

Create a prompt that gracefully handles unusual or edge-case inputs.

## Your Goal

Design a prompt for a data validator that:
1. Handles valid inputs correctly
2. Gracefully manages invalid inputs
3. Provides helpful error messages
4. Doesn't break on unexpected data

## Requirements

Your prompt must:
- Define expected input format
- Handle edge cases explicitly
- Provide informative responses for errors
- Include \`{{INPUT}}\` for the data to validate

## Example Edge Cases
- Empty input
- Wrong data type
- Malformed data
- Extremely long input
`,
  initialCode: `Validate this input:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: '',
        expectedCriteria: [
          'Handles empty input gracefully',
          'Provides helpful feedback',
          'Does not produce errors',
          'Explains what was expected',
        ],
        rubric: 'Response should gracefully handle the empty input edge case.',
      },
      {
        input: 'This is definitely not a valid email address!!!',
        expectedCriteria: [
          'Recognizes invalid format',
          'Explains what is wrong',
          'Suggests correct format',
          'Remains helpful and professional',
        ],
        rubric: 'Response should handle invalid input with helpful feedback.',
      },
    ],
    judgePrompt: `Evaluate edge case handling.

Criteria:
1. Graceful handling - Does it handle edge cases without breaking?
2. Helpfulness - Are error messages informative?
3. Robustness - Does it work for various edge cases?
4. User guidance - Does it help users correct issues?

Score 0-10:
- 0-3: Breaks on edge cases or unhelpful
- 4-6: Handles some edge cases but not all
- 7-8: Good edge case handling with helpful messages
- 9-10: Excellent robustness with great user guidance`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Обработка граничных случаев',
      description: `
# Обработка граничных случаев

Создайте промпт, который корректно обрабатывает необычные или граничные входные данные.

## Ваша цель

Создайте промпт для валидатора данных, который:
1. Правильно обрабатывает валидные входные данные
2. Корректно управляет невалидными данными
3. Предоставляет полезные сообщения об ошибках
`,
    },
    tr: {
      title: 'Uç Durum İşleme',
      description: `
# Uç Durum İşleme

Olağandışı veya uç durum girdilerini zarif bir şekilde işleyen bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir veri doğrulayıcı promptu tasarlayın:
1. Geçerli girdileri doğru şekilde işler
2. Geçersiz girdileri zarif bir şekilde yönetir
3. Yararlı hata mesajları sağlar
`,
    },
  },
};
