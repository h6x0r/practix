export default {
  slug: 'pe-self-verification',
  title: 'Verify Your Answer',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Verify Your Answer

Create a prompt that makes the AI check its own work before giving a final answer.

## Your Goal

Design a prompt for a code reviewer that:
1. Provides an initial analysis
2. Reviews its own analysis for errors
3. Corrects any mistakes found
4. Gives a verified final answer

## Requirements

Your prompt must:
- Ask for initial analysis
- Instruct to verify the analysis
- Include \`{{INPUT}}\` for the code to review

## Example Input
\`\`\`python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
\`\`\`

## Expected Output Style
- Initial assessment
- Self-verification check
- Final verified conclusion
`,
  initialCode: `Review this code:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: `function isPalindrome(str) {
  return str === str.split('').reverse().join('');
}`,
        expectedCriteria: [
          'Provides initial analysis',
          'Performs self-verification or double-check',
          'Considers edge cases (case sensitivity, spaces)',
          'Gives verified final assessment',
        ],
        rubric: 'Response should show initial analysis followed by self-verification before final answer.',
      },
      {
        input: `def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1`,
        expectedCriteria: [
          'Initial code analysis performed',
          'Self-verification or re-checking done',
          'Identifies the potential infinite loop bug',
          'Provides verified conclusion',
        ],
        rubric: 'Response should demonstrate self-verification and catch the bug in the binary search.',
      },
    ],
    judgePrompt: `Evaluate if the response includes self-verification.

Criteria:
1. Initial analysis - Is there a first-pass assessment?
2. Verification step - Does it check its own work?
3. Correction - Are any errors found and corrected?
4. Final answer - Is there a verified conclusion?

Score 0-10:
- 0-3: No verification, just single-pass analysis
- 4-6: Some verification but incomplete
- 7-8: Good self-verification with corrections
- 9-10: Excellent verification process with clear corrections`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Проверь свой ответ',
      description: `
# Проверь свой ответ

Создайте промпт, который заставит ИИ проверять свою работу перед выдачей финального ответа.

## Ваша цель

Создайте промпт для код-ревьюера, который:
1. Предоставляет первоначальный анализ
2. Проверяет свой анализ на ошибки
3. Исправляет найденные ошибки
4. Дает проверенный финальный ответ
`,
    },
    tr: {
      title: 'Cevabını Doğrula',
      description: `
# Cevabını Doğrula

AI'ın son cevabı vermeden önce kendi çalışmasını kontrol etmesini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir kod inceleyicisi promptu tasarlayın:
1. İlk analizi sağlar
2. Kendi analizini hatalar için gözden geçirir
3. Bulunan hataları düzeltir
4. Doğrulanmış son cevabı verir
`,
    },
  },
};
