export default {
  slug: 'pe-context-compression',
  title: 'Less is More',
  difficulty: 'hard' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Less is More

Create a prompt that provides maximum context with minimum tokens.

## Your Goal

Design a concise prompt for a code helper that:
1. Provides essential context only
2. Uses efficient formatting
3. Avoids redundant information
4. Maximizes signal-to-noise ratio

## Requirements

Your prompt must:
- Be concise but complete
- Use efficient structuring
- Include only relevant context
- Include \`{{INPUT}}\` for the code question

## Example Input
\`\`\`
Why doesn't my React component re-render when I update the state?
\`\`\`

## Expected Output Style
- Focused response
- No unnecessary preamble
- Direct and efficient
`,
  initialCode: `Help with this code issue:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'My Python script is slow when processing a large CSV file. How can I optimize it?',
        expectedCriteria: [
          'Response is focused and efficient',
          'Provides relevant solutions',
          'Avoids unnecessary filler',
          'Gets to the point quickly',
        ],
        rubric: 'Response should be concise while still being helpful and complete.',
      },
      {
        input: 'Getting a null pointer exception in my Java code. What could cause this?',
        expectedCriteria: [
          'Direct and focused response',
          'Lists common causes efficiently',
          'Minimal redundancy',
          'High information density',
        ],
        rubric: 'Response should maximize useful information per word.',
      },
    ],
    judgePrompt: `Evaluate the efficiency of context usage.

Criteria:
1. Conciseness - Is the response efficiently worded?
2. Relevance - Is all content relevant and useful?
3. Completeness - Despite brevity, is it complete?
4. Signal-to-noise - Is the information density high?

Score 0-10:
- 0-3: Verbose, redundant, low information density
- 4-6: Some efficiency but could be more concise
- 7-8: Good balance of conciseness and completeness
- 9-10: Excellent efficiency with maximum value per token`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Меньше — значит больше',
      description: `
# Меньше — значит больше

Создайте промпт, который предоставляет максимум контекста при минимуме токенов.

## Ваша цель

Создайте лаконичный промпт для помощника по коду, который:
1. Предоставляет только необходимый контекст
2. Использует эффективное форматирование
3. Избегает избыточной информации
4. Максимизирует соотношение сигнал/шум
`,
    },
    tr: {
      title: 'Az Çoktur',
      description: `
# Az Çoktur

Minimum tokenle maksimum bağlam sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan özlü bir kod yardımcısı promptu tasarlayın:
1. Sadece temel bağlamı sağlar
2. Verimli biçimlendirme kullanır
3. Gereksiz bilgilerden kaçınır
4. Sinyal/gürültü oranını maksimize eder
`,
    },
  },
};
