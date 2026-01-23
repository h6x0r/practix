export default {
  slug: 'pe-json-output',
  title: 'Structured JSON',
  difficulty: 'easy' as const,
  estimatedTime: '10m',
  taskType: 'PROMPT' as const,
  description: `
# Structured JSON

Create a prompt that makes the AI return responses in valid JSON format.

## Your Goal

Design a prompt for a data extractor that:
1. Outputs valid JSON only
2. Uses a specific schema
3. Handles missing data gracefully

## Requirements

Your prompt must:
- Request JSON output explicitly
- Define the expected schema
- Include \`{{INPUT}}\` for the text to extract from

## Example Input
\`\`\`
John Smith, Software Engineer at Google, john.smith@gmail.com, based in San Francisco
\`\`\`

## Expected Output
\`\`\`json
{
  "name": "John Smith",
  "title": "Software Engineer",
  "company": "Google",
  "email": "john.smith@gmail.com",
  "location": "San Francisco"
}
\`\`\`
`,
  initialCode: `Extract information from this text:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Sarah Johnson, Product Manager at Amazon, sjohnson@amazon.com, New York',
        expectedCriteria: [
          'Returns valid JSON format',
          'Extracts name correctly',
          'Extracts job title and company',
          'Includes email and location',
        ],
        rubric: 'Response should be valid, parseable JSON with all expected fields.',
      },
      {
        input: 'Mike Chen - CTO - Startup Inc.',
        expectedCriteria: [
          'Returns valid JSON',
          'Extracts available fields',
          'Handles missing fields gracefully (null or omitted)',
          'Maintains consistent schema',
        ],
        rubric: 'Response should be valid JSON even with partial data.',
      },
    ],
    judgePrompt: `Evaluate the JSON output quality.

Criteria:
1. Validity - Is the output valid JSON?
2. Schema adherence - Does it follow expected structure?
3. Accuracy - Is the extracted data correct?
4. Completeness - Are all available fields extracted?

Score 0-10:
- 0-3: Invalid JSON or wrong format
- 4-6: Valid JSON but missing fields or wrong schema
- 7-8: Good JSON with correct schema
- 9-10: Perfect JSON extraction with all data`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Структурированный JSON',
      description: `
# Структурированный JSON

Создайте промпт, который заставит ИИ возвращать ответы в валидном формате JSON.

## Ваша цель

Создайте промпт для экстрактора данных, который:
1. Выводит только валидный JSON
2. Использует определенную схему
3. Корректно обрабатывает отсутствующие данные
`,
    },
    tr: {
      title: 'Yapılandırılmış JSON',
      description: `
# Yapılandırılmış JSON

AI'ın geçerli JSON formatında yanıt vermesini sağlayan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir veri çıkarıcı promptu tasarlayın:
1. Sadece geçerli JSON çıktısı verir
2. Belirli bir şema kullanır
3. Eksik verileri düzgün şekilde işler
`,
    },
  },
};
