export default {
  slug: 'pe-multi-perspective',
  title: 'Multiple Perspectives',
  difficulty: 'medium' as const,
  estimatedTime: '15m',
  taskType: 'PROMPT' as const,
  description: `
# Multiple Perspectives

Create a prompt that makes the AI analyze a decision from **multiple professional perspectives**.

## Your Goal

Design a prompt that generates analysis from different viewpoints:
1. A product manager focused on user value
2. A developer focused on technical feasibility
3. A business analyst focused on ROI

## Requirements

Your prompt must:
- Define all three personas clearly
- Ask for distinct perspectives on the same topic
- Ensure each viewpoint is authentic to its role
- Include \`{{INPUT}}\` for the decision or feature

## Example Input
\`\`\`
Should we build a mobile app or improve our responsive web experience?
\`\`\`

## Expected Output Style
- Three clearly labeled sections
- Different priorities for each role
- Authentic concerns for each perspective
`,
  initialCode: `Analyze this from multiple perspectives.

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Should we adopt a new cloud provider to reduce costs by 30% but require 3 months of migration?',
        expectedCriteria: [
          'Provides at least 2-3 distinct perspectives',
          'Product/user perspective considers impact',
          'Technical perspective addresses migration complexity',
          'Business perspective weighs costs vs benefits',
        ],
        rubric: 'Response should provide authentic multi-perspective analysis of the cloud migration decision.',
      },
      {
        input: 'Should we implement real-time notifications using WebSockets?',
        expectedCriteria: [
          'Shows multiple viewpoints',
          'Technical view discusses implementation',
          'Product view considers user experience',
          'Business view addresses resource investment',
        ],
        rubric: 'Response should cover the WebSocket decision from different professional angles.',
      },
    ],
    judgePrompt: `Evaluate this multi-perspective analysis.

Criteria:
1. Distinctiveness - Are the perspectives clearly different?
2. Authenticity - Does each role sound genuine to its profession?
3. Completeness - Are key concerns from each role addressed?
4. Balance - Is the analysis fair across perspectives?

Score 0-10:
- 0-3: Single perspective or very similar viewpoints
- 4-6: Multiple views but lacking depth or authenticity
- 7-8: Good distinct perspectives with authentic concerns
- 9-10: Excellent multi-faceted analysis with deep insights from each role`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Множественные перспективы',
      description: `
# Множественные перспективы

Создайте промпт, который заставит ИИ анализировать решение с **нескольких профессиональных точек зрения**.

## Ваша цель

Создайте промпт, генерирующий анализ с разных позиций:
1. Продакт-менеджер, фокусирующийся на ценности для пользователя
2. Разработчик, фокусирующийся на технической осуществимости
3. Бизнес-аналитик, фокусирующийся на ROI
`,
    },
    tr: {
      title: 'Çoklu Perspektifler',
      description: `
# Çoklu Perspektifler

AI'ın bir kararı **birden fazla profesyonel perspektiften** analiz etmesini sağlayan bir prompt oluşturun.

## Hedefiniz

Farklı bakış açılarından analiz üreten bir prompt tasarlayın:
1. Kullanıcı değerine odaklanan ürün yöneticisi
2. Teknik fizibiliteye odaklanan geliştirici
3. ROI'ye odaklanan iş analisti
`,
    },
  },
};
