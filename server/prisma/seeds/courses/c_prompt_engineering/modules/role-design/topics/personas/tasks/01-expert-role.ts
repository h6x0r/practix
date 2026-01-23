export default {
  slug: 'pe-expert-role',
  title: 'The Expert Advisor',
  difficulty: 'easy' as const,
  estimatedTime: '10m',
  taskType: 'PROMPT' as const,
  description: `
# The Expert Advisor

Create a prompt that makes the AI act as a **senior software architect** who can explain complex system design concepts.

## Your Goal

Design a prompt that:
1. Establishes the AI as an expert software architect with 15+ years of experience
2. Makes it explain concepts clearly to junior developers
3. Uses real-world examples and analogies

## Requirements

Your prompt must include:
- A clear role definition with expertise level
- Instructions for the explanation style
- A \`{{INPUT}}\` placeholder for the topic to explain

## Example Input
\`\`\`
microservices vs monolith
\`\`\`

## Expected Output Style
The AI should provide:
- Professional yet accessible explanations
- Practical trade-offs and considerations
- Real-world scenarios and examples
`,
  initialCode: `You are a senior software architect with 15+ years of experience.

{{INPUT}}

Explain this concept clearly.`,
  promptConfig: {
    testScenarios: [
      {
        input: 'database sharding',
        expectedCriteria: [
          'Explains what database sharding is',
          'Mentions horizontal scaling or partitioning',
          'Provides trade-offs or considerations',
          'Uses professional but accessible language',
        ],
        rubric: 'Response should explain sharding clearly with practical context, as an expert would to a junior developer.',
      },
      {
        input: 'event-driven architecture',
        expectedCriteria: [
          'Explains event-driven architecture concept',
          'Mentions message queues or event buses',
          'Discusses when to use this pattern',
          'Provides examples or use cases',
        ],
        rubric: 'Response should cover event-driven concepts with expert perspective and practical examples.',
      },
    ],
    judgePrompt: `You are evaluating a prompt engineering task. The student created a prompt to make an AI act as a senior software architect.

Evaluate the AI's response based on:
1. Expert credibility - Does the response sound like it's from an experienced architect?
2. Clarity - Is the explanation accessible to junior developers?
3. Practical value - Are there real-world examples and trade-offs?
4. Professionalism - Is the tone appropriate for a technical mentor?

Score 0-10 where:
- 0-3: Response lacks expertise or clarity
- 4-6: Response is helpful but missing key elements
- 7-8: Good expert-level explanation with examples
- 9-10: Excellent, comprehensive explanation with deep insights`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Эксперт-консультант',
      description: `
# Эксперт-консультант

Создайте промпт, который заставит ИИ действовать как **ведущий архитектор ПО**, способный объяснять сложные концепции проектирования систем.

## Ваша цель

Создайте промпт, который:
1. Устанавливает ИИ как эксперта-архитектора с 15+ годами опыта
2. Заставляет его объяснять концепции понятно для младших разработчиков
3. Использует реальные примеры и аналогии

## Требования

Ваш промпт должен включать:
- Четкое определение роли с уровнем экспертизы
- Инструкции по стилю объяснений
- Плейсхолдер \`{{INPUT}}\` для темы объяснения
`,
    },
    tr: {
      title: 'Uzman Danışman',
      description: `
# Uzman Danışman

AI'ı karmaşık sistem tasarım konseptlerini açıklayabilen bir **kıdemli yazılım mimarı** olarak davranması için bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir prompt tasarlayın:
1. AI'ı 15+ yıl deneyimli uzman bir mimar olarak konumlandırır
2. Konseptleri junior geliştiricilere anlaşılır şekilde açıklar
3. Gerçek dünya örnekleri ve analojiler kullanır
`,
    },
  },
};
