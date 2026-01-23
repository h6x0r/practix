export default {
  slug: 'pe-system-prompt-design',
  title: 'System Prompt Master',
  difficulty: 'hard' as const,
  estimatedTime: '20m',
  taskType: 'PROMPT' as const,
  description: `
# System Prompt Master

Create a comprehensive system prompt that defines an AI assistant's complete behavior.

## Your Goal

Design a complete system prompt that covers:
1. Role and identity
2. Capabilities and limitations
3. Communication style
4. Behavioral guidelines
5. Safety considerations

## Requirements

Your prompt must:
- Be comprehensive and clear
- Cover all key aspects
- Be internally consistent
- Include \`{{INPUT}}\` for customization needs

## System Prompt Sections
- Identity & Role
- Core Capabilities
- Communication Style
- Boundaries & Limitations
- Error Handling
- Safety Guidelines
`,
  initialCode: `Design a system prompt for:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'A customer support chatbot for a software company.',
        expectedCriteria: [
          'Comprehensive system prompt',
          'Clear role definition',
          'Appropriate communication style',
          'Includes safety guidelines',
        ],
        rubric: 'Response should be a complete, production-ready system prompt.',
      },
      {
        input: 'An educational assistant for teaching programming to beginners.',
        expectedCriteria: [
          'Well-structured system prompt',
          'Appropriate for educational context',
          'Clear capabilities defined',
          'Comprehensive coverage',
        ],
        rubric: 'Response should demonstrate system prompt mastery.',
      },
    ],
    judgePrompt: `Evaluate system prompt design quality.

Criteria:
1. Comprehensiveness - Does it cover all aspects?
2. Clarity - Is it clear and unambiguous?
3. Consistency - Are all parts internally consistent?
4. Practicality - Would it work in production?

Score 0-10:
- 0-3: Incomplete or poorly designed
- 4-6: Basic system prompt but missing elements
- 7-8: Good comprehensive system prompt
- 9-10: Excellent production-ready system prompt`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Мастер системных промптов',
      description: `
# Мастер системных промптов

Создайте комплексный системный промпт, определяющий полное поведение ИИ-помощника.

## Ваша цель

Создайте полный системный промпт, охватывающий:
1. Роль и идентичность
2. Возможности и ограничения
3. Стиль коммуникации
4. Поведенческие рекомендации
5. Соображения безопасности
`,
    },
    tr: {
      title: 'Sistem Promptu Ustası',
      description: `
# Sistem Promptu Ustası

Bir AI asistanının tam davranışını tanımlayan kapsamlı bir sistem promptu oluşturun.

## Hedefiniz

Şunları kapsayan eksiksiz bir sistem promptu tasarlayın:
1. Rol ve kimlik
2. Yetenekler ve sınırlamalar
3. İletişim stili
4. Davranış yönergeleri
5. Güvenlik hususları
`,
    },
  },
};
