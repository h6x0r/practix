export default {
  slug: 'pe-scope-constraints',
  title: 'Define the Boundaries',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Define the Boundaries

Create a prompt that clearly sets the scope and constraints for the AI's response.

## Your Goal

Design a prompt for a feature planner that includes:
1. Technical constraints (budget, timeline, tech stack)
2. Scope limitations (what's in and out of scope)
3. Clear success criteria

## Requirements

Your prompt must:
- Define clear boundaries
- Specify what to include and exclude
- Set measurable criteria
- Include \`{{INPUT}}\` for the feature request

## Example Input
\`\`\`
Add a user profile page to our app.
\`\`\`

## Expected Output Style
- Respects stated constraints
- Stays within scope
- Addresses success criteria
`,
  initialCode: `Plan this feature:

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: 'Add a search feature to our blog.',
        expectedCriteria: [
          'Respects any stated constraints',
          'Stays within defined scope',
          'Does not suggest out-of-scope features',
          'Addresses success criteria if provided',
        ],
        rubric: 'Response should respect boundaries and constraints defined in the prompt.',
      },
      {
        input: 'Implement a notification system for our app.',
        expectedCriteria: [
          'Works within given limitations',
          'Focuses on in-scope requirements',
          'Avoids scope creep',
          'Provides focused, bounded solution',
        ],
        rubric: 'Response should demonstrate constraint-aware planning.',
      },
    ],
    judgePrompt: `Evaluate how well the response respects constraints and scope.

Criteria:
1. Constraint adherence - Are technical constraints respected?
2. Scope discipline - Does it stay within boundaries?
3. Focus - Is the response focused on in-scope items?
4. Practicality - Is the solution feasible within constraints?

Score 0-10:
- 0-3: Ignores constraints, exceeds scope
- 4-6: Partial adherence to constraints
- 7-8: Good constraint and scope adherence
- 9-10: Excellent bounded solution within all constraints`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Определи границы',
      description: `
# Определи границы

Создайте промпт, который четко устанавливает область действия и ограничения для ответа ИИ.

## Ваша цель

Создайте промпт для планировщика функций, который включает:
1. Технические ограничения (бюджет, сроки, технологический стек)
2. Ограничения области (что входит и что не входит в область)
3. Четкие критерии успеха
`,
    },
    tr: {
      title: 'Sınırları Belirle',
      description: `
# Sınırları Belirle

AI'ın yanıtı için kapsamı ve kısıtlamaları net bir şekilde belirleyen bir prompt oluşturun.

## Hedefiniz

Şunları içeren bir özellik planlayıcısı promptu tasarlayın:
1. Teknik kısıtlamalar (bütçe, zaman çizelgesi, teknoloji yığını)
2. Kapsam sınırlamaları (kapsam içi ve dışı olanlar)
3. Net başarı kriterleri
`,
    },
  },
};
