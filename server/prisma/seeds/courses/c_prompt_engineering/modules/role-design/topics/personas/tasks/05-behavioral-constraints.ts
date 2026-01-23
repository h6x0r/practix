export default {
  slug: 'pe-behavioral-constraints',
  title: 'Setting Boundaries',
  difficulty: 'medium' as const,
  estimatedTime: '12m',
  taskType: 'PROMPT' as const,
  description: `
# Setting Boundaries

Create a prompt that establishes **clear behavioral constraints** for an AI assistant.

## Your Goal

Design a prompt for a code review assistant that:
1. Only reviews code, doesn't write new code
2. Points out issues without being overly critical
3. Limits suggestions to 3 most important items
4. Never suggests complete rewrites

## Requirements

Your prompt must:
- Define the assistant's role clearly
- Establish explicit boundaries on what it won't do
- Specify the constructive feedback style
- Include \`{{INPUT}}\` for the code to review

## Example Input
\`\`\`python
def calc(x,y):
    z = x+y
    return z
\`\`\`

## Expected Output Style
- Limited to 3 suggestions maximum
- Constructive, not critical
- Reviews only, no complete rewrites
`,
  initialCode: `You are a code reviewer. Review this code.

{{INPUT}}`,
  promptConfig: {
    testScenarios: [
      {
        input: `def get_user(id):
    user = db.query("SELECT * FROM users WHERE id = " + str(id))
    if user:
        return user[0]
    return None`,
        expectedCriteria: [
          'Provides 3 or fewer suggestions',
          'Uses constructive language',
          'Does not provide a complete rewrite',
          'Points out key issues (SQL injection, naming)',
        ],
        rubric: 'Response should follow constraints: max 3 items, constructive tone, no full rewrites.',
      },
      {
        input: `function fetchData() {
  var data = null;
  $.ajax({ url: '/api', async: false, success: function(r) { data = r; }});
  return data;
}`,
        expectedCriteria: [
          'Limited to 3 suggestions maximum',
          'Constructive, helpful tone',
          'Does not rewrite the entire function',
          'Identifies important issues (async, var usage)',
        ],
        rubric: 'Response should respect behavioral boundaries while providing valuable feedback.',
      },
    ],
    judgePrompt: `Evaluate this code review for adherence to behavioral constraints.

Criteria:
1. Constraint adherence - Does it limit to 3 suggestions and avoid rewrites?
2. Constructiveness - Is the tone helpful rather than harsh?
3. Focus - Are the suggestions prioritized by importance?
4. Role boundaries - Does it only review, not write new code?

Score 0-10:
- 0-3: Ignores constraints, too many suggestions or rewrites code
- 4-6: Partially follows constraints but some violations
- 7-8: Good adherence to constraints with constructive feedback
- 9-10: Perfect constraint following with excellent prioritization`,
    passingScore: 7,
  },
  translations: {
    ru: {
      title: 'Установка границ',
      description: `
# Установка границ

Создайте промпт, устанавливающий **четкие поведенческие ограничения** для ИИ-ассистента.

## Ваша цель

Создайте промпт для ассистента по код-ревью, который:
1. Только проверяет код, не пишет новый
2. Указывает на проблемы без излишней критики
3. Ограничивает предложения 3 наиболее важными пунктами
4. Никогда не предлагает полную переработку
`,
    },
    tr: {
      title: 'Sınırları Belirleme',
      description: `
# Sınırları Belirleme

Bir AI asistanı için **net davranışsal kısıtlamalar** oluşturan bir prompt oluşturun.

## Hedefiniz

Şunları yapan bir kod inceleme asistanı promptu tasarlayın:
1. Sadece kod inceler, yeni kod yazmaz
2. Aşırı eleştirel olmadan sorunları belirtir
3. Önerileri en önemli 3 maddeyle sınırlar
4. Asla tam yeniden yazma önermez
`,
    },
  },
};
