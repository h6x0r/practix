import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pe-output-formatting',
	title: 'Output Formatting',
	difficulty: 'easy',
	tags: ['prompt-engineering', 'formatting', 'json', 'structured-output'],
	estimatedTime: '20 min',
	isPremium: true,
	order: 3,
	taskType: 'PROMPT',

	description: `# Output Formatting

Learn to request specific output formats from AI models.

## The Challenge

Create a prompt that extracts information from a product review and outputs it as **structured JSON**.

## Requirements

Your prompt should:
- Parse the review text in \`{{INPUT}}\`
- Extract: product name, rating (1-5), pros, cons, and recommendation (yes/no)
- Output valid JSON format
- Handle various review styles

## Expected JSON Structure

\`\`\`json
{
  "product": "Product Name",
  "rating": 4,
  "pros": ["pro 1", "pro 2"],
  "cons": ["con 1"],
  "recommendation": true
}
\`\`\`

## Tips

1. Provide the exact JSON schema you want
2. Be explicit that output should be ONLY JSON (no markdown, no explanation)
3. Specify data types for each field`,

	initialCode: `Extract info from this review:

{{INPUT}}`,

	solutionCode: `You are a data extraction specialist. Parse the following product review and extract structured information.

Output ONLY valid JSON with no additional text, markdown, or explanation.

Required JSON schema:
{
  "product": "<string: product name>",
  "rating": <number: 1-5, infer from sentiment if not explicit>,
  "pros": ["<string: positive point>", ...],
  "cons": ["<string: negative point>", ...],
  "recommendation": <boolean: would reviewer recommend?>
}

Rules:
- rating must be a number 1-5
- pros and cons are arrays (can be empty)
- recommendation is true/false based on overall sentiment
- Return ONLY the JSON object, no other text

Review to analyze:
{{INPUT}}`,

	hint1: 'Show the exact JSON schema with field types',
	hint2: 'Explicitly say "output ONLY valid JSON with no additional text"',

	whyItMatters: `Structured output is essential for integrating AI into applications. When you need to parse AI responses programmatically, JSON output lets you use standard parsing libraries. This is a core skill for building AI-powered features.`,

	promptConfig: {
		testScenarios: [
			{
				input: `I bought the Sony WH-1000XM5 headphones last month. Amazing noise cancellation and super comfortable for long listening sessions! Battery life is incredible - I only charge once a week. The only downsides are the price ($400 is steep) and they don't fold as compactly as the previous model. Still, absolutely worth it if you can afford them. 9/10 would recommend!`,
				expectedCriteria: [
					'Output is valid JSON (parseable)',
					'Contains "product" field with headphone name',
					'Contains "rating" as a number (should be 4 or 5)',
					'Contains "pros" array with noise cancellation, comfort, battery',
					'Contains "cons" array with price and folding issue',
					'Contains "recommendation" as boolean true',
				],
				rubric:
					'Score 9-10 for valid JSON with all correct fields. Score 6-8 for valid JSON with minor field errors. Score 3-5 for invalid JSON or major missing fields. Score 1-2 for non-JSON output.',
			},
			{
				input: `Terrible experience with the QuickBlend Pro blender. Motor burned out after 2 weeks. It was loud and the lid never sealed properly. Save your money and get a different brand. One star.`,
				expectedCriteria: [
					'Output is valid JSON',
					'Product name is extracted',
					'Rating is 1',
					'Cons include motor failure, noise, lid issues',
					'recommendation is false',
				],
				rubric:
					'Score 8-10 for valid JSON correctly capturing negative sentiment. Deduct for parsing errors or wrong recommendation value.',
			},
		],
		judgePrompt: `Evaluate the JSON output quality:

1. Is the output valid JSON (can be parsed)?
2. Does it contain all required fields: product, rating, pros, cons, recommendation?
3. Are the field types correct (rating=number, pros/cons=arrays, recommendation=boolean)?
4. Does the extracted data accurately reflect the review content?

Output to evaluate:
{{OUTPUT}}

Expected criteria:
- {{CRITERIA}}

{{RUBRIC}}`,
		passingScore: 7,
	},

	translations: {
		ru: {
			title: 'Форматирование вывода',
			description: `# Форматирование вывода

Научитесь запрашивать конкретные форматы вывода от AI моделей.

## Задача

Создайте промпт, который извлекает информацию из отзыва о продукте и выводит её в формате **структурированного JSON**.

## Требования

Ваш промпт должен:
- Парсить текст отзыва из \`{{INPUT}}\`
- Извлекать: название продукта, рейтинг (1-5), плюсы, минусы и рекомендацию (да/нет)
- Выводить валидный JSON
- Обрабатывать различные стили отзывов

## Ожидаемая структура JSON

\`\`\`json
{
  "product": "Название продукта",
  "rating": 4,
  "pros": ["плюс 1", "плюс 2"],
  "cons": ["минус 1"],
  "recommendation": true
}
\`\`\`

## Советы

1. Предоставьте точную JSON схему, которую хотите
2. Явно укажите, что вывод должен быть ТОЛЬКО JSON (без markdown, без объяснений)
3. Укажите типы данных для каждого поля`,
			hint1: 'Покажите точную JSON схему с типами полей',
			hint2: 'Явно скажите "выведи ТОЛЬКО валидный JSON без дополнительного текста"',
			whyItMatters:
				'Структурированный вывод необходим для интеграции AI в приложения. Когда нужно программно парсить ответы AI, JSON позволяет использовать стандартные библиотеки. Это ключевой навык для создания AI-функций.',
		},
		uz: {
			title: 'Chiqishni formatlash',
			description: `# Chiqishni formatlash

AI modellaridan ma'lum chiqish formatlarini so'rashni o'rganing.

## Vazifa

Mahsulot sharhidan ma'lumot oluvchi va uni **strukturalangan JSON** formatida chiqaruvchi prompt yarating.

## Talablar

Promptingiz quyidagilarni bajarishi kerak:
- \`{{INPUT}}\`dagi sharh matnini tahlil qilish
- Quyidagilarni ajratib olish: mahsulot nomi, reyting (1-5), afzalliklar, kamchiliklar, tavsiya (ha/yo'q)
- Yaroqli JSON formatida chiqarish
- Turli sharh uslublarini qayta ishlash

## Kutilgan JSON strukturasi

\`\`\`json
{
  "product": "Mahsulot nomi",
  "rating": 4,
  "pros": ["afzallik 1", "afzallik 2"],
  "cons": ["kamchilik 1"],
  "recommendation": true
}
\`\`\`

## Maslahatlar

1. Kerakli aniq JSON sxemasini ko'rsating
2. Chiqish FAQAT JSON bo'lishi kerakligini aniq ayting (markdown yoki izohsiz)
3. Har bir maydon uchun ma'lumot turlarini ko'rsating`,
			hint1: "Maydon turlari bilan aniq JSON sxemasini ko'rsating",
			hint2: "\"Faqat yaroqli JSON chiqaring, qo'shimcha matnsiz\" deb aniq ayting",
			whyItMatters:
				"Strukturalangan chiqish AIni ilovalarga integratsiya qilish uchun zarur. AI javoblarini programmatik tahlil qilish kerak bo'lganda, JSON standart kutubxonalardan foydalanish imkonini beradi. Bu AI-funksiyalar yaratish uchun asosiy ko'nikma.",
		},
	},
};

export default task;
