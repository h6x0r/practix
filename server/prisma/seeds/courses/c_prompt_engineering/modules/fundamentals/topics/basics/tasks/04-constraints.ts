import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pe-constraints',
	title: 'Constraints & Requirements',
	difficulty: 'easy',
	tags: ['prompt-engineering', 'constraints', 'requirements'],
	estimatedTime: '20 min',
	isPremium: true,
	order: 4,
	taskType: 'PROMPT',

	description: `# Constraints & Requirements

Learn to add explicit constraints that control AI output.

## The Challenge

Create a prompt that generates a **product description** with strict constraints:
- Exactly 3 sentences
- No superlatives (amazing, best, incredible, etc.)
- Include one specific technical specification
- End with a call-to-action

## Requirements

Your prompt should:
- Use \`{{INPUT}}\` for product details
- Enforce all constraints clearly
- Produce consistent, professional copy

## Example Input

\`\`\`
Product: Wireless Earbuds
Features: 8-hour battery, Bluetooth 5.3, noise cancellation, IPX4 water resistance
\`\`\`

## Expected Output

A 3-sentence description that:
1. Introduces the product
2. Highlights a key feature with a spec
3. Ends with a call-to-action

## Why Constraints Matter

Without constraints, AI tends to produce verbose, hyperbolic marketing copy. Explicit rules create predictable, usable output.`,

	initialCode: `Write a product description for:

{{INPUT}}`,

	solutionCode: `You are a professional copywriter. Write a product description following STRICT rules.

MANDATORY CONSTRAINTS:
1. EXACTLY 3 sentences - no more, no less
2. NO superlatives: avoid words like "best", "amazing", "incredible", "revolutionary", "unparalleled", "ultimate"
3. Include AT LEAST ONE specific number/specification from the product details
4. The FINAL sentence MUST be a call-to-action (e.g., "Order now", "Try it today", "Get yours")

Style guidelines:
- Professional and factual tone
- Focus on benefits, not just features
- Keep sentences concise (under 25 words each)

Product to describe:
{{INPUT}}

Write the 3-sentence description:`,

	hint1: 'List forbidden words explicitly so the AI knows what to avoid',
	hint2: 'Number your constraints and make them mandatory ("MUST", "EXACTLY")',

	whyItMatters: `Production AI applications need predictable output lengths and styles. By mastering constraints, you can integrate AI-generated content directly into templates, databases, and UIs without manual editing.`,

	promptConfig: {
		testScenarios: [
			{
				input: `Product: Wireless Earbuds
Features: 8-hour battery life, Bluetooth 5.3, active noise cancellation, IPX4 water resistance, 10mm drivers`,
				expectedCriteria: [
					'Output contains exactly 3 sentences',
					'No superlatives used (best, amazing, incredible, etc.)',
					'Contains at least one number/spec (8-hour, 5.3, 10mm, etc.)',
					'Last sentence is a call-to-action',
					'Professional, factual tone',
				],
				rubric:
					'Score 9-10 for meeting all 4 constraints. Score 7-8 for 3 constraints met. Score 4-6 for 2 constraints. Score 1-3 for 1 or fewer constraints.',
			},
			{
				input: `Product: Smart Thermostat
Features: WiFi connected, learns schedule in 1 week, saves up to 23% on heating, voice assistant compatible, 3.5-inch color display`,
				expectedCriteria: [
					'Exactly 3 sentences',
					'Includes specific stat (23% savings, 1 week, 3.5-inch)',
					'No hyperbolic language',
					'Ends with call-to-action',
				],
				rubric:
					'Full marks for all constraints. Deduct 2 points per violated constraint.',
			},
		],
		judgePrompt: `Evaluate if the product description meets all constraints:

1. Count the sentences - is it exactly 3?
2. Check for superlatives (amazing, best, incredible, revolutionary, ultimate, etc.)
3. Does it include at least one specific number or specification?
4. Does the last sentence contain a call-to-action?

Output to evaluate:
{{OUTPUT}}

Expected criteria:
- {{CRITERIA}}

{{RUBRIC}}`,
		passingScore: 7,
	},

	translations: {
		ru: {
			title: 'Ограничения и требования',
			description: `# Ограничения и требования

Научитесь добавлять явные ограничения, контролирующие вывод AI.

## Задача

Создайте промпт, генерирующий **описание продукта** со строгими ограничениями:
- Ровно 3 предложения
- Без превосходных степеней (потрясающий, лучший, невероятный и т.д.)
- Включить одну конкретную техническую характеристику
- Закончить призывом к действию

## Требования

Ваш промпт должен:
- Использовать \`{{INPUT}}\` для деталей продукта
- Четко применять все ограничения
- Создавать консистентный, профессиональный текст

## Пример входных данных

\`\`\`
Product: Беспроводные наушники
Features: 8 часов работы, Bluetooth 5.3, шумоподавление, IPX4 защита от воды
\`\`\`

## Ожидаемый результат

Описание из 3 предложений, которое:
1. Представляет продукт
2. Выделяет ключевую функцию с характеристикой
3. Заканчивается призывом к действию

## Почему важны ограничения

Без ограничений AI создаёт многословные, преувеличенные маркетинговые тексты. Явные правила создают предсказуемый, пригодный к использованию результат.`,
			hint1: 'Явно перечислите запрещённые слова, чтобы AI знал, чего избегать',
			hint2: 'Пронумеруйте ограничения и сделайте их обязательными ("ДОЛЖЕН", "РОВНО")',
			whyItMatters:
				'Production AI-приложениям нужны предсказуемые длины и стили вывода. Освоив ограничения, вы сможете интегрировать AI-контент напрямую в шаблоны, базы данных и UI без ручного редактирования.',
		},
		uz: {
			title: "Cheklovlar va talablar",
			description: `# Cheklovlar va talablar

AI chiqishini boshqaruvchi aniq cheklovlar qo'shishni o'rganing.

## Vazifa

Qat'iy cheklovlar bilan **mahsulot tavsifi** yaratuvchi prompt yarating:
- Aniq 3 ta gap
- Zo'r so'zlarsiz (ajoyib, eng yaxshi, aql bovar qilmas va h.k.)
- Bitta aniq texnik xususiyatni o'z ichiga olish
- Harakatga chaqiruv bilan yakunlash

## Talablar

Promptingiz quyidagilarni o'z ichiga olishi kerak:
- Mahsulot tafsilotlari uchun \`{{INPUT}}\` ishlatish
- Barcha cheklovlarni aniq qo'llash
- Izchil, professional nusxa yaratish

## Kirish misoli

\`\`\`
Product: Simsiz quloqchinlar
Features: 8 soatlik batareya, Bluetooth 5.3, shovqinni yo'qotish, IPX4 suv himoyasi
\`\`\`

## Kutilgan natija

3 gaplik tavsif:
1. Mahsulotni tanishtirish
2. Xususiyat bilan asosiy funktsiyani ajratib ko'rsatish
3. Harakatga chaqiruv bilan yakunlash

## Cheklovlar nima uchun muhim

Cheklovlarsiz AI ko'p so'zli, bo'rttirilgan marketing matnlarini yaratadi. Aniq qoidalar oldindan aytib bo'ladigan, foydalanish mumkin bo'lgan natija yaratadi.`,
			hint1: "AI nimadan qochishni bilishi uchun taqiqlangan so'zlarni aniq sanab o'ting",
			hint2: "Cheklovlaringizni raqamlang va ularni majburiy qiling (\"KERAK\", \"ANIQ\")",
			whyItMatters:
				"Production AI ilovalariga oldindan aytib bo'ladigan chiqish uzunliklari va uslublari kerak. Cheklovlarni o'zlashtirib, AI-yaratilgan kontentni qo'lda tahrirlashsiz to'g'ridan-to'g'ri shablonlar, ma'lumotlar bazalari va UIlarga integratsiya qilishingiz mumkin.",
		},
	},
};

export default task;
