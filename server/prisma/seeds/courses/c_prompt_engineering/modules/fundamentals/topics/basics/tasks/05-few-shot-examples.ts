import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pe-few-shot-examples',
	title: 'Few-Shot Examples',
	difficulty: 'medium',
	tags: ['prompt-engineering', 'few-shot', 'examples', 'learning'],
	estimatedTime: '25 min',
	isPremium: true,
	order: 5,
	taskType: 'PROMPT',

	description: `# Few-Shot Examples

Learn the powerful technique of teaching AI by example.

## The Challenge

Create a prompt that classifies customer support tickets into categories using **few-shot learning**.

## Categories

- \`billing\` - Payment, invoices, refunds
- \`technical\` - Bugs, errors, how-to questions
- \`account\` - Login, password, profile changes
- \`shipping\` - Delivery, tracking, address

## Requirements

Your prompt should:
- Include 2-3 examples showing the correct format
- Use \`{{INPUT}}\` for the ticket to classify
- Output ONLY the category label (one word)
- Handle edge cases and ambiguous tickets

## Example Format

\`\`\`
Ticket: "I can't log into my account"
Category: account

Ticket: "Where is my order?"
Category: shipping

Ticket: "{{INPUT}}"
Category:
\`\`\`

## Why Few-Shot Works

Instead of explaining rules, you show the AI patterns. This is often more effective than lengthy instructions, especially for classification and formatting tasks.`,

	initialCode: `Classify this support ticket:

{{INPUT}}`,

	solutionCode: `You are a customer support ticket classifier. Your task is to categorize tickets into exactly one of these categories: billing, technical, account, shipping.

EXAMPLES:

Ticket: "I was charged twice for my subscription"
Category: billing

Ticket: "The app crashes when I try to upload photos"
Category: technical

Ticket: "I need to change my email address on my profile"
Category: account

Ticket: "My package hasn't arrived and it's been 2 weeks"
Category: shipping

Ticket: "How do I reset my password?"
Category: account

Ticket: "I want a refund for my last order"
Category: billing

---

Now classify the following ticket. Respond with ONLY the category name (billing, technical, account, or shipping), nothing else.

Ticket: "{{INPUT}}"
Category:`,

	hint1: 'Include at least 2-3 diverse examples covering different categories',
	hint2: 'Make your examples short but representative of each category',

	whyItMatters: `Few-shot prompting is one of the most powerful techniques in prompt engineering. It's used extensively in production systems for classification, extraction, and formatting tasks. By showing rather than telling, you can achieve higher accuracy with less prompt complexity.`,

	promptConfig: {
		testScenarios: [
			{
				input: `I'm trying to track my order but the tracking link doesn't work`,
				expectedCriteria: [
					'Output is exactly one word',
					'Output is one of: billing, technical, account, shipping',
					'Correct classification: shipping (order tracking)',
				],
				rubric:
					'Score 10 for correct single-word answer "shipping". Score 5 for close category or correct with extra words. Score 0 for wrong category.',
			},
			{
				input: `My credit card was declined but the payment shows as pending`,
				expectedCriteria: [
					'Output is exactly one word',
					'Correct classification: billing (payment issue)',
				],
				rubric:
					'Score 10 for "billing". Score 5 for partially correct. Score 0 for wrong.',
			},
			{
				input: `The search feature keeps showing an error 500 message`,
				expectedCriteria: [
					'Output is exactly one word',
					'Correct classification: technical (error/bug)',
				],
				rubric:
					'Score 10 for "technical". Score 5 for partially correct. Score 0 for wrong.',
			},
		],
		judgePrompt: `Evaluate the classification output:

1. Is the output exactly one word?
2. Is it one of the valid categories (billing, technical, account, shipping)?
3. Is the classification correct for the given ticket?

Output to evaluate:
{{OUTPUT}}

Expected criteria:
- {{CRITERIA}}

{{RUBRIC}}`,
		passingScore: 7,
	},

	translations: {
		ru: {
			title: 'Примеры Few-Shot',
			description: `# Примеры Few-Shot

Изучите мощную технику обучения AI на примерах.

## Задача

Создайте промпт, который классифицирует тикеты поддержки клиентов по категориям, используя **few-shot learning**.

## Категории

- \`billing\` - Оплата, счета, возвраты
- \`technical\` - Баги, ошибки, вопросы "как сделать"
- \`account\` - Вход, пароль, изменения профиля
- \`shipping\` - Доставка, отслеживание, адрес

## Требования

Ваш промпт должен:
- Включать 2-3 примера с правильным форматом
- Использовать \`{{INPUT}}\` для классифицируемого тикета
- Выводить ТОЛЬКО метку категории (одно слово)
- Обрабатывать граничные случаи и неоднозначные тикеты

## Формат примеров

\`\`\`
Ticket: "Не могу войти в аккаунт"
Category: account

Ticket: "Где мой заказ?"
Category: shipping

Ticket: "{{INPUT}}"
Category:
\`\`\`

## Почему Few-Shot работает

Вместо объяснения правил вы показываете AI паттерны. Это часто эффективнее длинных инструкций, особенно для задач классификации и форматирования.`,
			hint1: 'Включите минимум 2-3 разнообразных примера, покрывающих разные категории',
			hint2: 'Делайте примеры короткими, но репрезентативными для каждой категории',
			whyItMatters:
				'Few-shot prompting - одна из самых мощных техник prompt engineering. Она широко используется в production-системах для классификации, извлечения и форматирования. Показывая вместо объяснения, вы достигаете большей точности с меньшей сложностью промпта.',
		},
		uz: {
			title: 'Few-Shot misollar',
			description: `# Few-Shot misollar

AIni misollar orqali o'rgatishning kuchli texnikasini o'rganing.

## Vazifa

**Few-shot learning** yordamida mijozlar yordam tiketlarini kategoriyalarga tasniflаydiган prompt yarating.

## Kategoriyalar

- \`billing\` - To'lov, hisob-fakturalar, qaytarishlar
- \`technical\` - Xatolar, buglar, qanday qilish savollari
- \`account\` - Kirish, parol, profil o'zgarishlari
- \`shipping\` - Yetkazib berish, kuzatish, manzil

## Talablar

Promptingiz quyidagilarni o'z ichiga olishi kerak:
- To'g'ri formatni ko'rsatuvchi 2-3 ta misolni o'z ichiga olish
- Tasniflanadigan tiket uchun \`{{INPUT}}\` ishlatish
- FAQAT kategoriya yorlig'ini chiqarish (bitta so'z)
- Chegaraviy holatlar va noaniq tiketlarni qayta ishlash

## Misol formati

\`\`\`
Ticket: "Hisobimga kira olmayapman"
Category: account

Ticket: "Buyurtmam qayerda?"
Category: shipping

Ticket: "{{INPUT}}"
Category:
\`\`\`

## Few-Shot nima uchun ishlaydi

Qoidalarni tushuntirish o'rniga, siz AIga patternlarni ko'rsatasiz. Bu ko'pincha uzun ko'rsatmalardan ko'ra samaraliroq, ayniqsa tasniflash va formatlash vazifalari uchun.`,
			hint1: "Turli kategoriyalarni qamrab oluvchi kamida 2-3 ta turli misolni kiriting",
			hint2: "Misollaringizni qisqa, lekin har bir kategoriyaga xos qiling",
			whyItMatters:
				"Few-shot prompting prompt engineeringdagi eng kuchli texnikalardan biri. U tasniflash, ajratib olish va formatlash vazifalari uchun production tizimlarida keng qo'llaniladi. Aytish o'rniga ko'rsatish orqali, kamroq prompt murakkabligi bilan yuqori aniqlikka erishishingiz mumkin.",
		},
	},
};

export default task;
