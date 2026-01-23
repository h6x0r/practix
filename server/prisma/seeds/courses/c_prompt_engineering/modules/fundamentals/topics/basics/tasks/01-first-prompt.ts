import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pe-first-prompt',
	title: 'Your First Prompt',
	difficulty: 'easy',
	tags: ['prompt-engineering', 'basics', 'instructions'],
	estimatedTime: '15 min',
	isPremium: true,
	order: 1,
	taskType: 'PROMPT',

	description: `# Your First Prompt

Welcome to Prompt Engineering! In this task, you'll learn to write your first effective prompt.

## The Challenge

Write a prompt that instructs an AI to **summarize a news article** into exactly 3 bullet points.

## Requirements

Your prompt should:
- Clearly state the task (summarization)
- Specify the output format (3 bullet points)
- Handle the input placeholder \`{{INPUT}}\` for the article text

## Example

If the input is a long article about climate change, the AI should produce:
- First key point from the article
- Second key point from the article
- Third key point from the article

## Tips

1. Be explicit about what you want
2. Specify the exact output format
3. Use the \`{{INPUT}}\` placeholder where the article will be inserted`,

	initialCode: `Summarize the following:

{{INPUT}}`,

	solutionCode: `You are a professional summarizer. Your task is to extract the 3 most important points from the given text.

Instructions:
1. Read the following article carefully
2. Identify the 3 most significant facts or ideas
3. Present each point as a concise bullet point (max 20 words each)

Output Format:
- [First key point]
- [Second key point]
- [Third key point]

Article to summarize:
{{INPUT}}`,

	hint1: 'Tell the AI exactly what format you want: 3 bullet points',
	hint2: 'Add explicit constraints like word limits for each point',

	whyItMatters: `Clear prompts are the foundation of effective AI interaction. Vague instructions lead to unpredictable outputs, while specific prompts produce consistent, useful results. This skill applies to every AI tool you'll ever use.`,

	promptConfig: {
		testScenarios: [
			{
				input: `Scientists have discovered a new species of deep-sea fish in the Mariana Trench. The creature, named Pseudoliparis swirei, lives at depths of up to 8,000 meters. It has adapted to extreme pressure and darkness by developing translucent skin and highly sensitive pressure receptors. Researchers believe this discovery could help us understand how life adapts to extreme environments.`,
				expectedCriteria: [
					'Output contains exactly 3 bullet points',
					'Each bullet point summarizes a key fact from the article',
					'Points are concise and focused',
					'Uses proper bullet point formatting (-, *, or numbers)',
				],
				rubric:
					'Score 8-10 if output has exactly 3 clear, relevant bullet points. Score 5-7 if format is slightly off but content is good. Score 1-4 if missing bullet format or wrong number of points.',
			},
			{
				input: `The global electric vehicle market grew by 35% in 2024. China leads with 60% of worldwide EV sales. Battery costs have dropped 50% since 2020, making EVs more affordable. However, charging infrastructure remains a challenge in rural areas.`,
				expectedCriteria: [
					'Output contains exactly 3 bullet points',
					'Captures the main statistics and facts',
					'Points are distinct and non-overlapping',
				],
				rubric:
					'Score 8-10 if output correctly summarizes 3 key points in bullet format. Deduct points for missing key facts or poor formatting.',
			},
		],
		judgePrompt: `Evaluate if the AI output follows the prompt requirements:

1. Does the output contain exactly 3 bullet points?
2. Are the bullet points relevant summaries of the input?
3. Is the format correct (using -, *, or numbered list)?

Output to evaluate:
{{OUTPUT}}

Expected criteria:
- {{CRITERIA}}

{{RUBRIC}}`,
		passingScore: 7,
	},

	translations: {
		ru: {
			title: 'Ваш первый промпт',
			description: `# Ваш первый промпт

Добро пожаловать в Prompt Engineering! В этом задании вы напишете свой первый эффективный промпт.

## Задача

Напишите промпт, который инструктирует AI **резюмировать новостную статью** в ровно 3 пункта.

## Требования

Ваш промпт должен:
- Четко указать задачу (резюмирование)
- Указать формат вывода (3 пункта)
- Использовать плейсхолдер \`{{INPUT}}\` для текста статьи

## Пример

Если на вход подается длинная статья о климате, AI должен выдать:
- Первый ключевой пункт из статьи
- Второй ключевой пункт из статьи
- Третий ключевой пункт из статьи

## Советы

1. Будьте явными в том, что хотите
2. Укажите точный формат вывода
3. Используйте плейсхолдер \`{{INPUT}}\` там, где будет вставлена статья`,
			hint1: 'Точно скажите AI, какой формат вам нужен: 3 пункта',
			hint2: 'Добавьте явные ограничения, например лимит слов для каждого пункта',
			whyItMatters:
				'Четкие промпты - основа эффективного взаимодействия с AI. Размытые инструкции приводят к непредсказуемым результатам, а конкретные промпты дают стабильные и полезные ответы.',
		},
		uz: {
			title: 'Birinchi promptingiz',
			description: `# Birinchi promptingiz

Prompt Engineeringga xush kelibsiz! Bu vazifada siz birinchi samarali promptni yozasiz.

## Vazifa

AIga **yangiliklar maqolasini** aniq 3 ta bandda qisqartirish bo'yicha ko'rsatma beruvchi prompt yozing.

## Talablar

Promptingiz quyidagilarni o'z ichiga olishi kerak:
- Vazifani aniq ko'rsatish (qisqartirish)
- Chiqish formatini belgilash (3 ta band)
- Maqola matni uchun \`{{INPUT}}\` placeholderni ishlatish

## Misol

Agar kiritish iqlim o'zgarishi haqida uzun maqola bo'lsa, AI quyidagilarni chiqarishi kerak:
- Maqoladan birinchi asosiy nuqta
- Maqoladan ikkinchi asosiy nuqta
- Maqoladan uchinchi asosiy nuqta

## Maslahatlar

1. Xohlagan narsangizni aniq aytib bering
2. Aniq chiqish formatini ko'rsating
3. Maqola kiritiladigan joyda \`{{INPUT}}\` placeholderni ishlating`,
			hint1: "AIga kerakli formatni aniq ayting: 3 ta band",
			hint2: "Har bir band uchun so'z chegarasi kabi aniq cheklovlar qo'shing",
			whyItMatters:
				"Aniq promptlar AI bilan samarali muloqotning asosidir. Noaniq ko'rsatmalar oldindan aytib bo'lmaydigan natijalarga olib keladi, aniq promptlar esa barqaror va foydali javoblar beradi.",
		},
	},
};

export default task;
