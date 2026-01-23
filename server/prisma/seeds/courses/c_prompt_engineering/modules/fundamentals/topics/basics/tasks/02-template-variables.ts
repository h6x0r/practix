import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pe-template-variables',
	title: 'Template Variables',
	difficulty: 'easy',
	tags: ['prompt-engineering', 'templates', 'variables'],
	estimatedTime: '15 min',
	isPremium: true,
	order: 2,
	taskType: 'PROMPT',

	description: `# Template Variables

Learn to create reusable prompt templates using variables.

## The Challenge

Create a prompt template that generates a **professional email** based on:
- The sender's name
- The recipient's name
- The email topic

## Requirements

Your prompt should:
- Use \`{{INPUT}}\` for the context/details
- Generate a professional, polite email
- Include a proper greeting and sign-off
- Be adaptable to different topics

## Input Format

The input will be structured as:
\`\`\`
Sender: [name]
Recipient: [name]
Topic: [topic]
Details: [additional context]
\`\`\`

## Expected Output

A complete, professional email with:
- Greeting addressing the recipient
- Body explaining the topic
- Polite closing
- Sender's signature`,

	initialCode: `Write an email about:

{{INPUT}}`,

	solutionCode: `You are a professional email writer. Generate a polite, concise business email based on the provided information.

Guidelines:
1. Start with "Dear [Recipient Name]," or "Hello [Recipient Name],"
2. Keep the body to 2-3 short paragraphs
3. Be professional but friendly in tone
4. End with "Best regards," or "Sincerely," followed by the sender's name

Email Parameters:
{{INPUT}}

Generate the complete email now:`,

	hint1: 'Specify the exact structure: greeting, body, closing, signature',
	hint2: 'Add guidelines for tone and length to ensure consistency',

	whyItMatters: `Template prompts let you create reusable patterns for common tasks. Instead of writing new prompts every time, you parameterize the variable parts. This is how production AI applications work at scale.`,

	promptConfig: {
		testScenarios: [
			{
				input: `Sender: Alex Johnson
Recipient: Sarah Chen
Topic: Project deadline extension
Details: Need 3 extra days to complete the analytics dashboard due to unexpected API changes`,
				expectedCriteria: [
					'Email addresses Sarah by name',
					'Mentions the project deadline extension request',
					'Includes the reason (API changes)',
					'Has proper greeting and sign-off',
					'Signed by Alex',
				],
				rubric:
					'Score 8-10 for complete professional email with all elements. Score 5-7 for missing greeting or sign-off. Score 1-4 for incomplete or unprofessional output.',
			},
			{
				input: `Sender: Maria Garcia
Recipient: Tech Support Team
Topic: Software access request
Details: Need access to the company VPN for remote work starting next week`,
				expectedCriteria: [
					'Addresses the team professionally',
					'Clearly states the access request',
					'Mentions the timeline (next week)',
					'Professional tone throughout',
				],
				rubric:
					'Score 8-10 for clear, complete email. Deduct points for missing elements or wrong tone.',
			},
		],
		judgePrompt: `Evaluate if the generated email meets professional standards:

1. Does it have a proper greeting addressing the recipient?
2. Is the body clear and covers the topic adequately?
3. Does it include a proper closing and signature?
4. Is the tone professional and appropriate?

Generated email:
{{OUTPUT}}

Expected criteria:
- {{CRITERIA}}

{{RUBRIC}}`,
		passingScore: 7,
	},

	translations: {
		ru: {
			title: 'Шаблонные переменные',
			description: `# Шаблонные переменные

Научитесь создавать переиспользуемые шаблоны промптов с переменными.

## Задача

Создайте шаблон промпта для генерации **профессионального письма** на основе:
- Имени отправителя
- Имени получателя
- Темы письма

## Требования

Ваш промпт должен:
- Использовать \`{{INPUT}}\` для контекста/деталей
- Генерировать профессиональное, вежливое письмо
- Включать приветствие и подпись
- Быть адаптируемым к разным темам

## Формат входных данных

Входные данные будут структурированы:
\`\`\`
Sender: [имя]
Recipient: [имя]
Topic: [тема]
Details: [дополнительный контекст]
\`\`\`

## Ожидаемый результат

Полное профессиональное письмо с:
- Приветствием с именем получателя
- Телом с объяснением темы
- Вежливым завершением
- Подписью отправителя`,
			hint1: 'Укажите точную структуру: приветствие, тело, завершение, подпись',
			hint2: 'Добавьте рекомендации по тону и длине для обеспечения консистентности',
			whyItMatters:
				'Шаблонные промпты позволяют создавать переиспользуемые паттерны для типичных задач. Вместо написания новых промптов каждый раз, вы параметризируете переменные части. Так работают production AI-приложения.',
		},
		uz: {
			title: 'Shablon o\'zgaruvchilari',
			description: `# Shablon o'zgaruvchilari

O'zgaruvchilar bilan qayta foydalaniladigan prompt shablonlarini yaratishni o'rganing.

## Vazifa

Quyidagilar asosida **professional email** yaratuvchi prompt shablonini yarating:
- Jo'natuvchi ismi
- Qabul qiluvchi ismi
- Email mavzusi

## Talablar

Promptingiz quyidagilarni o'z ichiga olishi kerak:
- Kontekst/tafsilotlar uchun \`{{INPUT}}\` ishlatish
- Professional, xushmuomala email yaratish
- To'g'ri salomlash va imzo qo'shish
- Turli mavzularga moslashuvchan bo'lish

## Kirish formati

Kirish quyidagicha strukturalangan bo'ladi:
\`\`\`
Sender: [ism]
Recipient: [ism]
Topic: [mavzu]
Details: [qo'shimcha kontekst]
\`\`\`

## Kutilgan natija

To'liq professional email:
- Qabul qiluvchiga salomlash
- Mavzuni tushuntiruvchi matn
- Xushmuomala yakunlash
- Jo'natuvchi imzosi`,
			hint1: "Aniq strukturani ko'rsating: salomlash, matn, yakunlash, imzo",
			hint2: "Izchillikni ta'minlash uchun ton va uzunlik bo'yicha ko'rsatmalar qo'shing",
			whyItMatters:
				"Shablon promptlar umumiy vazifalar uchun qayta foydalaniladigan patternlar yaratishga imkon beradi. Har safar yangi prompt yozish o'rniga, o'zgaruvchan qismlarni parametrlashtirасиз. Production AI ilovalari shunday ishlaydi.",
		},
	},
};

export default task;
