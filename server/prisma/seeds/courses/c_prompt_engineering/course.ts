import { CourseMeta } from '../../types';

const course: CourseMeta = {
	slug: 'prompt-engineering',
	title: 'Prompt Engineering',
	description:
		'Master the art and science of crafting effective prompts for AI systems. Learn systematic techniques for instruction design, context management, output formatting, and iterative refinement.',
	category: 'cs',
	icon: '🧠',
	estimatedTime: '15h',
	order: 25,
	translations: {
		ru: {
			title: 'Prompt Engineering',
			description:
				'Освойте искусство и науку создания эффективных промптов для AI систем. Изучите системные техники проектирования инструкций, управления контекстом, форматирования вывода и итеративного улучшения.',
		},
		uz: {
			title: 'Prompt Engineering',
			description:
				"AI tizimlari uchun samarali promptlar yaratish san'ati va fanini o'rganing. Ko'rsatmalar loyihalash, kontekstni boshqarish, chiqish formatlash va iterativ takomillashtirishning tizimli usullarini o'rganing.",
		},
	},
};

export default course;
