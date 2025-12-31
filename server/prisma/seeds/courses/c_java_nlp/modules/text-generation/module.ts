import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'text-generation',
	title: 'Text Generation',
	description: 'Learn text generation techniques from n-grams to neural language models.',
	order: 8,
	isPremium: true,
	translations: {
		ru: {
			title: 'Генерация текста',
			description:
				'Изучите техники генерации текста от n-грамм до нейронных языковых моделей.',
		},
		uz: {
			title: 'Matn generatsiyasi',
			description:
				"N-grammalardan neyron til modellarigacha matn generatsiya texnikalarini o'rganing.",
		},
	},
};

export default module;
