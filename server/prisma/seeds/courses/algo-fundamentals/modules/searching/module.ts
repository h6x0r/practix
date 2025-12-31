import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const searchingModule: Module = {
	slug: 'algo-searching',
	title: 'Searching',
	description: 'Master binary search and its variations for efficient searching in sorted data structures.',
	section: 'algorithms',
	order: 7,
	difficulty: 'medium',
	estimatedTime: '4h',
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Поиск',
			description: 'Освойте бинарный поиск и его вариации для эффективного поиска в отсортированных структурах данных.',
		},
		uz: {
			title: 'Qidiruv',
			description: 'Tartiblangan ma\'lumotlar strukturalarida samarali qidiruv uchun binar qidiruv va uning variantlarini o\'rganing.',
		},
	},
};

export default searchingModule;
