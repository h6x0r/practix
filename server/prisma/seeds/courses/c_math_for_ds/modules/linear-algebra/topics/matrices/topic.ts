import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'matrices',
	title: 'Matrix Operations',
	description: 'Learn matrix mathematics: multiplication, transpose, inverse, and determinants.',
	order: 2,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Операции с матрицами',
			description: 'Изучите матричную математику: умножение, транспонирование, обратную матрицу и определители.',
		},
		uz: {
			title: 'Matritsa operatsiyalari',
			description: "Matritsa matematikasini o'rganing: ko'paytirish, transpozitsiya, teskari matritsa va determinantlar.",
		},
	},
};
