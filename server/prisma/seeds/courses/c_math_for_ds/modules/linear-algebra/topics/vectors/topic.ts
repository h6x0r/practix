import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'vectors',
	title: 'Vector Operations',
	description: 'Learn vector mathematics: addition, scalar multiplication, dot product, and norms.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Операции с векторами',
			description: 'Изучите векторную математику: сложение, умножение на скаляр, скалярное произведение и нормы.',
		},
		uz: {
			title: 'Vektor operatsiyalari',
			description: "Vektor matematikasini o'rganing: qo'shish, skalarga ko'paytirish, skalyar ko'paytma va normalar.",
		},
	},
};
