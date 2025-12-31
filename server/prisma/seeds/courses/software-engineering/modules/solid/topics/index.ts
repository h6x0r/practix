import { Topic } from '../../../../../types';
import { tasks } from './principles/tasks';

export const principlesTopic: Topic = {
	slug: 'se-solid-principles',
	title: 'SOLID Principles',
	description: 'Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion principles.',
	difficulty: 'medium',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Принципы SOLID',
			description: 'Принципы единственной ответственности, открытости/закрытости, подстановки Барбары Лисков, разделения интерфейса и инверсии зависимостей.',
		},
		uz: {
			title: 'SOLID Prinsiplari',
			description: 'Yagona mas\'uliyat, Ochiq/Yopiq, Liskov almashtirish, Interfeys ajratish va Bog\'liqlik inversiyasi prinsiplari.',
		},
	},
};
