import { Module } from '../../../../types';
import { techniquesTopic } from './topics';

export const divideConquerModule: Module = {
	slug: 'algo-divide-conquer',
	title: 'Divide and Conquer',
	description: 'Master the divide and conquer paradigm: break problems into subproblems, solve recursively, and combine solutions',
	section: 'algorithms',
	order: 5,
	difficulty: 'medium',
	estimatedTime: '6h',
	topics: [techniquesTopic],
	translations: {
		ru: {
			title: 'Разделяй и властвуй',
			description: 'Освойте парадигму "разделяй и властвуй": разбиение задач на подзадачи, рекурсивное решение и объединение результатов'
		},
		uz: {
			title: "Bo'l va hukmronlik qil",
			description: "\"Bo'l va hukmronlik qil\" paradigmasini o'rganing: masalalarni qismlarga bo'lish, rekursiv yechish va natijalarni birlashtirish"
		}
	}
};

export default divideConquerModule;
