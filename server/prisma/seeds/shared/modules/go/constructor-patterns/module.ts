import { Module } from '../../../../types';
import { topics } from './topics';

export const constructorPatternsModule: Module = {
	title: 'Constructor Patterns',
	description: 'Master the Functional Options pattern, encapsulation with private fields, and builder patterns for flexible API design.',
	section: 'configuration-safety',
	order: 19,
	topics,
	translations: {
		ru: {
			title: 'Паттерны конструкторов',
			description: 'Освойте паттерн функциональных опций, инкапсуляцию с приватными полями и паттерны строителей для гибкого проектирования API.'
		},
		uz: {
			title: 'Konstruktor Patternlari',
			description: 'Moslashuvchan API dizayni uchun funksional parametrlar patternini, shaxsiy maydonlar bilan inkapsulyatsiyani va builder patternlarini o\'rganing.'
		}
	}
};
