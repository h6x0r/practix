/**
 * Go Fundamentals Module
 * Core Go concepts: data structures, IO, and constructors
 */

import { Module } from '../../../../types';
import { topics } from './topics';

export const fundamentalsModule: Module = {
	slug: 'fundamentals',
	title: 'Go Fundamentals',
	description: 'Master core Go concepts: data structures, IO interfaces, and constructor patterns',
	section: 'core-concepts',
	order: 1,
	topics,
	translations: {
		ru: {
			title: 'Основы Go',
			description: 'Освойте основные концепции Go: структуры данных, IO интерфейсы и паттерны конструкторов'
		},
		uz: {
			title: 'Go Asoslari',
			description: 'Go ning asosiy tushunchalarini o\'rganing: ma\'lumotlar tuzilmalari, IO interfeyslari va konstruktor patternlari'
		}
	}
};
