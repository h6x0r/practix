import { Module } from '../../../../types';
import { topics } from './topics';

export const goroutinesModule: Module = {
	title: 'Goroutine Management',
	description: 'Master goroutine lifecycle with context-aware patterns, proper cleanup, and leak prevention for production systems.',
	section: 'concurrency',
	order: 15,
	topics,
	translations: {
		ru: {
			title: 'Управление горутинами',
			description: 'Освойте жизненный цикл горутин с контекстно-ориентированными паттернами, правильной очисткой и предотвращением утечек для производственных систем.'
		},
		uz: {
			title: 'Goroutinalarni boshqarish',
			description: 'Ishlab chiqarish tizimlari uchun kontekstga asoslangan naqshlar, to\'g\'ri tozalash va oqish oldini olish bilan goroutine hayot siklini o\'zlashtiring.'
		}
	}
};
