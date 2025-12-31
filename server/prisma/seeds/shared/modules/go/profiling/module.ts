import { Module } from '../../../../types';
import { topics } from './topics';

export const profilingModule: Module = {
	title: 'Performance Profiling',
	description: 'Master CPU and memory profiling with pprof, benchmark optimization, and allocation reduction techniques.',
	section: 'production-patterns',
	order: 13,
	topics,
	translations: {
		ru: {
			title: 'Профилирование производительности',
			description: 'Освойте профилирование CPU и памяти с помощью pprof, оптимизацию бенчмарков и методы сокращения выделений памяти.'
		},
		uz: {
			title: 'Ishlash unumdorligini profillashtirish',
			description: 'pprof yordamida CPU va xotira profillashtirish, benchmark optimallashtiruvi va xotira ajratishni kamaytirish usullarini o\'rganing.'
		}
	}
};
