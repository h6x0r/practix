import { Module } from '../../../../types';
import { fundamentalsTopic } from './topics';

export const securityFundamentalsModule: Module = {
	slug: 'sec-fundamentals',
	title: 'Security Fundamentals',
	description: 'Learn the foundational concepts of application security: CIA Triad, Defense in Depth, Least Privilege, and Threat Modeling.',
	section: 'security',
	order: 1,
	difficulty: 'easy',
	estimatedTime: '4h',
	topics: [fundamentalsTopic],
	translations: {
		ru: {
			title: 'Основы безопасности',
			description: 'Изучите фундаментальные концепции безопасности приложений: триада CIA, глубокая защита, минимальные привилегии и моделирование угроз.'
		},
		uz: {
			title: 'Xavfsizlik asoslari',
			description: 'Ilova xavfsizligining asosiy tushunchalarini o\'rganing: CIA Triad, chuqur himoya, minimal huquqlar va tahdidlarni modellashtirish.'
		}
	}
};
