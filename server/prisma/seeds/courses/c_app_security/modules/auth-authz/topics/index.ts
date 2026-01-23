import { Topic } from '../../../../../types';
import { tasks } from './fundamentals/tasks';

export const fundamentalsTopic: Topic = {
	slug: 'sec-auth-fundamentals',
	title: 'Authentication & Authorization Fundamentals',
	description: 'RBAC, ABAC, OAuth 2.0, JWT Security, Session Management, and Multi-Factor Authentication.',
	difficulty: 'medium',
	estimatedTime: '6h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Основы аутентификации и авторизации',
			description: 'RBAC, ABAC, OAuth 2.0, безопасность JWT, управление сессиями и многофакторная аутентификация.',
		},
		uz: {
			title: 'Autentifikatsiya va avtorizatsiya asoslari',
			description: 'RBAC, ABAC, OAuth 2.0, JWT xavfsizligi, sessiya boshqaruvi va ko\'p faktorli autentifikatsiya.',
		},
	},
};
