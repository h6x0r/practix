import { Module } from '../../../../types';
import { fundamentalsTopic } from './topics';

export const authAuthzModule: Module = {
	slug: 'sec-auth-authz',
	title: 'Authentication & Authorization',
	description: 'Master authentication and authorization: RBAC, ABAC, OAuth 2.0, JWT security, session management, and MFA implementation.',
	section: 'security',
	order: 2,
	difficulty: 'medium',
	estimatedTime: '6h',
	topics: [fundamentalsTopic],
	translations: {
		ru: {
			title: 'Аутентификация и авторизация',
			description: 'Освойте аутентификацию и авторизацию: RBAC, ABAC, OAuth 2.0, безопасность JWT, управление сессиями и реализация MFA.'
		},
		uz: {
			title: 'Autentifikatsiya va avtorizatsiya',
			description: 'Autentifikatsiya va avtorizatsiyani o\'rganing: RBAC, ABAC, OAuth 2.0, JWT xavfsizligi, sessiya boshqaruvi va MFA amalga oshirish.'
		}
	}
};
