import { Module } from '../../../../types';
import { fundamentalsTopic } from './topics';

export const owaspTop10Module: Module = {
	slug: 'sec-owasp-top-10',
	title: 'OWASP Top 10',
	description: 'Learn and defend against the OWASP Top 10 vulnerabilities: Injection, XSS, CSRF, Broken Auth, Security Misconfiguration, and more.',
	section: 'security',
	order: 3,
	difficulty: 'medium',
	estimatedTime: '8h',
	topics: [fundamentalsTopic],
	translations: {
		ru: {
			title: 'OWASP Top 10',
			description: 'Изучите и защититесь от OWASP Top 10 уязвимостей: Injection, XSS, CSRF, Broken Auth, Security Misconfiguration и других.'
		},
		uz: {
			title: 'OWASP Top 10',
			description: 'OWASP Top 10 zaifliklarini o\'rganing va ularga qarshi himoyalaning: Injection, XSS, CSRF, Broken Auth, Security Misconfiguration va boshqalar.'
		}
	}
};
