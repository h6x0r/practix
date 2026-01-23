import { Topic } from '../../../../../types';
import { tasks } from './fundamentals/tasks';

export const fundamentalsTopic: Topic = {
	slug: 'sec-owasp-vulnerabilities',
	title: 'OWASP Top 10 Vulnerabilities',
	description: 'SQL Injection, XSS, CSRF, Broken Authentication, Security Misconfiguration, XXE, IDOR, and more.',
	difficulty: 'medium',
	estimatedTime: '8h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'OWASP Top 10 уязвимости',
			description: 'SQL Injection, XSS, CSRF, Broken Authentication, Security Misconfiguration, XXE, IDOR и другие.',
		},
		uz: {
			title: 'OWASP Top 10 zaifliklar',
			description: 'SQL Injection, XSS, CSRF, Broken Authentication, Security Misconfiguration, XXE, IDOR va boshqalar.',
		},
	},
};
