import { Topic } from '../../../../../types';
import { tasks } from './fundamentals/tasks';

export const fundamentalsTopic: Topic = {
	slug: 'sec-certs-overview',
	title: 'Security Certifications Overview',
	description: 'CISSP, CEH, CompTIA Security+, OSCP certification paths and requirements.',
	difficulty: 'easy',
	estimatedTime: '3h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Обзор сертификаций по безопасности',
			description: 'CISSP, CEH, CompTIA Security+, OSCP пути сертификации и требования.',
		},
		uz: {
			title: 'Xavfsizlik sertifikatlari sharhi',
			description: 'CISSP, CEH, CompTIA Security+, OSCP sertifikatlashtirish yo\'llari va talablari.',
		},
	},
};
