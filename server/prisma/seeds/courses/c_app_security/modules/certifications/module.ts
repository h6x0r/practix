import { Module } from '../../../../types';
import { fundamentalsTopic } from './topics';

export const certificationsModule: Module = {
	slug: 'sec-certifications',
	title: 'Security Certifications',
	description: 'Overview of popular security certifications: CISSP, CEH, CompTIA Security+, OSCP, and their requirements.',
	section: 'security',
	order: 6,
	difficulty: 'easy',
	estimatedTime: '3h',
	topics: [fundamentalsTopic],
	translations: {
		ru: {
			title: 'Сертификации по безопасности',
			description: 'Обзор популярных сертификаций по безопасности: CISSP, CEH, CompTIA Security+, OSCP и их требования.'
		},
		uz: {
			title: 'Xavfsizlik sertifikatlari',
			description: 'Mashhur xavfsizlik sertifikatlarining sharhi: CISSP, CEH, CompTIA Security+, OSCP va ularning talablari.'
		}
	}
};
