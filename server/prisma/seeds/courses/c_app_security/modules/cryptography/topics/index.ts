import { Topic } from '../../../../../types';
import { tasks } from './fundamentals/tasks';

export const fundamentalsTopic: Topic = {
	slug: 'sec-crypto-fundamentals',
	title: 'Cryptography Fundamentals',
	description: 'Hashing algorithms, symmetric/asymmetric encryption, TLS/SSL, password storage, and digital signatures.',
	difficulty: 'medium',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Основы криптографии',
			description: 'Алгоритмы хеширования, симметричное/асимметричное шифрование, TLS/SSL, хранение паролей и цифровые подписи.',
		},
		uz: {
			title: 'Kriptografiya asoslari',
			description: 'Xeshlash algoritmlari, simmetrik/asimmetrik shifrlash, TLS/SSL, parol saqlash va raqamli imzolar.',
		},
	},
};
