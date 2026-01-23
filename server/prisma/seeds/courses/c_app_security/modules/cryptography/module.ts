import { Module } from '../../../../types';
import { fundamentalsTopic } from './topics';

export const cryptographyModule: Module = {
	slug: 'sec-cryptography',
	title: 'Cryptography Basics',
	description: 'Learn cryptography fundamentals: hashing, symmetric/asymmetric encryption, TLS, password storage, and digital signatures.',
	section: 'security',
	order: 4,
	difficulty: 'medium',
	estimatedTime: '5h',
	topics: [fundamentalsTopic],
	translations: {
		ru: {
			title: 'Основы криптографии',
			description: 'Изучите основы криптографии: хеширование, симметричное/асимметричное шифрование, TLS, хранение паролей и цифровые подписи.'
		},
		uz: {
			title: 'Kriptografiya asoslari',
			description: 'Kriptografiya asoslarini o\'rganing: xeshlash, simmetrik/asimmetrik shifrlash, TLS, parol saqlash va raqamli imzolar.'
		}
	}
};
