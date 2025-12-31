import { Module } from '../../../../types';
import { techniquesTopic } from './topics';

export const bitManipulationModule: Module = {
	slug: 'algo-bit-manipulation',
	title: 'Bit Manipulation',
	description: 'Master bitwise operations: XOR for unique elements, bit counting tricks, binary arithmetic, and low-level optimizations',
	section: 'algorithms',
	order: 6,
	difficulty: 'medium',
	estimatedTime: '4h',
	topics: [techniquesTopic],
	translations: {
		ru: {
			title: 'Битовые манипуляции',
			description: 'Освойте побитовые операции: XOR для уникальных элементов, трюки подсчёта битов, двоичную арифметику и низкоуровневые оптимизации'
		},
		uz: {
			title: 'Bit manipulyatsiyalari',
			description: "Bitli operatsiyalarni o'rganing: noyob elementlar uchun XOR, bitlarni sanash hiylalari, ikkilik arifmetika va past darajadagi optimallashtirishlar"
		}
	}
};

export default bitManipulationModule;
