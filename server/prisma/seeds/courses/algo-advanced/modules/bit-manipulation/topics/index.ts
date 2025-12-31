import { Topic } from '../../../../../types';
import { tasks } from './techniques/tasks';

export const techniquesTopic: Topic = {
	slug: 'bit-manipulation-techniques',
	title: 'Bit Manipulation Techniques',
	description: 'Master bitwise operations: XOR tricks, bit counting, masking, and building arithmetic from logic gates',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Техники битовых манипуляций',
			description: 'Освойте побитовые операции: трюки с XOR, подсчёт битов, маскирование и построение арифметики из логических вентилей'
		},
		uz: {
			title: 'Bit manipulyatsiya texnikalari',
			description: "Bitli operatsiyalarni o'rganing: XOR hiylalari, bitlarni sanash, maskalash va mantiqiy eshiklardan arifmetika qurish"
		}
	}
};

export const topics = [techniquesTopic];

export default topics;
