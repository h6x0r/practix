import { Module } from '../../../../types';
import { topics } from './topics';

export const channelsModule: Module = {
	title: 'Channel Patterns',
	description: 'Master advanced channel patterns including fan-in, fan-out, worker pools, and pipeline processing for scalable concurrency.',
	section: 'concurrency',
	order: 14,
	topics,
	translations: {
		ru: {
			title: 'Паттерны каналов',
			description: 'Освойте продвинутые паттерны каналов, включая fan-in, fan-out, пулы воркеров и конвейерную обработку для масштабируемой конкурентности.'
		},
		uz: {
			title: 'Kanal naqshlari',
			description: 'Fan-in, fan-out, ishchi havuzlar va miqyoslanadigan parallellik uchun konveyer qayta ishlashni o\'z ichiga olgan ilg\'or kanal naqshlarini o\'zlashtiring.'
		}
	}
};
