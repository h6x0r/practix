import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'named-entity-recognition',
	title: 'Named Entity Recognition',
	description: 'Extract entities like names, locations, and organizations from text.',
	order: 4,
	isPremium: false,
	translations: {
		ru: {
			title: 'Распознавание именованных сущностей',
			description: 'Извлекайте сущности: имена, места и организации из текста.',
		},
		uz: {
			title: "Nomlangan ob'ektlarni aniqlash",
			description: "Matndan ismlar, joylar va tashkilotlar kabi ob'ektlarni ajratib oling.",
		},
	},
};

export default module;
