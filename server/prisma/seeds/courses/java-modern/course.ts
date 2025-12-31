import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'java-modern',
	title: 'Java Modern',
	description: 'Modern Java features: generics, lambdas, Stream API, Optional, Date/Time API, records, sealed classes, and pattern matching.',
	category: 'language',
	icon: '✨',
	estimatedTime: '14h',
	order: 6,
	translations: {
		ru: {
			title: 'Современная Java',
			description: 'Современные возможности Java: дженерики, лямбды, Stream API, Optional, Date/Time API, записи, запечатанные классы и сопоставление с образцом.'
		},
		uz: {
			title: 'Zamonaviy Java',
			description: 'Zamonaviy Java imkoniyatlari: generiklar, lambda, Stream API, Optional, Date/Time API, recordlar, sealed klasslar va pattern matching.'
		}
	}
};
