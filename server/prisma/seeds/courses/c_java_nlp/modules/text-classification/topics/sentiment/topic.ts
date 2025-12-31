import { Topic } from '../../../../../../types';

const topic: Topic = {
	slug: 'sentiment',
	title: 'Sentiment Analysis',
	description: 'Analyze sentiment and emotion in text.',
	order: 1,
	isPremium: false,
	translations: {
		ru: {
			title: 'Анализ тональности',
			description: 'Анализируйте тональность и эмоции в тексте.',
		},
		uz: {
			title: 'Sentiment tahlili',
			description: 'Matndagi sentimentni va hissiyotni tahlil qiling.',
		},
	},
};

export default topic;
