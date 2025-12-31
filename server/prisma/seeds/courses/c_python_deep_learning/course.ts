import { CourseMeta } from '../../types';

const course: CourseMeta = {
	slug: 'python-deep-learning',
	title: 'Deep Learning with PyTorch',
	description:
		'Master deep learning from neural network basics to production deployment. Build CNNs, RNNs, and modern architectures with PyTorch.',
	category: 'language',
	icon: '🔥',
	estimatedTime: '35h',
	order: 21,
	translations: {
		ru: {
			title: 'Глубокое обучение с PyTorch',
			description:
				'Освойте глубокое обучение от основ нейросетей до продакшен деплоя. Создавайте CNN, RNN и современные архитектуры с PyTorch.',
		},
		uz: {
			title: 'PyTorch bilan chuqur o\'rganish',
			description:
				"Chuqur o'rganishni neyrosetka asoslaridan ishlab chiqarish deployigacha o'rganing. PyTorch bilan CNN, RNN va zamonaviy arxitekturalarni yarating.",
		},
	},
};

export default course;
