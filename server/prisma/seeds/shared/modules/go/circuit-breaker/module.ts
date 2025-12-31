import { Module } from '../../../../types';
import { topics } from './topics';

export const circuitBreakerModule: Module = {
	title: 'Circuit Breaker',
	description: 'Build resilient systems with circuit breaker patterns.',
	section: 'production-patterns',
	order: 8,
	topics,
	translations: {
		ru: {
			title: 'Автоматический выключатель',
			description: 'Создавайте устойчивые системы с использованием паттернов автоматического выключателя.'
		},
		uz: {
			title: 'Circuit Breaker',
			description: 'Circuit breaker patternlari yordamida barqaror tizimlar yarating.'
		}
	}
};
