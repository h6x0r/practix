/**
 * gRPC Interceptors Module
 * Learn to build gRPC unary server interceptors for cross-cutting concerns
 */

import { Module } from '../../../types';
import { topics } from './topics';

export const grpcInterceptorsModule: Module = {
	title: 'gRPC Interceptors',
	description: 'Master gRPC unary server interceptors for logging, timeouts, retries, and middleware composition.',
	section: 'Microservices',
	order: 2,
	topics: topics,
	translations: {
		ru: {
			title: 'gRPC Interceptors',
			description: 'Освойте унарные серверные перехватчики gRPC для логирования, таймаутов, повторных попыток и композиции middleware.',
		},
		uz: {
			title: 'gRPC Interceptorlar',
			description: 'Logging, timeout, retry va middleware kompozitsiyasi uchun gRPC unary server interceptorlarini o\'rganing.',
		},
	},
};
