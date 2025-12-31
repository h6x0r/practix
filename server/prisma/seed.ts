/**
 * KODLA Platform Database Seeder
 * Seeds all courses from the new /courses structure
 *
 * Courses:
 * - Go Basics, Go Concurrency, Go Web & APIs, Go Production
 * - Java Core, Java Modern, Java Advanced
 */

// @ts-ignore
import { PrismaClient } from '@prisma/client';
import { ALL_COURSES } from './seeds/courses';
import { Course, Module, Topic, Task } from './seeds/types';
import { seedBadges } from './seeds/badges';
import * as bcrypt from 'bcrypt';
import Redis from 'ioredis';

const prisma = new PrismaClient();

// Redis client for cache invalidation
let redis: Redis | null = null;

async function invalidateCoursesCache(): Promise<void> {
	try {
		const redisHost = process.env.REDIS_HOST || 'redis';
		const redisPort = parseInt(process.env.REDIS_PORT || '6379', 10);

		redis = new Redis({
			host: redisHost,
			port: redisPort,
			maxRetriesPerRequest: 1,
			connectTimeout: 3000,
		});

		// Delete all course cache keys
		const keys = await redis.keys('courses:*');
		if (keys.length > 0) {
			await redis.del(...keys);
			console.log(`🗑️  Invalidated ${keys.length} cache entries`);
		} else {
			console.log('📭 No cache entries to invalidate');
		}

		await redis.quit();
		redis = null;
	} catch (error) {
		console.log('⚠️  Cache invalidation skipped (Redis not available)');
		if (redis) {
			try { await redis.quit(); } catch {}
			redis = null;
		}
	}
}

// ============================================================================
// Helper: Extract string from plain string or multi-language object
// Some Java topics use { en: '...', ru: '...', uz: '...' } format
// ============================================================================
type MultiLangString = string | { en: string; ru?: string; uz?: string };

function extractString(value: MultiLangString | undefined, fallback: string = ''): string {
	if (!value) return fallback;
	if (typeof value === 'string') return value;
	if (typeof value === 'object' && 'en' in value) return value.en;
	return fallback;
}

// ============================================================================
// Helper: Parse time string to minutes (e.g., "15m" -> 15, "1h 30m" -> 90, "2h" -> 120)
// ============================================================================
function parseTimeToMinutes(time: string): number {
	if (!time) return 15; // Default 15 minutes

	let totalMinutes = 0;

	// Handle formats: "15m", "1h", "1h 30m", "15-20m", "1.5h"
	const hourMatch = time.match(/(\d+(?:\.\d+)?)\s*h/i);
	const minMatch = time.match(/(\d+)(?:-\d+)?\s*m/i);

	if (hourMatch) {
		totalMinutes += parseFloat(hourMatch[1]) * 60;
	}
	if (minMatch) {
		totalMinutes += parseInt(minMatch[1]);
	}

	// If nothing matched, try to parse as plain number (assume minutes)
	if (totalMinutes === 0) {
		const plainNum = parseInt(time);
		if (!isNaN(plainNum)) totalMinutes = plainNum;
		else totalMinutes = 15; // Default fallback
	}

	return totalMinutes;
}

// ============================================================================
// Helper: Format minutes to readable time string (e.g., 90 -> "1h 30m", 45 -> "45m")
// IMPORTANT: Always use English format - frontend handles localization
// ============================================================================
function formatMinutesToTime(minutes: number): string {
	if (minutes <= 0) return '15m';

	const hours = Math.floor(minutes / 60);
	const mins = minutes % 60;

	if (hours > 0 && mins > 0) {
		return `${hours}h ${mins}m`;
	} else if (hours > 0) {
		return `${hours}h`;
	} else {
		return `${mins}m`;
	}
}

// ============================================================================
// Helper: Calculate total estimated time for a topic from all its tasks
// Only counts valid tasks (with slug) to ensure consistency
// ============================================================================
function calculateTopicTime(topic: Topic): string {
	let totalMinutes = 0;

	for (const task of topic.tasks || []) {
		if (task && task.slug) {
			totalMinutes += parseTimeToMinutes(task.estimatedTime || '15m');
		}
	}

	// Default to 30m if no valid tasks
	return totalMinutes > 0 ? formatMinutesToTime(totalMinutes) : '30m';
}

// ============================================================================
// Helper: Calculate total estimated time for a module from all its tasks
// If no tasks exist, use topic count × 30m as fallback for consistency
// ============================================================================
function calculateModuleTime(module: Module): string {
	let totalMinutes = 0;
	let hasValidTasks = false;

	for (const topic of module.topics || []) {
		for (const task of topic.tasks || []) {
			if (task && task.slug) {
				totalMinutes += parseTimeToMinutes(task.estimatedTime || '15m');
				hasValidTasks = true;
			}
		}
	}

	// If no valid tasks, use topic count × 30m (matches topic default)
	if (!hasValidTasks) {
		const topicCount = (module.topics || []).length;
		totalMinutes = topicCount * 30; // 30m per topic default
	}

	// Default to 30m if no topics either
	return totalMinutes > 0 ? formatMinutesToTime(totalMinutes) : '30m';
}

// ============================================================================
// Helper: Calculate total estimated time for a course from all its tasks
// ============================================================================
function calculateCourseTime(course: Course): string {
	let totalMinutes = 0;

	for (const module of course.modules || []) {
		if (!module || !module.topics) continue;
		for (const topic of module.topics) {
			if (!topic || !topic.tasks) continue;
			for (const task of topic.tasks) {
				if (task && task.slug) {
					totalMinutes += parseTimeToMinutes(task.estimatedTime || '15m');
				}
			}
		}
	}

	// Default to 1h if no valid tasks
	return totalMinutes > 0 ? formatMinutesToTime(totalMinutes) : '1h';
}

// ============================================================================
// DEPRECATED: Old course definitions (kept for reference)
// ============================================================================
// @deprecated - Use ALL_COURSES from ./seeds/courses instead
// const GO_COURSE = { slug: 'c_go', title: 'Go Language Mastery', ... };
// const GO_PATTERNS_COURSE = { slug: 'c_go_patterns', ... };
// const JAVA_COURSE = { slug: 'c_java', ... };
// const ALGO_COURSE = { slug: 'c_algo', ... };
// const SYS_COURSE = { slug: 'c_sys', ... };
// ============================================================================

/**
 * Seed test users for development and testing
 */
async function seedTestUsers(): Promise<void> {
	console.log('👤 Seeding test users...');

	// Check if test user already exists
	const existingUser = await prisma.user.findUnique({
		where: { email: 'alex@example.com' },
	});

	if (existingUser) {
		console.log('   ⚠️  Test user alex@example.com already exists, skipping...');
		return;
	}

	// Hash the password
	const hashedPassword = await bcrypt.hash('12345', 10);

	// Create test user
	const testUser = await prisma.user.create({
		data: {
			email: 'alex@example.com',
			password: hashedPassword,
			name: 'Alex',
			isPremium: true,
			preferences: {
				editorFontSize: 14,
				editorMinimap: false,
				editorTheme: 'vs-dark',
				editorLineNumbers: true,
				notifications: {
					emailDigest: true,
					newCourses: true,
				},
			},
		},
	});

	console.log('   ✅ Test user created: alex@example.com (password: 12345)\n');
}

/**
 * Assign subscription to test user (called after subscription plans are seeded)
 */
async function assignTestUserSubscription(): Promise<void> {
	const testUser = await prisma.user.findUnique({
		where: { email: 'alex@example.com' },
	});

	if (!testUser) return;

	// Check if already has subscription
	const existingSub = await prisma.subscription.findFirst({
		where: { userId: testUser.id },
	});

	if (existingSub) return;

	const globalPlan = await prisma.subscriptionPlan.findFirst({
		where: { type: 'global' },
	});

	if (globalPlan) {
		await prisma.subscription.create({
			data: {
				userId: testUser.id,
				planId: globalPlan.id,
				status: 'active',
				startDate: new Date(),
				endDate: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000), // 1 year
			},
		});
		console.log('   ✅ Test user subscription activated\n');
	}
}

/**
 * Seed a single task into the database
 */
async function seedTask(task: Task, topicId: string): Promise<void> {
	// Skip invalid tasks
	if (!task || !task.slug) {
		console.warn(`   ⚠️  Skipping invalid task (no slug)`);
		return;
	}

	// Extract string values (handles both plain strings and { en, ru, uz } objects)
	const title = extractString(task.title as MultiLangString, 'Untitled Task');
	const description = extractString(task.description as MultiLangString, '');
	const hint1 = extractString(task.hint1 as MultiLangString, '');
	const hint2 = extractString(task.hint2 as MultiLangString, '');
	const whyItMatters = extractString(task.whyItMatters as MultiLangString, '');
	const initialCode = extractString(task.initialCode as MultiLangString, '');
	const solutionCode = extractString(task.solutionCode as MultiLangString, '');

	// Build translations from multilingual objects if translations field is missing
	let translations = task.translations as any;
	if (!translations && typeof task.title === 'object') {
		// Task uses multilingual object format - build translations from it
		const taskAny = task as any;
		translations = {
			ru: {
				title: taskAny.title?.ru || title,
				description: taskAny.description?.ru || description,
				solutionCode: taskAny.solutionCode?.ru || solutionCode,
				hint1: Array.isArray(taskAny.hints?.ru) ? taskAny.hints.ru[0] || '' : '',
				hint2: Array.isArray(taskAny.hints?.ru) ? taskAny.hints.ru[1] || '' : '',
				whyItMatters: '',
			},
			uz: {
				title: taskAny.title?.uz || title,
				description: taskAny.description?.uz || description,
				solutionCode: taskAny.solutionCode?.uz || solutionCode,
				hint1: Array.isArray(taskAny.hints?.uz) ? taskAny.hints.uz[0] || '' : '',
				hint2: Array.isArray(taskAny.hints?.uz) ? taskAny.hints.uz[1] || '' : '',
				whyItMatters: '',
			},
		};
	}

	// Extract hints from array format if present
	const taskAny = task as any;
	const extractedHint1 = hint1 || (Array.isArray(taskAny.hints?.en) ? taskAny.hints.en[0] || '' : '');
	const extractedHint2 = hint2 || (Array.isArray(taskAny.hints?.en) ? taskAny.hints.en[1] || '' : '');

	await prisma.task.create({
		data: {
			slug: task.slug,
			title,
			difficulty: task.difficulty || 'medium',
			tags: task.tags || [],
			estimatedTime: task.estimatedTime || '15m',
			isPremium: task.isPremium || false,
			youtubeUrl: task.youtubeUrl || '',
			description,
			initialCode: initialCode || (task as any).template || '',
			solutionCode,
			testCode: task.testCode || '',
			hint1: extractedHint1,
			hint2: extractedHint2,
			whyItMatters,
			order: task.order || 0,
			topicId,
			// ML visualization support
			visualizationType: (task as any).visualizationType || null,
			// Translations stored as JSON
			translations,
		},
	});
}

/**
 * Seed a single topic with all its tasks
 */
async function seedTopic(topic: Topic, moduleId: string): Promise<number> {
	// Skip invalid topics
	if (!topic || !topic.title) {
		console.warn(`   ⚠️  Skipping invalid topic (no title)`);
		return 0;
	}

	// Extract string values (handles both plain strings and { en, ru, uz } objects)
	const title = extractString(topic.title as MultiLangString, 'Untitled Topic');
	const description = extractString(topic.description as MultiLangString, '');

	// Calculate topic time from sum of all task times
	const estimatedTime = calculateTopicTime(topic);

	const createdTopic = await prisma.topic.create({
		data: {
			title,
			description,
			difficulty: topic.difficulty || 'medium', // Default to medium if not specified
			estimatedTime, // Calculated from sum of task times
			order: topic.order || 0,
			moduleId,
			// Save translations for topics
			translations: (topic as any).translations as any,
		},
	});

	// Seed all tasks for this topic
	const tasks = topic.tasks || [];
	for (const task of tasks) {
		await seedTask(task, createdTopic.id);
	}

	return tasks.filter(t => t && t.slug).length;
}

/**
 * Seed a single module with all its topics
 */
async function seedModule(module: Module, courseId: string): Promise<number> {
	// Skip invalid modules
	if (!module || !module.title) {
		console.warn(`   ⚠️  Skipping invalid module (no title)`);
		return 0;
	}

	// Extract string values (handles both plain strings and { en, ru, uz } objects)
	const title = extractString(module.title as MultiLangString, 'Untitled Module');
	const description = extractString(module.description as MultiLangString, '');

	// Build translations from either translations field or {en, ru, uz} format
	let translations = (module as any).translations;
	if (!translations && typeof module.title === 'object') {
		// Convert {en, ru, uz} format to translations format
		const titleObj = module.title as any;
		const descObj = module.description as any;
		translations = {
			ru: { title: titleObj.ru || title, description: descObj?.ru || description },
			uz: { title: titleObj.uz || title, description: descObj?.uz || description },
		};
	}

	// Calculate estimated time from all tasks in this module
	const estimatedTime = calculateModuleTime(module);

	const createdModule = await prisma.module.create({
		data: {
			title,
			description,
			section: module.section || 'core', // Default section
			estimatedTime, // Calculated from sum of task times
			order: module.order || 0,
			courseId,
			// Save translations for modules
			translations: translations as any,
		},
	});

	// Seed all topics for this module
	let taskCount = 0;
	const topics = module.topics || [];
	for (const topic of topics) {
		const count = await seedTopic(topic, createdModule.id);
		taskCount += count;
	}

	return taskCount;
}

/**
 * Seed a single course with all its modules
 */
async function seedCourse(course: Course): Promise<{ moduleCount: number; taskCount: number }> {
	// Calculate course time from sum of all task times
	const estimatedTime = calculateCourseTime(course);

	// Create or update the course
	const createdCourse = await prisma.course.upsert({
		where: { slug: course.slug },
		update: {
			title: course.title,
			description: course.description,
			category: course.category,
			icon: course.icon,
			estimatedTime, // Calculated from sum of all task times
			order: course.order,
			// Save translations for courses
			translations: (course as any).translations as any,
		},
		create: {
			slug: course.slug,
			title: course.title,
			description: course.description,
			category: course.category,
			icon: course.icon,
			estimatedTime, // Calculated from sum of all task times
			order: course.order,
			// Save translations for courses
			translations: (course as any).translations as any,
		},
	});

	// Seed all modules for this course
	let totalTasks = 0;
	for (const module of course.modules) {
		const taskCount = await seedModule(module, createdCourse.id);
		totalTasks += taskCount;
	}

	return {
		moduleCount: course.modules.length,
		taskCount: totalTasks,
	};
}

/**
 * Seed subscription plans for all courses
 */
async function seedSubscriptionPlans(): Promise<void> {
	console.log('💳 Seeding subscription plans...');

	// Get all courses
	const courses = await prisma.course.findMany({
		select: { id: true, slug: true, title: true, translations: true },
	});

	// Check if global plan exists
	const existingGlobal = await prisma.subscriptionPlan.findUnique({
		where: { slug: 'global' },
	});

	if (!existingGlobal) {
		// Create global subscription plan
		await prisma.subscriptionPlan.create({
			data: {
				slug: 'global',
				name: 'All Courses',
				nameRu: 'Все курсы',
				type: 'global',
				priceMonthly: 9900000, // 99,000 UZS in tiyn
				currency: 'UZS',
				isActive: true,
			},
		});
		console.log('   ✅ Created global subscription plan');
	} else {
		console.log('   ⚠️  Global plan already exists, skipping...');
	}

	// Create course-specific plans
	let createdCount = 0;
	for (const course of courses) {
		const existingPlan = await prisma.subscriptionPlan.findUnique({
			where: { slug: course.slug },
		});

		if (!existingPlan) {
			const translations = course.translations as any;
			await prisma.subscriptionPlan.create({
				data: {
					slug: course.slug,
					name: course.title,
					nameRu: translations?.ru?.title || course.title,
					type: 'course',
					courseId: course.id,
					priceMonthly: 4900000, // 49,000 UZS in tiyn
					currency: 'UZS',
					isActive: true,
				},
			});
			createdCount++;
		}
	}

	if (createdCount > 0) {
		console.log(`   ✅ Created ${createdCount} course subscription plans\n`);
	} else {
		console.log('   ⚠️  All course plans already exist, skipping...\n');
	}
}

/**
 * Main seeding function
 */
async function main() {
	console.log('🌱 Starting KODLA seed...');
	console.log(`📚 Found ${ALL_COURSES.length} courses to seed\n`);

	// Seed test users first
	await seedTestUsers();

	// Seed gamification badges
	await seedBadges();

	// Get existing courses
	const existingCourses = await prisma.course.findMany({
		select: { slug: true, title: true },
	});
	const existingSlugs = new Set(existingCourses.map(c => c.slug));

	// Filter to only new courses that don't exist in the database
	const newCourses = ALL_COURSES.filter(c => !existingSlugs.has(c.slug));

	if (existingCourses.length > 0) {
		console.log('📋 Existing courses:');
		existingCourses.forEach((c, i) => console.log(`   ${i + 1}. ${c.title} (${c.slug})`));
		console.log('');
	}

	if (newCourses.length === 0) {
		console.log('✅ All courses already exist in database.');
		// Still seed subscription plans for existing courses
		await seedSubscriptionPlans();
		await assignTestUserSubscription();
		// Invalidate cache in case content was updated
		await invalidateCoursesCache();
		console.log('💡 Tip: Use `make db-refresh` to reset and reseed the database.');
		return;
	}

	console.log(`🆕 Found ${newCourses.length} new course(s) to add:\n`);

	// Seed only new courses
	let totalModules = 0;
	let totalTasks = 0;

	for (const course of newCourses) {
		console.log(`📚 Seeding: ${course.title}`);

		const { moduleCount, taskCount } = await seedCourse(course);
		totalModules += moduleCount;
		totalTasks += taskCount;

		console.log(`   ✅ ${moduleCount} modules, ${taskCount} tasks\n`);
	}

	// Seed subscription plans (after courses are seeded)
	await seedSubscriptionPlans();
	await assignTestUserSubscription();

	// Invalidate cache after seeding
	await invalidateCoursesCache();

	// Summary
	console.log('═'.repeat(50));
	console.log('✅ Seeding finished!');
	console.log(`   📚 New Courses: ${newCourses.length}`);
	console.log(`   📦 Modules: ${totalModules}`);
	console.log(`   📝 Tasks: ${totalTasks}`);
	console.log('═'.repeat(50));
}

main()
	.catch((e) => {
		console.error('❌ Seed failed:', e);
		process.exit(1);
	})
	.finally(async () => {
		await prisma.$disconnect();
	});
