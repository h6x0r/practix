import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Badge definitions
const BADGES = [
  // Milestone badges (based on tasks solved)
  {
    slug: 'first-task',
    name: 'First Steps',
    description: 'Complete your first task',
    icon: '🎯',
    category: 'milestone',
    requirement: 1,
    xpReward: 50,
    translations: {
      ru: { name: 'Первые шаги', description: 'Выполните первую задачу' },
      uz: { name: 'Birinchi qadamlar', description: 'Birinchi vazifani bajaring' },
    },
  },
  {
    slug: 'task-10',
    name: 'Getting Started',
    description: 'Complete 10 tasks',
    icon: '🌱',
    category: 'milestone',
    requirement: 10,
    xpReward: 100,
    translations: {
      ru: { name: 'Начало пути', description: 'Выполните 10 задач' },
      uz: { name: 'Yo\'l boshlanishi', description: '10 ta vazifani bajaring' },
    },
  },
  {
    slug: 'task-25',
    name: 'Rising Star',
    description: 'Complete 25 tasks',
    icon: '⭐',
    category: 'milestone',
    requirement: 25,
    xpReward: 200,
    translations: {
      ru: { name: 'Восходящая звезда', description: 'Выполните 25 задач' },
      uz: { name: 'Ko\'tarilayotgan yulduz', description: '25 ta vazifani bajaring' },
    },
  },
  {
    slug: 'task-50',
    name: 'Problem Solver',
    description: 'Complete 50 tasks',
    icon: '🧩',
    category: 'milestone',
    requirement: 50,
    xpReward: 300,
    translations: {
      ru: { name: 'Решатель задач', description: 'Выполните 50 задач' },
      uz: { name: 'Muammolarni hal qiluvchi', description: '50 ta vazifani bajaring' },
    },
  },
  {
    slug: 'task-100',
    name: 'Century Club',
    description: 'Complete 100 tasks',
    icon: '💯',
    category: 'milestone',
    requirement: 100,
    xpReward: 500,
    translations: {
      ru: { name: 'Клуб сотни', description: 'Выполните 100 задач' },
      uz: { name: 'Yuztalik klubi', description: '100 ta vazifani bajaring' },
    },
  },
  {
    slug: 'task-250',
    name: 'Code Warrior',
    description: 'Complete 250 tasks',
    icon: '⚔️',
    category: 'milestone',
    requirement: 250,
    xpReward: 1000,
    translations: {
      ru: { name: 'Воин кода', description: 'Выполните 250 задач' },
      uz: { name: 'Kod jangchisi', description: '250 ta vazifani bajaring' },
    },
  },
  {
    slug: 'task-500',
    name: 'Coding Legend',
    description: 'Complete 500 tasks',
    icon: '🏆',
    category: 'milestone',
    requirement: 500,
    xpReward: 2000,
    translations: {
      ru: { name: 'Легенда кодинга', description: 'Выполните 500 задач' },
      uz: { name: 'Kod afsonasi', description: '500 ta vazifani bajaring' },
    },
  },

  // Streak badges
  {
    slug: 'streak-3',
    name: 'On a Roll',
    description: 'Maintain a 3-day streak',
    icon: '🔥',
    category: 'streak',
    requirement: 3,
    xpReward: 50,
    translations: {
      ru: { name: 'В ударе', description: 'Поддерживайте серию 3 дня' },
      uz: { name: 'Faol davom', description: '3 kunlik seriyani saqlang' },
    },
  },
  {
    slug: 'streak-7',
    name: 'Week Warrior',
    description: 'Maintain a 7-day streak',
    icon: '🌟',
    category: 'streak',
    requirement: 7,
    xpReward: 100,
    translations: {
      ru: { name: 'Недельный воин', description: 'Поддерживайте серию 7 дней' },
      uz: { name: 'Haftalik jangchi', description: '7 kunlik seriyani saqlang' },
    },
  },
  {
    slug: 'streak-14',
    name: 'Two Week Champion',
    description: 'Maintain a 14-day streak',
    icon: '💪',
    category: 'streak',
    requirement: 14,
    xpReward: 200,
    translations: {
      ru: { name: 'Двухнедельный чемпион', description: 'Поддерживайте серию 14 дней' },
      uz: { name: 'Ikki haftalik chempion', description: '14 kunlik seriyani saqlang' },
    },
  },
  {
    slug: 'streak-30',
    name: 'Monthly Master',
    description: 'Maintain a 30-day streak',
    icon: '👑',
    category: 'streak',
    requirement: 30,
    xpReward: 500,
    translations: {
      ru: { name: 'Мастер месяца', description: 'Поддерживайте серию 30 дней' },
      uz: { name: 'Oylik usta', description: '30 kunlik seriyani saqlang' },
    },
  },
  {
    slug: 'streak-100',
    name: 'Unstoppable',
    description: 'Maintain a 100-day streak',
    icon: '🎖️',
    category: 'streak',
    requirement: 100,
    xpReward: 2000,
    translations: {
      ru: { name: 'Неудержимый', description: 'Поддерживайте серию 100 дней' },
      uz: { name: 'To\'xtab bo\'lmaydigan', description: '100 kunlik seriyani saqlang' },
    },
  },

  // Level badges
  {
    slug: 'level-5',
    name: 'Apprentice',
    description: 'Reach level 5',
    icon: '📘',
    category: 'level',
    requirement: 5,
    xpReward: 100,
    translations: {
      ru: { name: 'Ученик', description: 'Достигните 5 уровня' },
      uz: { name: 'Shogird', description: '5-darajaga yeting' },
    },
  },
  {
    slug: 'level-10',
    name: 'Journeyman',
    description: 'Reach level 10',
    icon: '📗',
    category: 'level',
    requirement: 10,
    xpReward: 250,
    translations: {
      ru: { name: 'Подмастерье', description: 'Достигните 10 уровня' },
      uz: { name: 'Hunarmand', description: '10-darajaga yeting' },
    },
  },
  {
    slug: 'level-15',
    name: 'Expert',
    description: 'Reach level 15',
    icon: '📙',
    category: 'level',
    requirement: 15,
    xpReward: 500,
    translations: {
      ru: { name: 'Эксперт', description: 'Достигните 15 уровня' },
      uz: { name: 'Ekspert', description: '15-darajaga yeting' },
    },
  },
  {
    slug: 'level-20',
    name: 'Master',
    description: 'Reach level 20',
    icon: '📕',
    category: 'level',
    requirement: 20,
    xpReward: 1000,
    translations: {
      ru: { name: 'Мастер', description: 'Достигните 20 уровня' },
      uz: { name: 'Usta', description: '20-darajaga yeting' },
    },
  },

  // XP badges
  {
    slug: 'xp-1000',
    name: 'XP Hunter',
    description: 'Earn 1,000 XP',
    icon: '💎',
    category: 'xp',
    requirement: 1000,
    xpReward: 100,
    translations: {
      ru: { name: 'Охотник за XP', description: 'Заработайте 1,000 XP' },
      uz: { name: 'XP ovchisi', description: '1,000 XP to\'plang' },
    },
  },
  {
    slug: 'xp-5000',
    name: 'XP Collector',
    description: 'Earn 5,000 XP',
    icon: '💰',
    category: 'xp',
    requirement: 5000,
    xpReward: 250,
    translations: {
      ru: { name: 'Коллекционер XP', description: 'Заработайте 5,000 XP' },
      uz: { name: 'XP kolleksioneri', description: '5,000 XP to\'plang' },
    },
  },
  {
    slug: 'xp-10000',
    name: 'XP Master',
    description: 'Earn 10,000 XP',
    icon: '🏅',
    category: 'xp',
    requirement: 10000,
    xpReward: 500,
    translations: {
      ru: { name: 'Мастер XP', description: 'Заработайте 10,000 XP' },
      uz: { name: 'XP ustasi', description: '10,000 XP to\'plang' },
    },
  },
];

export async function seedBadges() {
  console.log('Seeding badges...');

  for (const badge of BADGES) {
    await prisma.badge.upsert({
      where: { slug: badge.slug },
      update: {
        name: badge.name,
        description: badge.description,
        icon: badge.icon,
        category: badge.category,
        requirement: badge.requirement,
        xpReward: badge.xpReward,
        translations: badge.translations,
      },
      create: badge,
    });
  }

  console.log(`Seeded ${BADGES.length} badges`);
}

// Allow running directly
if (require.main === module) {
  seedBadges()
    .then(() => {
      console.log('Badge seeding complete');
      process.exit(0);
    })
    .catch((e) => {
      console.error('Badge seeding failed:', e);
      process.exit(1);
    });
}
