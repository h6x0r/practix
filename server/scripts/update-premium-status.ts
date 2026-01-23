/**
 * Script to update isPremium status on existing tasks
 * First 3 tasks in first 3 modules of each course are free
 * Everything else is premium
 */

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function updatePremiumStatus() {
  console.log('🔄 Updating task premium status...\n');

  // Get all courses with their modules, topics, and tasks
  const courses = await prisma.course.findMany({
    include: {
      modules: {
        orderBy: { order: 'asc' },
        include: {
          topics: {
            orderBy: { order: 'asc' },
            include: {
              tasks: {
                orderBy: { order: 'asc' },
              },
            },
          },
        },
      },
    },
  });

  let freeCount = 0;
  let premiumCount = 0;

  for (const course of courses) {
    console.log(`📚 Processing course: ${course.title}`);

    for (let moduleIndex = 0; moduleIndex < course.modules.length; moduleIndex++) {
      const module = course.modules[moduleIndex];
      let taskIndexInModule = 0;

      for (const topic of module.topics) {
        for (const task of topic.tasks) {
          // First 3 modules (index 0, 1, 2) have first 3 tasks free
          const shouldBePremium = !(moduleIndex < 3 && taskIndexInModule < 3);

          // Update only if different
          if (task.isPremium !== shouldBePremium) {
            await prisma.task.update({
              where: { id: task.id },
              data: { isPremium: shouldBePremium },
            });
          }

          if (shouldBePremium) {
            premiumCount++;
          } else {
            freeCount++;
          }

          taskIndexInModule++;
        }
      }
    }
  }

  console.log(`\n✅ Update complete!`);
  console.log(`   🆓 Free tasks: ${freeCount}`);
  console.log(`   💎 Premium tasks: ${premiumCount}`);
}

updatePremiumStatus()
  .catch((error) => {
    console.error('❌ Error updating premium status:', error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
