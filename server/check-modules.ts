import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function checkModules() {
  const count = await prisma.module.count();
  console.log('Total modules:', count);

  const configSafetyModules = await prisma.module.findMany({
    where: { section: 'configuration-safety' },
    orderBy: { order: 'asc' },
    select: { title: true, order: true }
  });

  console.log('\nConfiguration & Safety modules:');
  configSafetyModules.forEach(m => {
    console.log(`  Order ${m.order}: ${m.title}`);
  });

  await prisma.$disconnect();
}

checkModules();
