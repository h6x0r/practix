import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class CoursesService {
  constructor(private prisma: PrismaService) {}

  async findAll(userId?: string) {
    const courses = await (this.prisma as any).course.findMany({
      orderBy: { order: 'asc' }, // Ensure courses have an order or sort by createdAt
      include: {
        modules: {
          include: {
            topics: {
              include: {
                tasks: { select: { id: true } }
              }
            }
          }
        }
      }
    });

    // Calculate progress roughly (completed tasks / total tasks)
    return Promise.all(courses.map(async (c) => {
        let totalTasks = 0;
        c.modules.forEach(m => m.topics.forEach(t => totalTasks += t.tasks.length));
        
        let completed = 0;
        if (userId && totalTasks > 0) {
            const completedCount = await (this.prisma as any).submission.count({
                where: {
                    userId,
                    status: 'passed',
                    task: { topic: { module: { courseId: c.id } } }
                }
            });
            completed = completedCount;
        }

        const progress = totalTasks === 0 ? 0 : Math.round((completed / totalTasks) * 100);

        return {
            id: c.slug, // Use slug as ID for frontend compatibility
            uuid: c.id,
            title: c.title,
            description: c.description,
            category: c.category,
            icon: c.icon,
            estimatedTime: c.estimatedTime,
            totalTopics: c.modules.reduce((acc, m) => acc + m.topics.length, 0),
            progress
        };
    }));
  }

  async findOne(slug: string, userId?: string) {
    const course = await (this.prisma as any).course.findUnique({
      where: { slug },
    });

    if (!course) {
        // Fallback for ID lookup if slug fails
        return (this.prisma as any).course.findUnique({ where: { id: slug } });
    }
    
    // We reuse findAll logic for single entity hydration if needed, 
    // or just return the metadata matching frontend type
    const list = await this.findAll(userId);
    return list.find(c => c.id === slug);
  }

  async getStructure(slug: string, userId?: string) {
    const course = await (this.prisma as any).course.findUnique({
      where: { slug },
      include: {
        modules: {
          orderBy: { order: 'asc' },
          include: {
            topics: {
              orderBy: { order: 'asc' },
              include: {
                tasks: {
                  orderBy: { order: 'asc' },
                  select: {
                    id: true,
                    slug: true,
                    title: true,
                    difficulty: true,
                    estimatedTime: true,
                    isPremium: true
                  }
                }
              }
            }
          }
        }
      }
    });

    if (!course) throw new NotFoundException('Course not found');

    // Get completion status for all tasks in this course
    let completedTaskIds: string[] = [];
    if (userId) {
        const subs = await (this.prisma as any).submission.findMany({
            where: {
                userId,
                status: 'passed',
                task: { topic: { module: { courseId: course.id } } }
            },
            select: { taskId: true }
        });
        completedTaskIds = subs.map(s => s.taskId);
    }

    // Map to Frontend Structure
    return course.modules.map(m => ({
        id: m.id,
        title: m.title,
        description: m.description,
        section: m.section || 'core',
        topics: m.topics.map(t => ({
            id: t.id,
            title: t.title,
            description: t.description,
            difficulty: t.difficulty,
            estimatedTime: t.estimatedTime,
            tasks: t.tasks.map(task => ({
                ...task,
                status: completedTaskIds.includes(task.id) ? 'completed' : 'pending'
            }))
        }))
    }));
  }
}