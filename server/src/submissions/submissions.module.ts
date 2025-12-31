import { Module } from '@nestjs/common';
import { SubmissionsController } from './submissions.controller';
import { SubmissionsService } from './submissions.service';
import { QueueModule } from '../queue/queue.module';
import { TasksModule } from '../tasks/tasks.module';
import { SubscriptionsModule } from '../subscriptions/subscriptions.module';
import { GamificationModule } from '../gamification/gamification.module';

@Module({
  imports: [QueueModule, TasksModule, SubscriptionsModule, GamificationModule],
  controllers: [SubmissionsController],
  providers: [SubmissionsService],
})
export class SubmissionsModule {}
