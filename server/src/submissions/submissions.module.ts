import { Module } from '@nestjs/common';
import { SubmissionsController } from './submissions.controller';
import { SubmissionsService } from './submissions.service';
import { JudgeModule } from '../judge/judge.module';
import { TasksModule } from '../tasks/tasks.module';

@Module({
  imports: [JudgeModule, TasksModule],
  controllers: [SubmissionsController],
  providers: [SubmissionsService],
})
export class SubmissionsModule {}
