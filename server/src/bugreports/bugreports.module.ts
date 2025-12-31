import { Module } from '@nestjs/common';
import { BugReportsController } from './bugreports.controller';
import { BugReportsService } from './bugreports.service';

@Module({
  controllers: [BugReportsController],
  providers: [BugReportsService],
  exports: [BugReportsService],
})
export class BugReportsModule {}
