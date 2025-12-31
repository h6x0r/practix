import { Module } from '@nestjs/common';
import { UserCoursesService } from './user-courses.service';
import { UserCoursesController } from './user-courses.controller';
import { PrismaModule } from '../prisma/prisma.module';

@Module({
  imports: [PrismaModule],
  controllers: [UserCoursesController],
  providers: [UserCoursesService],
  exports: [UserCoursesService],
})
export class UserCoursesModule {}
