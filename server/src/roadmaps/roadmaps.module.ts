
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { RoadmapsController } from './roadmaps.controller';
import { RoadmapsService } from './roadmaps.service';
import { PrismaModule } from '../prisma/prisma.module';

@Module({
  imports: [PrismaModule, ConfigModule],
  controllers: [RoadmapsController],
  providers: [RoadmapsService],
  exports: [RoadmapsService],
})
export class RoadmapsModule {}
