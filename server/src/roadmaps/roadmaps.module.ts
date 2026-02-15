import { Module } from "@nestjs/common";
import { ConfigModule } from "@nestjs/config";
import { RoadmapsController } from "./roadmaps.controller";
import { RoadmapsService } from "./roadmaps.service";
import { RoadmapAIService } from "./roadmap-ai.service";
import { RoadmapVariantsService } from "./roadmap-variants.service";
import { PrismaModule } from "../prisma/prisma.module";

@Module({
  imports: [PrismaModule, ConfigModule],
  controllers: [RoadmapsController],
  providers: [RoadmapsService, RoadmapAIService, RoadmapVariantsService],
  exports: [RoadmapsService],
})
export class RoadmapsModule {}
