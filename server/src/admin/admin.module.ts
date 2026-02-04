import { Module } from "@nestjs/common";
import { AdminController } from "./admin.controller";
import { AdminService } from "./admin.service";
import { SettingsController } from "./settings/settings.controller";
import { SettingsService } from "./settings/settings.service";
import { PrismaModule } from "../prisma/prisma.module";
import { CacheModule } from "../cache/cache.module";

@Module({
  imports: [PrismaModule, CacheModule],
  controllers: [AdminController, SettingsController],
  providers: [AdminService, SettingsService],
  exports: [AdminService, SettingsService],
})
export class AdminModule {}
