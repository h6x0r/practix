import { Module } from "@nestjs/common";
import { PromoCodesService } from "./promocodes.service";
import { PromoCodesController } from "./promocodes.controller";
import { PromoCodesPublicController } from "./promocodes-public.controller";
import { PrismaModule } from "../prisma/prisma.module";

@Module({
  imports: [PrismaModule],
  controllers: [PromoCodesController, PromoCodesPublicController],
  providers: [PromoCodesService],
  exports: [PromoCodesService],
})
export class PromoCodesModule {}
