import { Module } from "@nestjs/common";
import { AiService } from "./ai.service";
import { AiController } from "./ai.controller";
import { UsersModule } from "../users/users.module";
import { SubscriptionsModule } from "../subscriptions/subscriptions.module";
import { AdminModule } from "../admin/admin.module";

@Module({
  imports: [UsersModule, SubscriptionsModule, AdminModule],
  controllers: [AiController],
  providers: [AiService],
  exports: [AiService],
})
export class AiModule {}
