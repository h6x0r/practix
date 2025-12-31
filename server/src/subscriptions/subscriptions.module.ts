import { Module } from '@nestjs/common';
import { SubscriptionsController } from './subscriptions.controller';
import { SubscriptionsService } from './subscriptions.service';
import { AccessControlService } from './access-control.service';
import { PrismaModule } from '../prisma/prisma.module';

@Module({
  imports: [PrismaModule],
  controllers: [SubscriptionsController],
  providers: [SubscriptionsService, AccessControlService],
  exports: [SubscriptionsService, AccessControlService],
})
export class SubscriptionsModule {}
