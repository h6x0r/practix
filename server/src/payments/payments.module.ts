import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { PrismaModule } from '../prisma/prisma.module';
import { PaymentsController } from './payments.controller';
import { PaymentsService } from './payments.service';
import { PaymeProvider } from './providers/payme.provider';
import { ClickProvider } from './providers/click.provider';
import { IpWhitelistGuard } from '../common/guards/ip-whitelist.guard';

@Module({
  imports: [PrismaModule, ConfigModule],
  controllers: [PaymentsController],
  providers: [PaymentsService, PaymeProvider, ClickProvider, IpWhitelistGuard],
  exports: [PaymentsService],
})
export class PaymentsModule {}
