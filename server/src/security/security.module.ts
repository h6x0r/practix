import { Module, Global } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { CacheModule } from '../cache/cache.module';
import { PrismaModule } from '../prisma/prisma.module';
import { CodeScannerService } from './code-scanner.service';
import { ActivityLoggerService } from './activity-logger.service';
import { IpBanService } from './ip-ban.service';
import { SecurityValidationService } from './security-validation.service';
import { IpBanGuard } from './guards/ip-ban.guard';
import { SecurityMiddleware } from './middleware/security.middleware';

@Global()
@Module({
  imports: [ConfigModule, CacheModule, PrismaModule],
  providers: [
    CodeScannerService,
    ActivityLoggerService,
    IpBanService,
    SecurityValidationService,
    IpBanGuard,
    SecurityMiddleware,
  ],
  exports: [
    CodeScannerService,
    ActivityLoggerService,
    IpBanService,
    SecurityValidationService,
    IpBanGuard,
    SecurityMiddleware,
  ],
})
export class SecurityModule {}
