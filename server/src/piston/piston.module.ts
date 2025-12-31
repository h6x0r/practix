import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { PistonService } from './piston.service';

@Module({
  imports: [ConfigModule],
  providers: [PistonService],
  exports: [PistonService],
})
export class PistonModule {}
