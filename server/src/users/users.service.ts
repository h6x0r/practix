
import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class UsersService {
  constructor(private prisma: PrismaService) {}

  async findOne(email: string) {
    return (this.prisma as any).user.findUnique({
      where: { email },
    });
  }

  async findById(id: string) {
    return (this.prisma as any).user.findUnique({
      where: { id },
    });
  }

  async create(data: any) {
    return (this.prisma as any).user.create({
      data,
    });
  }

  async updatePreferences(userId: string, preferences: any) {
    return (this.prisma as any).user.update({
      where: { id: userId },
      data: { preferences },
    });
  }

  async updatePlan(userId: string, isPremium: boolean, plan: any) {
    return (this.prisma as any).user.update({
      where: { id: userId },
      data: { 
        isPremium,
        plan 
      },
    });
  }
}
