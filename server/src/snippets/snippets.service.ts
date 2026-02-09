import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateSnippetDto } from './dto/create-snippet.dto';
import { nanoid } from 'nanoid';

@Injectable()
export class SnippetsService {
  constructor(private prisma: PrismaService) {}

  async create(dto: CreateSnippetDto, userId?: string) {
    const shortId = nanoid(8); // 8-character URL-friendly ID

    const snippet = await this.prisma.codeSnippet.create({
      data: {
        shortId,
        userId,
        title: dto.title,
        code: dto.code,
        language: dto.language,
        isPublic: dto.isPublic ?? true,
        expiresAt: dto.expiresAt ? new Date(dto.expiresAt) : null,
      },
    });

    return {
      id: snippet.id,
      shortId: snippet.shortId,
      title: snippet.title,
      language: snippet.language,
      createdAt: snippet.createdAt,
    };
  }

  async findByShortId(shortId: string) {
    const snippet = await this.prisma.codeSnippet.findUnique({
      where: { shortId },
    });

    if (!snippet) {
      throw new NotFoundException('Snippet not found');
    }

    // Check expiration
    if (snippet.expiresAt && snippet.expiresAt < new Date()) {
      throw new NotFoundException('Snippet has expired');
    }

    // Increment view count
    await this.prisma.codeSnippet.update({
      where: { id: snippet.id },
      data: { viewCount: { increment: 1 } },
    });

    return {
      id: snippet.id,
      shortId: snippet.shortId,
      title: snippet.title,
      code: snippet.code,
      language: snippet.language,
      viewCount: snippet.viewCount + 1,
      createdAt: snippet.createdAt,
    };
  }

  async findUserSnippets(userId: string, limit = 20) {
    const snippets = await this.prisma.codeSnippet.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: limit,
      select: {
        id: true,
        shortId: true,
        title: true,
        language: true,
        viewCount: true,
        createdAt: true,
      },
    });

    return snippets;
  }

  async delete(shortId: string, userId: string) {
    const snippet = await this.prisma.codeSnippet.findUnique({
      where: { shortId },
    });

    if (!snippet) {
      throw new NotFoundException('Snippet not found');
    }

    if (snippet.userId !== userId) {
      throw new NotFoundException('Snippet not found');
    }

    await this.prisma.codeSnippet.delete({
      where: { id: snippet.id },
    });

    return { success: true };
  }
}
