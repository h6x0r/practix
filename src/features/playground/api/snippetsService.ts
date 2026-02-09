import { api, ApiResponse } from '@/lib/api';

export interface Snippet {
  id: string;
  shortId: string;
  title?: string;
  code: string;
  language: string;
  viewCount: number;
  createdAt: string;
}

export interface SnippetListItem {
  id: string;
  shortId: string;
  title?: string;
  language: string;
  viewCount: number;
  createdAt: string;
}

export interface CreateSnippetDto {
  title?: string;
  code: string;
  language: string;
  isPublic?: boolean;
  expiresAt?: string;
}

export interface CreateSnippetResponse {
  id: string;
  shortId: string;
  title?: string;
  language: string;
  createdAt: string;
}

export const snippetsService = {
  async create(dto: CreateSnippetDto): Promise<CreateSnippetResponse> {
    const response = await api.post<ApiResponse<CreateSnippetResponse>>(
      '/snippets',
      dto,
    );
    return response.data;
  },

  async getByShortId(shortId: string): Promise<Snippet> {
    const response = await api.get<ApiResponse<Snippet>>(
      `/snippets/${shortId}`,
    );
    return response.data;
  },

  async getMySnippets(): Promise<SnippetListItem[]> {
    const response = await api.get<ApiResponse<SnippetListItem[]>>(
      '/snippets/user/my',
    );
    return response.data;
  },

  async delete(shortId: string): Promise<{ success: boolean }> {
    const response = await api.delete<ApiResponse<{ success: boolean }>>(
      `/snippets/${shortId}`,
    );
    return response.data;
  },

  getShareUrl(shortId: string): string {
    return `${window.location.origin}/playground/${shortId}`;
  },
};
