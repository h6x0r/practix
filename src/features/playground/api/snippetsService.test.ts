import { describe, it, expect, vi, beforeEach } from 'vitest';
import { snippetsService } from './snippetsService';
import { api } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  api: {
    post: vi.fn(),
    get: vi.fn(),
    delete: vi.fn(),
  },
}));

describe('snippetsService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('create', () => {
    it('should create a snippet and return response', async () => {
      const mockResponse = {
        data: {
          id: 'snippet-1',
          shortId: 'abc123',
          title: 'Test Snippet',
          language: 'javascript',
          createdAt: '2024-01-01T00:00:00Z',
        },
      };
      vi.mocked(api.post).mockResolvedValue(mockResponse);

      const result = await snippetsService.create({
        code: 'console.log("hello")',
        language: 'javascript',
        title: 'Test Snippet',
      });

      expect(api.post).toHaveBeenCalledWith('/snippets', {
        code: 'console.log("hello")',
        language: 'javascript',
        title: 'Test Snippet',
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should create snippet with optional fields', async () => {
      const mockResponse = {
        data: {
          id: 'snippet-2',
          shortId: 'def456',
          language: 'python',
          createdAt: '2024-01-01T00:00:00Z',
        },
      };
      vi.mocked(api.post).mockResolvedValue(mockResponse);

      await snippetsService.create({
        code: 'print("hello")',
        language: 'python',
        isPublic: false,
        expiresAt: '2024-12-31T23:59:59Z',
      });

      expect(api.post).toHaveBeenCalledWith('/snippets', {
        code: 'print("hello")',
        language: 'python',
        isPublic: false,
        expiresAt: '2024-12-31T23:59:59Z',
      });
    });
  });

  describe('getByShortId', () => {
    it('should fetch snippet by short id', async () => {
      const mockSnippet = {
        data: {
          id: 'snippet-1',
          shortId: 'abc123',
          title: 'My Snippet',
          code: 'const x = 1;',
          language: 'javascript',
          viewCount: 42,
          createdAt: '2024-01-01T00:00:00Z',
        },
      };
      vi.mocked(api.get).mockResolvedValue(mockSnippet);

      const result = await snippetsService.getByShortId('abc123');

      expect(api.get).toHaveBeenCalledWith('/snippets/abc123');
      expect(result).toEqual(mockSnippet.data);
      expect(result.viewCount).toBe(42);
    });
  });

  describe('getMySnippets', () => {
    it('should fetch user snippets', async () => {
      const mockSnippets = {
        data: [
          {
            id: 'snippet-1',
            shortId: 'abc123',
            title: 'Snippet 1',
            language: 'javascript',
            viewCount: 10,
            createdAt: '2024-01-01T00:00:00Z',
          },
          {
            id: 'snippet-2',
            shortId: 'def456',
            language: 'python',
            viewCount: 5,
            createdAt: '2024-01-02T00:00:00Z',
          },
        ],
      };
      vi.mocked(api.get).mockResolvedValue(mockSnippets);

      const result = await snippetsService.getMySnippets();

      expect(api.get).toHaveBeenCalledWith('/snippets/user/my');
      expect(result).toHaveLength(2);
      expect(result[0].shortId).toBe('abc123');
    });

    it('should return empty array when no snippets', async () => {
      vi.mocked(api.get).mockResolvedValue({ data: [] });

      const result = await snippetsService.getMySnippets();

      expect(result).toEqual([]);
    });
  });

  describe('delete', () => {
    it('should delete snippet by short id', async () => {
      vi.mocked(api.delete).mockResolvedValue({ data: { success: true } });

      const result = await snippetsService.delete('abc123');

      expect(api.delete).toHaveBeenCalledWith('/snippets/abc123');
      expect(result.success).toBe(true);
    });
  });

  describe('getShareUrl', () => {
    it('should generate correct share URL', () => {
      // Mock window.location.origin
      Object.defineProperty(window, 'location', {
        value: { origin: 'https://practix.com' },
        writable: true,
      });

      const url = snippetsService.getShareUrl('abc123');

      expect(url).toBe('https://practix.com/playground/abc123');
    });

    it('should handle localhost', () => {
      Object.defineProperty(window, 'location', {
        value: { origin: 'http://localhost:3000' },
        writable: true,
      });

      const url = snippetsService.getShareUrl('xyz789');

      expect(url).toBe('http://localhost:3000/playground/xyz789');
    });
  });
});
