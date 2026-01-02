import { describe, it, expect, beforeEach, vi } from 'vitest';
import { authService } from './authService';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
  },
}));

// Mock storage
vi.mock('@/lib/storage', () => ({
  storage: {
    getToken: vi.fn(),
    setToken: vi.fn(),
    removeToken: vi.fn(),
  },
}));

import { api } from '@/lib/api';
import { storage } from '@/lib/storage';

describe('authService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('login', () => {
    it('should login and store token', async () => {
      const mockResponse = {
        token: 'jwt-token-123',
        user: { id: 'user-1', email: 'test@example.com', name: 'Test User' },
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResponse);

      const result = await authService.login({
        email: 'test@example.com',
        password: 'password123',
      });

      expect(api.post).toHaveBeenCalledWith('/auth/login', {
        email: 'test@example.com',
        password: 'password123',
      });
      expect(storage.setToken).toHaveBeenCalledWith('jwt-token-123');
      expect(result).toEqual(mockResponse);
    });

    it('should throw on invalid credentials', async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error('Invalid credentials'));

      await expect(
        authService.login({
          email: 'wrong@example.com',
          password: 'wrongpass',
        })
      ).rejects.toThrow('Invalid credentials');

      expect(storage.setToken).not.toHaveBeenCalled();
    });
  });

  describe('register', () => {
    it('should register and store token', async () => {
      const mockResponse = {
        token: 'new-jwt-token',
        user: { id: 'user-2', email: 'new@example.com', name: 'New User' },
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResponse);

      const result = await authService.register({
        name: 'New User',
        email: 'new@example.com',
        password: 'newpassword',
      });

      expect(api.post).toHaveBeenCalledWith('/auth/register', {
        name: 'New User',
        email: 'new@example.com',
        password: 'newpassword',
      });
      expect(storage.setToken).toHaveBeenCalledWith('new-jwt-token');
      expect(result).toEqual(mockResponse);
    });

    it('should throw on duplicate email', async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error('Email already exists'));

      await expect(
        authService.register({
          name: 'Test',
          email: 'existing@example.com',
          password: 'password',
        })
      ).rejects.toThrow('Email already exists');
    });
  });

  describe('resetPassword', () => {
    it('should request password reset', async () => {
      vi.mocked(api.post).mockResolvedValueOnce({});

      await authService.resetPassword('user@example.com');

      expect(api.post).toHaveBeenCalledWith('/auth/reset-password', {
        email: 'user@example.com',
      });
    });
  });

  describe('getMe', () => {
    it('should fetch current user', async () => {
      const mockUser = {
        id: 'user-1',
        email: 'test@example.com',
        name: 'Test User',
        tier: 'free',
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockUser);

      const result = await authService.getMe();

      expect(api.get).toHaveBeenCalledWith('/users/me');
      expect(result).toEqual(mockUser);
    });
  });

  describe('updatePreferences', () => {
    it('should update user preferences', async () => {
      const mockUpdatedUser = {
        id: 'user-1',
        preferences: { theme: 'dark', fontSize: 16 },
      };

      vi.mocked(api.patch).mockResolvedValueOnce(mockUpdatedUser);

      const result = await authService.updatePreferences({ theme: 'dark' });

      expect(api.patch).toHaveBeenCalledWith('/users/me/preferences', { theme: 'dark' });
      expect(result).toEqual(mockUpdatedUser);
    });
  });

  describe('logout', () => {
    it('should remove token from storage', () => {
      authService.logout();

      expect(storage.removeToken).toHaveBeenCalled();
    });
  });

  describe('upgrade', () => {
    it('should upgrade user to premium', async () => {
      const mockUpgradedUser = {
        id: 'user-1',
        tier: 'premium',
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockUpgradedUser);

      const result = await authService.upgrade();

      expect(api.post).toHaveBeenCalledWith('/users/upgrade', {});
      expect(result).toEqual(mockUpgradedUser);
    });
  });

  describe('isAuthenticated', () => {
    it('should return true when token exists', () => {
      vi.mocked(storage.getToken).mockReturnValueOnce('some-token');

      expect(authService.isAuthenticated()).toBe(true);
    });

    it('should return false when no token', () => {
      vi.mocked(storage.getToken).mockReturnValueOnce(null);

      expect(authService.isAuthenticated()).toBe(false);
    });

    it('should return false for empty token', () => {
      vi.mocked(storage.getToken).mockReturnValueOnce('');

      expect(authService.isAuthenticated()).toBe(false);
    });
  });

  describe('updateAvatar', () => {
    it('should update user avatar', async () => {
      const mockUser = {
        id: 'user-1',
        avatarUrl: 'https://example.com/avatar.png',
      };

      vi.mocked(api.patch).mockResolvedValueOnce(mockUser);

      const result = await authService.updateAvatar('https://example.com/avatar.png');

      expect(api.patch).toHaveBeenCalledWith('/users/me/avatar', {
        avatarUrl: 'https://example.com/avatar.png',
      });
      expect(result).toEqual(mockUser);
    });

    it('should accept base64 encoded image', async () => {
      const base64Image = 'data:image/png;base64,iVBORw0KGgo...';
      vi.mocked(api.patch).mockResolvedValueOnce({});

      await authService.updateAvatar(base64Image);

      expect(api.patch).toHaveBeenCalledWith('/users/me/avatar', {
        avatarUrl: base64Image,
      });
    });
  });
});
