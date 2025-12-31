import { User, UserPreferences } from '@/types';
import { api } from '@/lib/api';
import { storage } from '@/lib/storage';

interface AuthResponse {
  token: string;
  user: User;
}

interface LoginPayload {
  email: string;
  password: string;
}

interface RegisterPayload {
  name: string;
  email: string;
  password: string;
}

/**
 * Auth Service - Connected to Real Backend API
 *
 * Endpoints:
 * - POST /auth/login
 * - POST /auth/register
 * - GET /users/me
 * - PATCH /users/me/preferences
 * - POST /users/upgrade
 */
export const authService = {

  /**
   * Login with email and password
   * Returns JWT token and user data
   */
  login: async (creds: LoginPayload): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/auth/login', creds);
    storage.setToken(response.token);
    return response;
  },

  /**
   * Register new user
   * Returns JWT token and user data
   */
  register: async (creds: RegisterPayload): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/auth/register', creds);
    storage.setToken(response.token);
    return response;
  },

  /**
   * Request password reset (not yet implemented on backend)
   */
  resetPassword: async (email: string): Promise<void> => {
    // TODO: Implement when backend supports password reset
    await api.post('/auth/reset-password', { email });
  },

  /**
   * Get current authenticated user profile
   */
  getMe: async (): Promise<User> => {
    return api.get<User>('/users/me');
  },

  /**
   * Update user preferences (editor settings, notifications)
   */
  updatePreferences: async (prefs: Partial<UserPreferences>): Promise<User> => {
    return api.patch<User>('/users/me/preferences', prefs);
  },

  /**
   * Logout - clear token from storage
   */
  logout: (): void => {
    storage.removeToken();
  },

  /**
   * Upgrade to premium tier
   * Note: In production, this would be called after payment webhook
   */
  upgrade: async (): Promise<User> => {
    return api.post<User>('/users/upgrade', {});
  },

  /**
   * Check if user is authenticated (has valid token)
   */
  isAuthenticated: (): boolean => {
    return !!storage.getToken();
  },

  /**
   * Update user avatar
   * Accepts either a base64-encoded image or a URL (for preset avatars)
   */
  updateAvatar: async (avatarData: string): Promise<User> => {
    return api.patch<User>('/users/me/avatar', { avatarUrl: avatarData });
  }
};
