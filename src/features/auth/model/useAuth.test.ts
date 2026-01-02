import { describe, it, expect, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import React from 'react';
import { useAuth } from './useAuth';

// Mock the AuthContext
const mockUser = {
  id: 'user-1',
  email: 'test@example.com',
  name: 'Test User',
  tier: 'free',
};

const mockLogin = vi.fn();
const mockLogout = vi.fn();

// Create a wrapper with mock context
const createWrapper = (user: typeof mockUser | null = mockUser) => {
  return ({ children }: { children: React.ReactNode }) => {
    // We need to mock the Layout's AuthContext
    const AuthContext = React.createContext({
      user,
      login: mockLogin,
      logout: mockLogout,
    });

    return React.createElement(AuthContext.Provider, {
      value: { user, login: mockLogin, logout: mockLogout },
      children,
    });
  };
};

// Mock the Layout module to provide AuthContext
vi.mock('@/components/Layout', () => {
  const AuthContext = React.createContext({
    user: null as any,
    login: async () => {},
    logout: () => {},
  });

  return { AuthContext };
});

describe('useAuth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('with authenticated user', () => {
    it('should return user and isAuthenticated true', async () => {
      // We need to set up the mock context with a user
      const { AuthContext } = await import('@/components/Layout');

      const wrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(AuthContext.Provider, {
          value: {
            user: mockUser,
            login: mockLogin,
            logout: mockLogout,
          },
          children,
        });

      const { result } = renderHook(() => useAuth(), { wrapper });

      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
      expect(typeof result.current.login).toBe('function');
      expect(typeof result.current.logout).toBe('function');
    });
  });

  describe('without authenticated user', () => {
    it('should return null user and isAuthenticated false', async () => {
      const { AuthContext } = await import('@/components/Layout');

      const wrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(AuthContext.Provider, {
          value: {
            user: null,
            login: mockLogin,
            logout: mockLogout,
          },
          children,
        });

      const { result } = renderHook(() => useAuth(), { wrapper });

      expect(result.current.user).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('login function', () => {
    it('should expose login from context', async () => {
      const { AuthContext } = await import('@/components/Layout');

      const wrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(AuthContext.Provider, {
          value: {
            user: null,
            login: mockLogin,
            logout: mockLogout,
          },
          children,
        });

      const { result } = renderHook(() => useAuth(), { wrapper });

      expect(result.current.login).toBe(mockLogin);
    });
  });

  describe('logout function', () => {
    it('should expose logout from context', async () => {
      const { AuthContext } = await import('@/components/Layout');

      const wrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(AuthContext.Provider, {
          value: {
            user: mockUser,
            login: mockLogin,
            logout: mockLogout,
          },
          children,
        });

      const { result } = renderHook(() => useAuth(), { wrapper });

      expect(result.current.logout).toBe(mockLogout);
    });
  });
});
