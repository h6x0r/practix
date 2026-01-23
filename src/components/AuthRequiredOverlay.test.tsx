import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { renderHook, act } from '@testing-library/react';
import React from 'react';
import { AuthRequiredOverlay, useRequireAuth } from './AuthRequiredOverlay';
import { AuthContext, AuthContextType } from './Layout';

// Mock dependencies
vi.mock('@/contexts/LanguageContext', () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        'auth.loginRequired': 'Sign in Required',
        'auth.loginRequiredDesc': 'Please sign in to access this content',
        'auth.signInToContinue': 'Sign In to Continue',
        'auth.signIn': 'Sign In',
        'auth.createAccount': 'Create Account',
        'auth.email': 'Email',
        'auth.password': 'Password',
        'auth.noAccount': "Don't have an account?",
        'auth.signUp': 'Sign Up',
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock('./AuthModal', () => ({
  AuthModal: ({ isOpen, onClose, onSuccess, message }: any) => (
    isOpen ? (
      <div data-testid="auth-modal">
        <span data-testid="modal-message">{message}</span>
        <button onClick={onClose} data-testid="close-modal">Close</button>
        <button onClick={onSuccess} data-testid="login-success">Login Success</button>
      </div>
    ) : null
  ),
}));

describe('AuthRequiredOverlay', () => {
  const mockUser = { id: '1', name: 'Test User', email: 'test@example.com' };

  const defaultAuthContext: AuthContextType = {
    user: null,
    login: vi.fn(),
    logout: vi.fn(),
    upgrade: vi.fn(),
    updateUser: vi.fn(),
  };

  const authenticatedContext: AuthContextType = {
    ...defaultAuthContext,
    user: mockUser,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderOverlay = (
    authContext: AuthContextType = defaultAuthContext,
    props: Partial<React.ComponentProps<typeof AuthRequiredOverlay>> = {}
  ) => {
    return render(
      <AuthContext.Provider value={authContext}>
        <AuthRequiredOverlay {...props}>
          <div data-testid="protected-content">Protected Content</div>
        </AuthRequiredOverlay>
      </AuthContext.Provider>
    );
  };

  describe('authenticated user', () => {
    it('should render children directly when user is authenticated', () => {
      renderOverlay(authenticatedContext);

      expect(screen.getByTestId('protected-content')).toBeInTheDocument();
      expect(screen.getByText('Protected Content')).toBeInTheDocument();
    });

    it('should not show overlay when authenticated', () => {
      renderOverlay(authenticatedContext);

      expect(screen.queryByText('Sign in Required')).not.toBeInTheDocument();
      expect(screen.queryByText('Sign In to Continue')).not.toBeInTheDocument();
    });

    it('should not show blurred content when authenticated', () => {
      renderOverlay(authenticatedContext);

      const content = screen.getByTestId('protected-content');
      expect(content.closest('.blur-sm')).not.toBeInTheDocument();
    });
  });

  describe('unauthenticated user', () => {
    it('should show overlay with default title and description', () => {
      renderOverlay();

      expect(screen.getByText('Sign in Required')).toBeInTheDocument();
      expect(screen.getByText('Please sign in to access this content')).toBeInTheDocument();
    });

    it('should show sign in button', () => {
      renderOverlay();

      expect(screen.getByText('Sign In to Continue')).toBeInTheDocument();
    });

    it('should blur content when showPreview is true (default)', () => {
      renderOverlay();

      const blurredContent = document.querySelector('.blur-sm');
      expect(blurredContent).toBeInTheDocument();
    });

    it('should show placeholder when showPreview is false', () => {
      renderOverlay(defaultAuthContext, { showPreview: false });

      const placeholder = document.querySelector('.min-h-\\[400px\\]');
      expect(placeholder).toBeInTheDocument();
      expect(document.querySelector('.blur-sm')).not.toBeInTheDocument();
    });

    it('should show custom title when provided', () => {
      renderOverlay(defaultAuthContext, { title: 'Custom Title' });

      expect(screen.getByText('Custom Title')).toBeInTheDocument();
      expect(screen.queryByText('Sign in Required')).not.toBeInTheDocument();
    });

    it('should show custom description when provided', () => {
      renderOverlay(defaultAuthContext, { description: 'Custom Description' });

      expect(screen.getByText('Custom Description')).toBeInTheDocument();
      expect(screen.queryByText('Please sign in to access this content')).not.toBeInTheDocument();
    });

    it('should have aria-hidden on blurred content', () => {
      renderOverlay();

      const blurredContent = document.querySelector('[aria-hidden="true"]');
      expect(blurredContent).toBeInTheDocument();
    });

    it('should have pointer-events-none on blurred content', () => {
      renderOverlay();

      const blurredContent = document.querySelector('.pointer-events-none');
      expect(blurredContent).toBeInTheDocument();
    });
  });

  describe('auth modal interaction', () => {
    it('should open auth modal when sign in button is clicked', () => {
      renderOverlay();

      expect(screen.queryByTestId('auth-modal')).not.toBeInTheDocument();

      fireEvent.click(screen.getByText('Sign In to Continue'));

      expect(screen.getByTestId('auth-modal')).toBeInTheDocument();
    });

    it('should close auth modal when close button is clicked', () => {
      renderOverlay();

      fireEvent.click(screen.getByText('Sign In to Continue'));
      expect(screen.getByTestId('auth-modal')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('close-modal'));
      expect(screen.queryByTestId('auth-modal')).not.toBeInTheDocument();
    });

    it('should pass description to auth modal as message', () => {
      renderOverlay(defaultAuthContext, { description: 'Custom Message' });

      fireEvent.click(screen.getByText('Sign In to Continue'));

      expect(screen.getByTestId('modal-message')).toHaveTextContent('Custom Message');
    });

    it('should pass default description to auth modal when no custom description', () => {
      renderOverlay();

      fireEvent.click(screen.getByText('Sign In to Continue'));

      expect(screen.getByTestId('modal-message')).toHaveTextContent('Please sign in to access this content');
    });
  });
});

describe('useRequireAuth', () => {
  const mockUser = { id: '1', name: 'Test User', email: 'test@example.com' };

  const defaultAuthContext: AuthContextType = {
    user: null,
    login: vi.fn(),
    logout: vi.fn(),
    upgrade: vi.fn(),
    updateUser: vi.fn(),
  };

  const authenticatedContext: AuthContextType = {
    ...defaultAuthContext,
    user: mockUser,
  };

  const wrapper = (authContext: AuthContextType) => ({ children }: { children: React.ReactNode }) => (
    <AuthContext.Provider value={authContext}>
      {children}
    </AuthContext.Provider>
  );

  describe('authenticated user', () => {
    it('should execute callback immediately when authenticated', () => {
      const callback = vi.fn();
      const { result } = renderHook(() => useRequireAuth(), {
        wrapper: wrapper(authenticatedContext),
      });

      act(() => {
        result.current.requireAuth(callback);
      });

      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('should return isAuthenticated as true', () => {
      const { result } = renderHook(() => useRequireAuth(), {
        wrapper: wrapper(authenticatedContext),
      });

      expect(result.current.isAuthenticated).toBe(true);
    });
  });

  describe('unauthenticated user', () => {
    it('should not execute callback when not authenticated', () => {
      const callback = vi.fn();
      const { result } = renderHook(() => useRequireAuth(), {
        wrapper: wrapper(defaultAuthContext),
      });

      act(() => {
        result.current.requireAuth(callback);
      });

      expect(callback).not.toHaveBeenCalled();
    });

    it('should return isAuthenticated as false', () => {
      const { result } = renderHook(() => useRequireAuth(), {
        wrapper: wrapper(defaultAuthContext),
      });

      expect(result.current.isAuthenticated).toBe(false);
    });

    it('should return AuthModal component', () => {
      const { result } = renderHook(() => useRequireAuth(), {
        wrapper: wrapper(defaultAuthContext),
      });

      expect(result.current.AuthModal).toBeDefined();
    });
  });

  describe('hook with TestComponent', () => {
    const TestComponent = () => {
      const { requireAuth, AuthModal, isAuthenticated } = useRequireAuth();
      const [executed, setExecuted] = React.useState(false);

      const handleClick = () => {
        requireAuth(() => setExecuted(true));
      };

      return (
        <div>
          <span data-testid="auth-status">{isAuthenticated ? 'authenticated' : 'not-authenticated'}</span>
          <span data-testid="execution-status">{executed ? 'executed' : 'not-executed'}</span>
          <button onClick={handleClick} data-testid="action-button">Perform Action</button>
          {AuthModal}
        </div>
      );
    };

    it('should show not-authenticated status for unauthenticated user', () => {
      render(
        <AuthContext.Provider value={defaultAuthContext}>
          <TestComponent />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId('auth-status')).toHaveTextContent('not-authenticated');
    });

    it('should show authenticated status for authenticated user', () => {
      render(
        <AuthContext.Provider value={authenticatedContext}>
          <TestComponent />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId('auth-status')).toHaveTextContent('authenticated');
    });

    it('should execute action for authenticated user', () => {
      render(
        <AuthContext.Provider value={authenticatedContext}>
          <TestComponent />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId('execution-status')).toHaveTextContent('not-executed');

      fireEvent.click(screen.getByTestId('action-button'));

      expect(screen.getByTestId('execution-status')).toHaveTextContent('executed');
    });

    it('should open modal for unauthenticated user', () => {
      render(
        <AuthContext.Provider value={defaultAuthContext}>
          <TestComponent />
        </AuthContext.Provider>
      );

      fireEvent.click(screen.getByTestId('action-button'));

      expect(screen.getByTestId('auth-modal')).toBeInTheDocument();
    });

    it('should not execute action until login for unauthenticated user', () => {
      render(
        <AuthContext.Provider value={defaultAuthContext}>
          <TestComponent />
        </AuthContext.Provider>
      );

      fireEvent.click(screen.getByTestId('action-button'));

      expect(screen.getByTestId('execution-status')).toHaveTextContent('not-executed');
    });

    it('should execute pending action after successful login', () => {
      render(
        <AuthContext.Provider value={defaultAuthContext}>
          <TestComponent />
        </AuthContext.Provider>
      );

      // Click action button - should open modal
      fireEvent.click(screen.getByTestId('action-button'));
      expect(screen.getByTestId('auth-modal')).toBeInTheDocument();

      // Simulate successful login
      fireEvent.click(screen.getByTestId('login-success'));

      // Action should now be executed
      expect(screen.getByTestId('execution-status')).toHaveTextContent('executed');
    });

    it('should close modal and clear pending action on close', () => {
      render(
        <AuthContext.Provider value={defaultAuthContext}>
          <TestComponent />
        </AuthContext.Provider>
      );

      fireEvent.click(screen.getByTestId('action-button'));
      expect(screen.getByTestId('auth-modal')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('close-modal'));

      expect(screen.queryByTestId('auth-modal')).not.toBeInTheDocument();
      expect(screen.getByTestId('execution-status')).toHaveTextContent('not-executed');
    });
  });
});
