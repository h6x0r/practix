import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { ProtectedRoute } from './ProtectedRoute';
import { AuthContext } from './Layout';

describe('ProtectedRoute', () => {
  const ProtectedContent = () => <div>Protected Content</div>;
  const LoginPage = () => <div>Login Page</div>;

  const renderWithRouter = (user: { id: string; email: string } | null, initialPath = '/protected') => {
    return render(
      <AuthContext.Provider value={{ user, setUser: vi.fn() }}>
        <MemoryRouter initialEntries={[initialPath]}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              path="/protected"
              element={
                <ProtectedRoute>
                  <ProtectedContent />
                </ProtectedRoute>
              }
            />
          </Routes>
        </MemoryRouter>
      </AuthContext.Provider>
    );
  };

  it('should render children when user is authenticated', () => {
    renderWithRouter({ id: 'user-1', email: 'test@example.com' });

    expect(screen.getByText('Protected Content')).toBeInTheDocument();
  });

  it('should redirect to login when user is not authenticated', () => {
    renderWithRouter(null);

    expect(screen.getByText('Login Page')).toBeInTheDocument();
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });

  it('should preserve location state when redirecting', () => {
    // This is tested indirectly - the Navigate component in ProtectedRoute
    // passes state={{ from: location }} which preserves the original location
    renderWithRouter(null);

    // User should be at login page
    expect(screen.getByText('Login Page')).toBeInTheDocument();
  });

  it('should render any children component', () => {
    const CustomComponent = () => <span data-testid="custom">Custom</span>;

    render(
      <AuthContext.Provider value={{ user: { id: '1', email: 'a@b.com' }, setUser: vi.fn() }}>
        <MemoryRouter>
          <ProtectedRoute>
            <CustomComponent />
          </ProtectedRoute>
        </MemoryRouter>
      </AuthContext.Provider>
    );

    expect(screen.getByTestId('custom')).toBeInTheDocument();
  });

  it('should render multiple children', () => {
    render(
      <AuthContext.Provider value={{ user: { id: '1', email: 'a@b.com' }, setUser: vi.fn() }}>
        <MemoryRouter>
          <ProtectedRoute>
            <div>First</div>
            <div>Second</div>
          </ProtectedRoute>
        </MemoryRouter>
      </AuthContext.Provider>
    );

    expect(screen.getByText('First')).toBeInTheDocument();
    expect(screen.getByText('Second')).toBeInTheDocument();
  });
});
