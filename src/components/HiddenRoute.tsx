
import React, { useContext } from 'react';
import { AuthContext } from './Layout';
import { NotFoundPage } from './NotFoundPage';

interface HiddenRouteProps {
  children: React.ReactNode;
}

/**
 * HiddenRoute - Shows 404 page for unauthenticated users
 * Use this for routes that should appear to not exist for logged out users
 * (e.g., Settings, Payments) rather than redirecting to login
 */
export const HiddenRoute = ({ children }: HiddenRouteProps) => {
  const { user } = useContext(AuthContext);

  if (!user) {
    return <NotFoundPage />;
  }

  return <>{children}</>;
};
