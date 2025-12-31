import { useContext } from 'react';
import { AuthContext } from '@/components/Layout';

export function useAuth() {
  const context = useContext(AuthContext);

  return {
    user: context.user,
    isAuthenticated: !!context.user,
    login: context.login,
    logout: context.logout,
  };
}
