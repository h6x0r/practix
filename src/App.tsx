import React, { useState, useEffect, useMemo } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { User } from "./types";
import { authService } from "./features/auth/api/authService";
import { setupInterceptors } from "@/lib/api";
import { storage } from "./lib/storage";
import { createLogger } from "./lib/logger";

const log = createLogger("App");

// Layout & Contexts
import { Layout, ThemeContext, AuthContext } from "./components/Layout";
import { ToastProvider } from "./components/Toast";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { HiddenRoute } from "./components/HiddenRoute";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { LanguageProvider } from "./contexts/LanguageContext";
import { SubscriptionProvider } from "./contexts/SubscriptionContext";
import { GlobalKeyboardHandler } from "./components/GlobalKeyboardHandler";
import OnboardingTour from "./components/OnboardingTour";

// Route Config
import { routes } from "./config/routes";

const App = () => {
  // Load theme from localStorage, default to dark if not set
  const [isDark, setIsDark] = useState(() => {
    const savedTheme = storage.getTheme();
    return savedTheme === "dark";
  });
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isNewUser, setIsNewUser] = useState(false);

  // Theme Init & Persist
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    // Persist to localStorage
    storage.setTheme(isDark ? "dark" : "light");
  }, [isDark]);

  // Auth Init & Interceptor Setup
  useEffect(() => {
    // 1. Setup global 401 handler
    setupInterceptors(() => {
      log.warn("401 Unauthorized detected. Logging out...");
      authService.logout();
      setUser(null);
    });

    // 2. Check current session
    const initAuth = async () => {
      const token = storage.getToken();
      if (token) {
        try {
          const userData = await authService.getMe();
          setUser(userData);
        } catch (error) {
          log.error("Session expired or invalid", error);
          storage.removeToken();
          setUser(null);
        }
      }
      setLoading(false);
    };
    initAuth();
  }, []);

  // Optimize context value to prevent unnecessary re-renders
  const authValue = useMemo(
    () => ({
      user,
      login: async (u: User, isNew?: boolean) => {
        setUser(u);
        if (isNew) setIsNewUser(true);
      },
      logout: () => {
        authService.logout();
        setUser(null);
      },
      upgrade: async () => {
        try {
          const updatedUser = await authService.upgrade();
          setUser(updatedUser);
        } catch (e) {
          log.error("Upgrade failed", e);
        }
      },
      updateUser: (u: User) => {
        setUser(u);
      },
    }),
    [user],
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center text-white font-mono animate-pulse">
        Initializing Practix Engine...
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <ThemeContext.Provider
        value={{ isDark, toggleTheme: () => setIsDark(!isDark) }}
      >
        <LanguageProvider>
          <ToastProvider>
            <GlobalKeyboardHandler />
            <AuthContext.Provider value={authValue}>
              <SubscriptionProvider>
                <OnboardingTour
                  isNewUser={isNewUser}
                  onComplete={() => setIsNewUser(false)}
                />
                <BrowserRouter>
                  <Routes>
                    {routes.map((route, index) => {
                      // 1. Wrap in Layout if needed
                      let content = route.layout ? (
                        <Layout>{route.element}</Layout>
                      ) : (
                        route.element
                      );

                      // 2. Wrap in ProtectedRoute if needed (redirects to login)
                      if (route.protected) {
                        content = <ProtectedRoute>{content}</ProtectedRoute>;
                      }

                      // 3. Wrap in HiddenRoute if needed (shows 404 for unauthenticated)
                      if (route.hidden) {
                        content = <HiddenRoute>{content}</HiddenRoute>;
                      }

                      return (
                        <Route
                          key={index}
                          path={route.path}
                          element={content}
                        />
                      );
                    })}

                    {/* Fallback */}
                    <Route path="*" element={<Navigate to="/" />} />
                  </Routes>
                </BrowserRouter>
              </SubscriptionProvider>
            </AuthContext.Provider>
          </ToastProvider>
        </LanguageProvider>
      </ThemeContext.Provider>
    </ErrorBoundary>
  );
};

export default App;
