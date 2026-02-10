import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { AuthModal } from "./AuthModal";
import { AuthContext, AuthContextType } from "./Layout";

// Create a mock for showToast
const mockShowToast = vi.fn();

// Mock dependencies
vi.mock("@/features/auth/api/authService", () => ({
  authService: {
    login: vi.fn(),
    register: vi.fn(),
  },
}));

vi.mock("@/lib/api", () => {
  // Define MockApiError inside the factory to avoid hoisting issues
  class MockApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }
  return { ApiError: MockApiError };
});

vi.mock("./Toast", () => ({
  useToast: () => ({
    showToast: mockShowToast,
  }),
}));

vi.mock("@/contexts/LanguageContext", () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        "auth.signIn": "Sign In",
        "auth.createAccount": "Create Account",
        "auth.fullName": "Full Name",
        "auth.email": "Email",
        "auth.password": "Password",
        "auth.noAccount": "Don't have an account?",
        "auth.signUp": "Sign Up",
        "auth.hasAccount": "Already have an account?",
        "auth.logIn": "Log In",
      };
      return translations[key] || key;
    },
  }),
}));

import { authService } from "@/features/auth/api/authService";
import { ApiError } from "@/lib/api";

describe("AuthModal", () => {
  const mockLogin = vi.fn();
  const mockOnClose = vi.fn();
  const mockOnSuccess = vi.fn();

  const defaultAuthContext: AuthContextType = {
    user: null,
    login: mockLogin,
    logout: vi.fn(),
    upgrade: vi.fn(),
    updateUser: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockShowToast.mockClear();
  });

  const renderModal = (
    props: Partial<React.ComponentProps<typeof AuthModal>> = {},
  ) => {
    return render(
      <AuthContext.Provider value={defaultAuthContext}>
        <AuthModal
          isOpen={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          {...props}
        />
      </AuthContext.Provider>,
    );
  };

  describe("rendering", () => {
    it("should render nothing when closed", () => {
      const { container } = render(
        <AuthContext.Provider value={defaultAuthContext}>
          <AuthModal isOpen={false} onClose={mockOnClose} />
        </AuthContext.Provider>,
      );

      expect(container.firstChild).toBeNull();
    });

    it("should render modal when open", () => {
      renderModal();

      // Sign In appears as heading and button
      expect(
        screen.getByRole("heading", { name: "Sign In" }),
      ).toBeInTheDocument();
    });

    it("should render login form by default", () => {
      renderModal();

      expect(
        screen.getByPlaceholderText("alex@example.com"),
      ).toBeInTheDocument();
      expect(screen.getByPlaceholderText("••••••••")).toBeInTheDocument();
      expect(
        screen.queryByPlaceholderText("Alex Developer"),
      ).not.toBeInTheDocument();
    });

    it("should display custom message when provided", () => {
      renderModal({ message: "Please sign in to continue" });

      expect(
        screen.getByText("Please sign in to continue"),
      ).toBeInTheDocument();
    });

    it("should render close button", () => {
      renderModal();

      const closeButtons = document.querySelectorAll("button");
      const closeButton = Array.from(closeButtons).find((btn) =>
        btn.querySelector('svg path[d*="M6 18L18 6"]'),
      );

      expect(closeButton).toBeInTheDocument();
    });
  });

  describe("mode switching", () => {
    it("should switch to register mode when clicking Sign Up", async () => {
      renderModal();

      expect(
        screen.getByRole("heading", { name: "Sign In" }),
      ).toBeInTheDocument();

      fireEvent.click(screen.getByText("Sign Up"));

      expect(
        screen.getByRole("heading", { name: "Create Account" }),
      ).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Alex Developer")).toBeInTheDocument();
    });

    it("should switch back to login mode when clicking Log In", async () => {
      renderModal();

      // Switch to register
      fireEvent.click(screen.getByText("Sign Up"));
      expect(
        screen.getByRole("heading", { name: "Create Account" }),
      ).toBeInTheDocument();

      // Switch back to login
      fireEvent.click(screen.getByText("Log In"));
      expect(
        screen.getByRole("heading", { name: "Sign In" }),
      ).toBeInTheDocument();
    });

    it("should show correct toggle text in login mode", () => {
      renderModal();

      expect(screen.getByText("Don't have an account?")).toBeInTheDocument();
      expect(screen.getByText("Sign Up")).toBeInTheDocument();
    });

    it("should show correct toggle text in register mode", () => {
      renderModal();
      fireEvent.click(screen.getByText("Sign Up"));

      expect(screen.getByText("Already have an account?")).toBeInTheDocument();
      expect(screen.getByText("Log In")).toBeInTheDocument();
    });
  });

  describe("form handling", () => {
    it("should update email input", async () => {
      renderModal();

      const emailInput = screen.getByPlaceholderText("alex@example.com");
      await userEvent.type(emailInput, "test@example.com");

      expect(emailInput).toHaveValue("test@example.com");
    });

    it("should update password input", async () => {
      renderModal();

      const passwordInput = screen.getByPlaceholderText("••••••••");
      await userEvent.type(passwordInput, "password123");

      expect(passwordInput).toHaveValue("password123");
    });

    it("should update name input in register mode", async () => {
      renderModal();
      fireEvent.click(screen.getByText("Sign Up"));

      const nameInput = screen.getByPlaceholderText("Alex Developer");
      await userEvent.type(nameInput, "John Doe");

      expect(nameInput).toHaveValue("John Doe");
    });

    it("should reset form when modal closes", async () => {
      const { rerender } = render(
        <AuthContext.Provider value={defaultAuthContext}>
          <AuthModal isOpen={true} onClose={mockOnClose} />
        </AuthContext.Provider>,
      );

      const emailInput = screen.getByPlaceholderText("alex@example.com");
      await userEvent.type(emailInput, "test@example.com");

      // Close and reopen modal
      rerender(
        <AuthContext.Provider value={defaultAuthContext}>
          <AuthModal isOpen={false} onClose={mockOnClose} />
        </AuthContext.Provider>,
      );

      rerender(
        <AuthContext.Provider value={defaultAuthContext}>
          <AuthModal isOpen={true} onClose={mockOnClose} />
        </AuthContext.Provider>,
      );

      const newEmailInput = screen.getByPlaceholderText("alex@example.com");
      expect(newEmailInput).toHaveValue("");
    });
  });

  describe("login submission", () => {
    it("should call authService.login on form submit", async () => {
      const mockUser = {
        id: "1",
        name: "Test User",
        email: "test@example.com",
      };
      vi.mocked(authService.login).mockResolvedValue({
        user: mockUser,
        token: "token123",
      });

      renderModal();

      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "test@example.com",
      );
      await userEvent.type(
        screen.getByPlaceholderText("••••••••"),
        "password123",
      );

      const submitButton = screen.getByRole("button", { name: "Sign In" });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(authService.login).toHaveBeenCalledWith({
          email: "test@example.com",
          password: "password123",
        });
      });
    });

    it("should call login context method after successful login", async () => {
      const mockUser = {
        id: "1",
        name: "Test User",
        email: "test@example.com",
      };
      vi.mocked(authService.login).mockResolvedValue({
        user: mockUser,
        token: "token123",
      });

      renderModal();

      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "test@example.com",
      );
      await userEvent.type(
        screen.getByPlaceholderText("••••••••"),
        "password123",
      );

      fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith(mockUser);
      });
    });

    it("should call onClose and onSuccess after successful login", async () => {
      const mockUser = {
        id: "1",
        name: "Test User",
        email: "test@example.com",
      };
      vi.mocked(authService.login).mockResolvedValue({
        user: mockUser,
        token: "token123",
      });

      renderModal();

      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "test@example.com",
      );
      await userEvent.type(
        screen.getByPlaceholderText("••••••••"),
        "password123",
      );

      fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });

    it("should show loading spinner during submission", async () => {
      vi.mocked(authService.login).mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100)),
      );

      renderModal();

      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "test@example.com",
      );
      await userEvent.type(
        screen.getByPlaceholderText("••••••••"),
        "password123",
      );

      fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

      // Should show spinner
      const spinner = document.querySelector(".animate-spin");
      expect(spinner).toBeInTheDocument();
    });
  });

  describe("register submission", () => {
    it("should call authService.register on register submit", async () => {
      const mockUser = { id: "1", name: "New User", email: "new@example.com" };
      vi.mocked(authService.register).mockResolvedValue({
        user: mockUser,
        token: "token123",
      });

      renderModal();
      fireEvent.click(screen.getByText("Sign Up"));

      await userEvent.type(
        screen.getByPlaceholderText("Alex Developer"),
        "New User",
      );
      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "new@example.com",
      );
      await userEvent.type(
        screen.getByPlaceholderText("••••••••"),
        "password123",
      );

      fireEvent.click(screen.getByRole("button", { name: "Create Account" }));

      await waitFor(() => {
        expect(authService.register).toHaveBeenCalledWith({
          name: "New User",
          email: "new@example.com",
          password: "password123",
        });
      });
    });

    it("should call login after successful registration", async () => {
      const mockUser = { id: "1", name: "New User", email: "new@example.com" };
      vi.mocked(authService.register).mockResolvedValue({
        user: mockUser,
        token: "token123",
      });

      renderModal();
      fireEvent.click(screen.getByText("Sign Up"));

      await userEvent.type(
        screen.getByPlaceholderText("Alex Developer"),
        "New User",
      );
      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "new@example.com",
      );
      await userEvent.type(
        screen.getByPlaceholderText("••••••••"),
        "password123",
      );

      fireEvent.click(screen.getByRole("button", { name: "Create Account" }));

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith(mockUser, true); // isNew = true for registration
      });
    });
  });

  describe("error handling", () => {
    it("should show toast on API error", async () => {
      vi.mocked(authService.login).mockRejectedValue(
        new ApiError("Invalid credentials", 401),
      );

      renderModal();

      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "test@example.com",
      );
      await userEvent.type(screen.getByPlaceholderText("••••••••"), "wrong");

      fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

      await waitFor(() => {
        expect(mockShowToast).toHaveBeenCalledWith(
          "Invalid credentials",
          "error",
        );
      });
    });

    it("should show generic error message on network error", async () => {
      vi.mocked(authService.login).mockRejectedValue(
        new Error("Network error"),
      );

      renderModal();

      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "test@example.com",
      );
      await userEvent.type(screen.getByPlaceholderText("••••••••"), "password");

      fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

      await waitFor(() => {
        expect(mockShowToast).toHaveBeenCalledWith(
          "Connection failed. Please check your internet.",
          "error",
        );
      });
    });

    it("should not close modal on error", async () => {
      vi.mocked(authService.login).mockRejectedValue(
        new ApiError("Invalid credentials", 401),
      );

      renderModal();

      await userEvent.type(
        screen.getByPlaceholderText("alex@example.com"),
        "test@example.com",
      );
      await userEvent.type(screen.getByPlaceholderText("••••••••"), "wrong");

      fireEvent.click(screen.getByRole("button", { name: "Sign In" }));

      await waitFor(() => {
        expect(mockShowToast).toHaveBeenCalled();
      });

      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe("close functionality", () => {
    it("should call onClose when clicking backdrop", () => {
      renderModal();

      const backdrop = document.querySelector(".fixed.inset-0");
      if (backdrop) {
        fireEvent.click(backdrop);
        expect(mockOnClose).toHaveBeenCalled();
      }
    });

    it("should not close when clicking modal content", () => {
      renderModal();

      const modalContent = document.querySelector(".rounded-3xl");
      if (modalContent) {
        fireEvent.click(modalContent);
        expect(mockOnClose).not.toHaveBeenCalled();
      }
    });

    it("should call onClose when clicking close button", () => {
      renderModal();

      const closeButtons = document.querySelectorAll("button");
      const closeButton = Array.from(closeButtons).find((btn) =>
        btn.querySelector('svg path[d*="M6 18L18 6"]'),
      );

      if (closeButton) {
        fireEvent.click(closeButton);
        expect(mockOnClose).toHaveBeenCalled();
      }
    });

    it("should call onClose when pressing Escape", () => {
      renderModal();

      fireEvent.keyDown(window, { key: "Escape" });

      expect(mockOnClose).toHaveBeenCalled();
    });
  });
});
