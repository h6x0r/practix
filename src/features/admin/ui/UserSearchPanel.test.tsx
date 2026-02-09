import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import UserSearchPanel from "./UserSearchPanel";
import { adminService, UserSearchResult } from "../api/adminService";

vi.mock("../api/adminService", () => ({
  adminService: {
    searchUsers: vi.fn(),
    banUser: vi.fn(),
    unbanUser: vi.fn(),
  },
}));

vi.mock("@/contexts/LanguageContext", () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        "common.showing": "Showing",
        "common.of": "of",
        "common.noData": "No data available",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("@/lib/logger", () => ({
  createLogger: () => ({
    error: vi.fn(),
    info: vi.fn(),
    debug: vi.fn(),
  }),
}));

const mockUsers: UserSearchResult[] = [
  {
    id: "1",
    email: "john@example.com",
    name: "John Doe",
    role: "USER",
    isPremium: true,
    isBanned: false,
    bannedAt: null,
    bannedReason: null,
    createdAt: "2024-01-01T00:00:00Z",
    lastActivityAt: "2024-01-15T10:00:00Z",
    submissionsCount: 50,
    coursesCount: 3,
  },
  {
    id: "2",
    email: "admin@example.com",
    name: null,
    role: "ADMIN",
    isPremium: false,
    isBanned: false,
    bannedAt: null,
    bannedReason: null,
    createdAt: "2023-06-01T00:00:00Z",
    lastActivityAt: null,
    submissionsCount: 100,
    coursesCount: 5,
  },
  {
    id: "3",
    email: "banned@example.com",
    name: "Banned User",
    role: "USER",
    isPremium: false,
    isBanned: true,
    bannedAt: "2024-01-10T00:00:00Z",
    bannedReason: "Spam",
    createdAt: "2023-12-01T00:00:00Z",
    lastActivityAt: "2024-01-09T10:00:00Z",
    submissionsCount: 10,
    coursesCount: 1,
  },
];

describe("UserSearchPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("basic rendering", () => {
    it("should render search input and button", () => {
      render(<UserSearchPanel />);

      expect(
        screen.getByPlaceholderText("admin.userSearch.placeholder"),
      ).toBeInTheDocument();
      expect(screen.getByText("admin.userSearch.search")).toBeInTheDocument();
    });

    it("should show hint text initially", () => {
      render(<UserSearchPanel />);

      expect(screen.getByText("admin.userSearch.hint")).toBeInTheDocument();
    });
  });

  describe("search functionality", () => {
    it("should show error for short query", async () => {
      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "a" } });

      const button = screen.getByText("admin.userSearch.search");
      fireEvent.click(button);

      await waitFor(() => {
        expect(
          screen.getByText("admin.userSearch.minChars"),
        ).toBeInTheDocument();
      });
    });

    it("should search users and display results", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "john" } });

      const button = screen.getByText("admin.userSearch.search");
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });

      expect(screen.getByText("john@example.com")).toBeInTheDocument();
    });

    it("should search on Enter key", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.keyDown(input, { key: "Enter" });

      await waitFor(() => {
        expect(adminService.searchUsers).toHaveBeenCalledWith("test");
      });
    });

    it("should show no results message when empty", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue([]);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "nonexistent" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(
          screen.getByText("admin.userSearch.noResults"),
        ).toBeInTheDocument();
      });
    });

    it("should show error on search failure", async () => {
      vi.mocked(adminService.searchUsers).mockRejectedValue(
        new Error("Network error"),
      );

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(
          screen.getByText("admin.userSearch.searchError"),
        ).toBeInTheDocument();
      });
    });
  });

  describe("user badges", () => {
    it("should show Premium badge for premium users", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "john" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("Premium")).toBeInTheDocument();
      });
    });

    it("should show Admin badge for admin users", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "admin" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        // Use getAllByText because "Admin" appears in both badge and filter option
        const adminTexts = screen.getAllByText("Admin");
        // At least one should be the badge (span element)
        const adminBadge = adminTexts.find(
          (el) => el.tagName === "SPAN" && el.classList.contains("rounded"),
        );
        expect(adminBadge).toBeInTheDocument();
      });
    });

    it("should show Banned badge for banned users", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "banned" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("admin.userSearch.banned")).toBeInTheDocument();
      });
    });

    it("should show 'No name' for users without name", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "admin" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("admin.userSearch.noName")).toBeInTheDocument();
      });
    });
  });

  describe("filters", () => {
    it("should show filters after search results", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByTestId("status-filter")).toBeInTheDocument();
        expect(screen.getByTestId("role-filter")).toBeInTheDocument();
      });
    });

    it("should filter by banned status", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });

      const statusFilter = screen.getByTestId("status-filter");
      fireEvent.change(statusFilter, { target: { value: "banned" } });

      await waitFor(() => {
        expect(screen.queryByText("John Doe")).not.toBeInTheDocument();
        expect(screen.getByText("Banned User")).toBeInTheDocument();
      });
    });

    it("should filter by active status", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("Banned User")).toBeInTheDocument();
      });

      const statusFilter = screen.getByTestId("status-filter");
      fireEvent.change(statusFilter, { target: { value: "active" } });

      await waitFor(() => {
        expect(screen.queryByText("Banned User")).not.toBeInTheDocument();
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });
    });

    it("should filter by premium status", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });

      const statusFilter = screen.getByTestId("status-filter");
      fireEvent.change(statusFilter, { target: { value: "premium" } });

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
        expect(screen.queryByText("Banned User")).not.toBeInTheDocument();
      });
    });

    it("should filter by role", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });

      const roleFilter = screen.getByTestId("role-filter");
      fireEvent.change(roleFilter, { target: { value: "ADMIN" } });

      await waitFor(() => {
        expect(screen.queryByText("John Doe")).not.toBeInTheDocument();
        expect(screen.getByText("admin@example.com")).toBeInTheDocument();
      });
    });

    it("should show found count", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "test" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(
          screen.getByText(/admin\.userSearch\.found.*3/),
        ).toBeInTheDocument();
      });
    });
  });

  describe("ban modal", () => {
    it("should open ban modal when clicking ban button", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue([mockUsers[0]]);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "john" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });

      const banButton = screen.getByText("admin.userSearch.ban");
      fireEvent.click(banButton);

      await waitFor(() => {
        expect(screen.getByTestId("ban-modal")).toBeInTheDocument();
      });
    });

    it("should close ban modal when clicking cancel", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue([mockUsers[0]]);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "john" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("admin.userSearch.ban"));

      await waitFor(() => {
        expect(screen.getByTestId("ban-modal")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("common.cancel"));

      await waitFor(() => {
        expect(screen.queryByTestId("ban-modal")).not.toBeInTheDocument();
      });
    });

    it("should ban user when confirming", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue([mockUsers[0]]);
      vi.mocked(adminService.banUser).mockResolvedValue(undefined);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "john" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("John Doe")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("admin.userSearch.ban"));

      await waitFor(() => {
        expect(screen.getByTestId("ban-modal")).toBeInTheDocument();
      });

      const reasonInput = screen.getByTestId("ban-reason-input");
      fireEvent.change(reasonInput, { target: { value: "Spam account" } });

      fireEvent.click(screen.getByTestId("confirm-ban-button"));

      await waitFor(() => {
        expect(adminService.banUser).toHaveBeenCalledWith("1", "Spam account");
      });
    });
  });

  describe("unban functionality", () => {
    it("should show unban button for banned users", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue([mockUsers[2]]);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "banned" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("admin.userSearch.unban")).toBeInTheDocument();
      });
    });

    it("should unban user when clicking unban", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue([mockUsers[2]]);
      vi.mocked(adminService.unbanUser).mockResolvedValue(undefined);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "banned" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("admin.userSearch.unban")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("admin.userSearch.unban"));

      await waitFor(() => {
        expect(adminService.unbanUser).toHaveBeenCalledWith("3");
      });
    });
  });

  describe("data display", () => {
    it("should display submissions and courses count", async () => {
      vi.mocked(adminService.searchUsers).mockResolvedValue(mockUsers);

      render(<UserSearchPanel />);

      const input = screen.getByPlaceholderText("admin.userSearch.placeholder");
      fireEvent.change(input, { target: { value: "john" } });
      fireEvent.click(screen.getByText("admin.userSearch.search"));

      await waitFor(() => {
        expect(screen.getByText("50")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
      });
    });
  });
});
