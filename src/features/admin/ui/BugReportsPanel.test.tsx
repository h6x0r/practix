import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import BugReportsPanel from "./BugReportsPanel";
import { adminService, BugReport } from "../api/adminService";

vi.mock("../api/adminService", () => ({
  adminService: {
    getBugReports: vi.fn(),
    updateBugReportStatus: vi.fn(),
  },
}));

vi.mock("@/contexts/LanguageContext", () => ({
  useUITranslation: () => ({
    tUI: (key: string) => key,
  }),
}));

vi.mock("@/lib/logger", () => ({
  createLogger: () => ({
    error: vi.fn(),
    info: vi.fn(),
    debug: vi.fn(),
  }),
}));

const mockReports: BugReport[] = [
  {
    id: "1",
    userId: "user-1",
    taskId: "task-1",
    category: "editor",
    severity: "high",
    status: "open",
    title: "Editor crashes on save",
    description: "When I click save, the editor crashes",
    metadata: { browser: "Chrome" },
    createdAt: "2024-01-15T10:00:00Z",
    updatedAt: "2024-01-15T10:00:00Z",
    user: { name: "John Doe", email: "john@example.com" },
    task: { title: "Two Sum", slug: "two-sum" },
  },
  {
    id: "2",
    userId: "user-2",
    taskId: null,
    category: "other",
    severity: "low",
    status: "resolved",
    title: "Minor UI issue",
    description: "Button is slightly misaligned",
    metadata: null,
    createdAt: "2024-01-14T09:00:00Z",
    updatedAt: "2024-01-14T12:00:00Z",
    user: { name: "", email: "jane@example.com" },
    task: null,
  },
];

describe("BugReportsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render loading state initially", () => {
    vi.mocked(adminService.getBugReports).mockImplementation(
      () => new Promise(() => {}),
    );

    render(<BugReportsPanel />);
    expect(document.querySelector(".animate-pulse")).toBeInTheDocument();
  });

  it("should load and display bug reports", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue(mockReports);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(screen.getByText("Editor crashes on save")).toBeInTheDocument();
    });

    expect(screen.getByText("Minor UI issue")).toBeInTheDocument();
  });

  it("should show report count in subtitle", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue(mockReports);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(screen.getByText(/\(2\)/)).toBeInTheDocument();
    });
  });

  it("should display severity badges", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue(mockReports);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(screen.getByText("HIGH")).toBeInTheDocument();
    });

    expect(screen.getByText("LOW")).toBeInTheDocument();
  });

  it("should show empty state when no reports", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue([]);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(
        screen.getByText("admin.bugReports.noReports"),
      ).toBeInTheDocument();
    });
  });

  it("should show error message on load failure", async () => {
    vi.mocked(adminService.getBugReports).mockRejectedValue(
      new Error("Network error"),
    );

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(
        screen.getByText("admin.bugReports.loadError"),
      ).toBeInTheDocument();
    });
  });

  it("should expand report details on click", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue(mockReports);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(screen.getByText("Editor crashes on save")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Editor crashes on save"));

    await waitFor(() => {
      expect(
        screen.getByText("When I click save, the editor crashes"),
      ).toBeInTheDocument();
    });
  });

  it("should display user name or email", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue(mockReports);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(screen.getByText(/John Doe/)).toBeInTheDocument();
    });

    // Second report has no name, should show email
    expect(screen.getByText(/jane@example.com/)).toBeInTheDocument();
  });

  it("should have status filter dropdown", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue(mockReports);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(
        screen.getByText("admin.bugReports.allStatuses"),
      ).toBeInTheDocument();
    });
  });

  it("should filter reports by status", async () => {
    vi.mocked(adminService.getBugReports).mockResolvedValue(mockReports);

    render(<BugReportsPanel />);

    await waitFor(() => {
      expect(screen.getByText("Editor crashes on save")).toBeInTheDocument();
    });

    // Find the filter dropdown (first select element)
    const selects = screen.getAllByRole("combobox");
    const filterSelect = selects[0];
    fireEvent.change(filterSelect, { target: { value: "open" } });

    await waitFor(() => {
      expect(adminService.getBugReports).toHaveBeenCalledWith({
        status: "open",
      });
    });
  });
});
