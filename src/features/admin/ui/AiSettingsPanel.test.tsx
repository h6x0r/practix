import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AiSettingsPanel from "./AiSettingsPanel";
import { adminService } from "../api/adminService";

// Mock adminService
vi.mock("../api/adminService", () => ({
  adminService: {
    getAiSettings: vi.fn(),
    updateAiSettings: vi.fn(),
  },
}));

// Mock LanguageContext
vi.mock("@/contexts/LanguageContext", () => ({
  useUITranslation: () => ({
    tUI: (key: string) => key,
  }),
}));

// Mock logger
vi.mock("@/lib/logger", () => ({
  createLogger: () => ({
    error: vi.fn(),
    info: vi.fn(),
    debug: vi.fn(),
  }),
}));

const mockSettings = {
  enabled: true,
  limits: {
    free: 5,
    course: 30,
    premium: 100,
    promptEngineering: 100,
  },
};

describe("AiSettingsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render loading state initially", () => {
    vi.mocked(adminService.getAiSettings).mockImplementation(
      () => new Promise(() => {}),
    );

    render(<AiSettingsPanel />);
    expect(document.querySelector(".animate-pulse")).toBeInTheDocument();
  });

  it("should load and display settings", async () => {
    vi.mocked(adminService.getAiSettings).mockResolvedValue(mockSettings);

    render(<AiSettingsPanel />);

    await waitFor(() => {
      expect(screen.getByText("admin.aiSettings.title")).toBeInTheDocument();
    });

    expect(screen.getByDisplayValue("5")).toBeInTheDocument();
    expect(screen.getByDisplayValue("30")).toBeInTheDocument();
  });

  it("should show enabled status when enabled", async () => {
    vi.mocked(adminService.getAiSettings).mockResolvedValue(mockSettings);

    render(<AiSettingsPanel />);

    await waitFor(() => {
      expect(screen.getByText("admin.aiSettings.enabled")).toBeInTheDocument();
    });
  });

  it("should show disabled status when disabled", async () => {
    vi.mocked(adminService.getAiSettings).mockResolvedValue({
      ...mockSettings,
      enabled: false,
    });

    render(<AiSettingsPanel />);

    await waitFor(() => {
      expect(screen.getByText("admin.aiSettings.disabled")).toBeInTheDocument();
    });
  });

  it("should show error message on load failure", async () => {
    vi.mocked(adminService.getAiSettings).mockRejectedValue(
      new Error("Network error"),
    );

    render(<AiSettingsPanel />);

    await waitFor(() => {
      expect(
        screen.getByText("admin.aiSettings.loadError"),
      ).toBeInTheDocument();
    });
  });

  it("should display all tier labels", async () => {
    vi.mocked(adminService.getAiSettings).mockResolvedValue(mockSettings);

    render(<AiSettingsPanel />);

    await waitFor(() => {
      expect(screen.getByText("admin.aiSettings.freeTier")).toBeInTheDocument();
    });

    expect(screen.getByText("admin.aiSettings.courseTier")).toBeInTheDocument();
    expect(
      screen.getByText("admin.aiSettings.premiumTier"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("admin.aiSettings.promptEngineeringTier"),
    ).toBeInTheDocument();
  });

  it("should enable save button after changes", async () => {
    vi.mocked(adminService.getAiSettings).mockResolvedValue(mockSettings);

    render(<AiSettingsPanel />);

    await waitFor(() => {
      expect(screen.getByDisplayValue("5")).toBeInTheDocument();
    });

    // Initially save button should be disabled
    const buttons = screen.getAllByRole("button");
    const saveButton = buttons.find(
      (btn) => btn.textContent === "admin.aiSettings.saveChanges",
    );
    expect(saveButton).toBeDefined();
    expect(saveButton).toBeDisabled();

    // Change free tier limit
    const freeInput = screen.getByDisplayValue("5");
    fireEvent.change(freeInput, { target: { value: "10" } });

    // Save button should now be enabled
    expect(saveButton).not.toBeDisabled();
  });
});
