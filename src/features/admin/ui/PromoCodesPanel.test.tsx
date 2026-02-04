import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import PromoCodesPanel from "./PromoCodesPanel";
import { adminService } from "../api/adminService";

// Mock adminService
vi.mock("../api/adminService", () => ({
  adminService: {
    getPromoCodes: vi.fn(),
    getPromoCodeStats: vi.fn(),
    createPromoCode: vi.fn(),
    activatePromoCode: vi.fn(),
    deactivatePromoCode: vi.fn(),
    deletePromoCode: vi.fn(),
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
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

const mockStats = {
  total: 5,
  active: 3,
  expired: 2,
  totalUsages: 100,
  totalDiscountGiven: 500000,
};

const mockPromoCodes = {
  promoCodes: [
    {
      id: "promo1",
      code: "SUMMER20",
      type: "PERCENTAGE" as const,
      discount: 20,
      maxUses: 100,
      maxUsesPerUser: 1,
      usesCount: 25,
      minPurchaseAmount: null,
      validFrom: "2026-01-01T00:00:00Z",
      validUntil: "2026-12-31T00:00:00Z",
      isActive: true,
      applicableTo: "ALL" as const,
      courseIds: [],
      description: "Summer sale",
      createdBy: "admin1",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      _count: { usages: 25 },
    },
    {
      id: "promo2",
      code: "FIXED50K",
      type: "FIXED" as const,
      discount: 5000000,
      maxUses: null,
      maxUsesPerUser: 1,
      usesCount: 0,
      minPurchaseAmount: 10000000,
      validFrom: "2026-01-01T00:00:00Z",
      validUntil: "2026-06-30T00:00:00Z",
      isActive: false,
      applicableTo: "SUBSCRIPTIONS" as const,
      courseIds: [],
      description: null,
      createdBy: "admin1",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:00:00Z",
      _count: { usages: 0 },
    },
  ],
  total: 2,
};

describe("PromoCodesPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(adminService.getPromoCodes).mockResolvedValue(mockPromoCodes);
    vi.mocked(adminService.getPromoCodeStats).mockResolvedValue(mockStats);
  });

  it("renders panel title", async () => {
    render(<PromoCodesPanel />);
    expect(screen.getByText("admin.promocodes.title")).toBeInTheDocument();
  });

  it("renders create button", async () => {
    render(<PromoCodesPanel />);
    expect(screen.getByText("admin.promocodes.create")).toBeInTheDocument();
  });

  it("loads promo codes on mount", async () => {
    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(adminService.getPromoCodes).toHaveBeenCalled();
      expect(adminService.getPromoCodeStats).toHaveBeenCalled();
    });
  });

  it("displays stats cards", async () => {
    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(
        screen.getByText("admin.promocodes.totalCodes"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("admin.promocodes.activeCodes"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("admin.promocodes.expiredCodes"),
      ).toBeInTheDocument();
    });
  });

  it("displays promo codes list", async () => {
    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(screen.getByText("SUMMER20")).toBeInTheDocument();
      expect(screen.getByText("FIXED50K")).toBeInTheDocument();
    });
  });

  it("shows discount type labels", async () => {
    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(
        screen.getByText("admin.promocodes.typePercentage"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("admin.promocodes.typeFixed"),
      ).toBeInTheDocument();
    });
  });

  it("opens create modal when clicking create button", async () => {
    render(<PromoCodesPanel />);

    const createButton = screen.getByText("admin.promocodes.create");
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(
        screen.getByText("admin.promocodes.createTitle"),
      ).toBeInTheDocument();
    });
  });

  it("closes create modal when clicking cancel", async () => {
    render(<PromoCodesPanel />);

    fireEvent.click(screen.getByText("admin.promocodes.create"));

    await waitFor(() => {
      expect(
        screen.getByText("admin.promocodes.createTitle"),
      ).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("common.cancel"));

    await waitFor(() => {
      expect(
        screen.queryByText("admin.promocodes.createTitle"),
      ).not.toBeInTheDocument();
    });
  });

  it("handles activate/deactivate toggle", async () => {
    vi.mocked(adminService.deactivatePromoCode).mockResolvedValue({
      ...mockPromoCodes.promoCodes[0],
      isActive: false,
    });

    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(screen.getByText("SUMMER20")).toBeInTheDocument();
    });

    const deactivateButtons = screen.getAllByText(
      "admin.promocodes.deactivate",
    );
    fireEvent.click(deactivateButtons[0]);

    await waitFor(() => {
      expect(adminService.deactivatePromoCode).toHaveBeenCalledWith("promo1");
    });
  });

  it("handles activate for inactive code", async () => {
    vi.mocked(adminService.activatePromoCode).mockResolvedValue({
      ...mockPromoCodes.promoCodes[1],
      isActive: true,
    });

    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(screen.getByText("FIXED50K")).toBeInTheDocument();
    });

    const activateButtons = screen.getAllByText("admin.promocodes.activate");
    fireEvent.click(activateButtons[0]);

    await waitFor(() => {
      expect(adminService.activatePromoCode).toHaveBeenCalledWith("promo2");
    });
  });

  it("shows delete button only for unused codes", async () => {
    render(<PromoCodesPanel />);

    await waitFor(() => {
      // FIXED50K has 0 usages, so delete button should be shown
      // SUMMER20 has 25 usages, so no delete button
      const deleteButtons = screen.getAllByText("common.delete");
      expect(deleteButtons).toHaveLength(1);
    });
  });

  it("handles delete with confirmation", async () => {
    vi.mocked(adminService.deletePromoCode).mockResolvedValue({
      success: true,
    });
    window.confirm = vi.fn(() => true);

    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(screen.getByText("FIXED50K")).toBeInTheDocument();
    });

    const deleteButton = screen.getByText("common.delete");
    fireEvent.click(deleteButton);

    expect(window.confirm).toHaveBeenCalled();

    await waitFor(() => {
      expect(adminService.deletePromoCode).toHaveBeenCalledWith("promo2");
    });
  });

  it("does not delete if confirmation cancelled", async () => {
    window.confirm = vi.fn(() => false);

    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(screen.getByText("FIXED50K")).toBeInTheDocument();
    });

    const deleteButton = screen.getByText("common.delete");
    fireEvent.click(deleteButton);

    expect(window.confirm).toHaveBeenCalled();
    expect(adminService.deletePromoCode).not.toHaveBeenCalled();
  });

  it("filters by active status", async () => {
    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(adminService.getPromoCodes).toHaveBeenCalled();
    });

    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "true" } });

    await waitFor(() => {
      expect(adminService.getPromoCodes).toHaveBeenCalledWith(
        expect.objectContaining({ isActive: true }),
      );
    });
  });

  it("displays error message on load failure", async () => {
    vi.mocked(adminService.getPromoCodes).mockRejectedValue(
      new Error("Network error"),
    );

    render(<PromoCodesPanel />);

    await waitFor(() => {
      expect(
        screen.getByText("admin.promocodes.loadError"),
      ).toBeInTheDocument();
    });
  });

  it("creates promo code with form data", async () => {
    vi.mocked(adminService.createPromoCode).mockResolvedValue({
      id: "new-promo",
      code: "NEWCODE",
      type: "PERCENTAGE",
      discount: 15,
      maxUses: 50,
      maxUsesPerUser: 1,
      usesCount: 0,
      minPurchaseAmount: null,
      validFrom: "2026-02-01T00:00:00Z",
      validUntil: "2026-03-01T00:00:00Z",
      isActive: true,
      applicableTo: "ALL",
      courseIds: [],
      description: null,
      createdBy: "admin1",
      createdAt: "2026-02-01T00:00:00Z",
      updatedAt: "2026-02-01T00:00:00Z",
    });

    render(<PromoCodesPanel />);

    fireEvent.click(screen.getByText("admin.promocodes.create"));

    await waitFor(() => {
      expect(
        screen.getByText("admin.promocodes.createTitle"),
      ).toBeInTheDocument();
    });

    // Fill form
    const codeInput = screen.getByPlaceholderText("SUMMER2026");
    fireEvent.change(codeInput, { target: { value: "NEWCODE" } });

    // Find discount input (number input)
    const numberInputs = document.querySelectorAll('input[type="number"]');
    if (numberInputs[0]) {
      fireEvent.change(numberInputs[0], { target: { value: "15" } });
    }

    // Find date inputs by type
    const dateInputs = document.querySelectorAll('input[type="date"]');
    if (dateInputs[0]) {
      fireEvent.change(dateInputs[0], { target: { value: "2026-02-01" } });
    }
    if (dateInputs[1]) {
      fireEvent.change(dateInputs[1], { target: { value: "2026-03-01" } });
    }

    const createBtn = screen.getByText("admin.promocodes.createBtn");
    fireEvent.click(createBtn);

    await waitFor(() => {
      expect(adminService.createPromoCode).toHaveBeenCalled();
    });
  });
});
