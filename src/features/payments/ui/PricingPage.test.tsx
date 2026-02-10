import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { BrowserRouter } from "react-router-dom";
import PricingPage from "./PricingPage";
import { AuthContext } from "@/components/Layout";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { subscriptionService } from "@/features/subscriptions/api/subscriptionService";

vi.mock("@/features/subscriptions/api/subscriptionService");

const mockPlans = [
  {
    id: "plan-global",
    slug: "global-premium",
    name: "Global Premium",
    nameRu: "Глобальный Premium",
    type: "global",
    priceMonthly: 9900000,
    priceYearly: 9900000 * 10,
    features: ["All courses", "100 AI requests/day"],
    durationDays: 30,
  },
  {
    id: "plan-course-1",
    slug: "java-course",
    name: "Java Course",
    nameRu: "Курс Java",
    type: "course",
    courseId: "course-java",
    priceMonthly: 4900000,
    priceYearly: 4900000 * 10,
    features: ["Java course access", "30 AI requests/day"],
    durationDays: 30,
    course: { icon: "☕" },
  },
];

const mockNavigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual("react-router-dom");
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const renderPage = (user: { id: string; name: string } | null = null) => {
  return render(
    <BrowserRouter>
      <LanguageProvider>
        <AuthContext.Provider
          value={{
            user: user as any,
            login: vi.fn(),
            logout: vi.fn(),
            upgrade: vi.fn(),
            updateUser: vi.fn(),
          }}
        >
          <PricingPage />
        </AuthContext.Provider>
      </LanguageProvider>
    </BrowserRouter>,
  );
};

describe("PricingPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(subscriptionService.getPlans).mockResolvedValue(mockPlans as any);
  });

  describe("rendering", () => {
    it("should render loading state initially", () => {
      vi.mocked(subscriptionService.getPlans).mockImplementation(
        () => new Promise(() => {}), // Never resolves
      );
      renderPage();
      expect(document.querySelector(".animate-spin")).toBeInTheDocument();
    });

    it("should render pricing title after loading", async () => {
      renderPage();
      await waitFor(() => {
        expect(screen.getByText(/Choose Your Plan/i)).toBeInTheDocument();
      });
    });

    it("should render all three plan cards", async () => {
      renderPage();
      await waitFor(() => {
        expect(screen.getByTestId("select-free")).toBeInTheDocument();
        expect(screen.getByTestId("select-course")).toBeInTheDocument();
        expect(screen.getByTestId("select-premium")).toBeInTheDocument();
      });
    });

    it("should display feature comparison table", async () => {
      renderPage();
      await waitFor(() => {
        expect(screen.getByText(/Compare Features/i)).toBeInTheDocument();
      });
    });

    it("should render FAQ section", async () => {
      renderPage();
      await waitFor(() => {
        expect(
          screen.getByText(/Frequently Asked Questions/i),
        ).toBeInTheDocument();
      });
    });
  });

  describe("billing toggle", () => {
    it("should default to monthly billing", async () => {
      renderPage();
      await waitFor(() => {
        expect(screen.getByText("Monthly")).toBeInTheDocument();
        expect(screen.getByText("Yearly")).toBeInTheDocument();
      });
    });

    it("should have billing toggle buttons", async () => {
      renderPage();
      await waitFor(() => {
        expect(screen.getByTestId("select-free")).toBeInTheDocument();
      });
      // Billing toggle is rendered
      expect(screen.getByText("Monthly")).toBeInTheDocument();
      expect(screen.getByText("Yearly")).toBeInTheDocument();
    });
  });

  describe("plan selection", () => {
    it("should redirect to login when not authenticated and selecting plan", async () => {
      renderPage(null);
      await waitFor(() => {
        expect(screen.getByTestId("select-premium")).toBeInTheDocument();
      });

      const premiumButton = screen.getByTestId("select-premium");
      fireEvent.click(premiumButton);

      expect(mockNavigate).toHaveBeenCalledWith(
        expect.stringContaining("/login"),
      );
    });

    it("should redirect to payments when authenticated", async () => {
      renderPage({ id: "user-1", name: "Test User" });
      await waitFor(() => {
        expect(screen.getByTestId("select-premium")).toBeInTheDocument();
      });

      const premiumButton = screen.getByTestId("select-premium");
      fireEvent.click(premiumButton);

      expect(mockNavigate).toHaveBeenCalledWith(
        expect.stringContaining("/payments"),
      );
    });

    it("should navigate to payments without plan for free tier", async () => {
      renderPage({ id: "user-1", name: "Test User" });
      await waitFor(() => {
        expect(screen.getByTestId("select-free")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByTestId("select-free"));
      expect(mockNavigate).toHaveBeenCalled();
    });
  });

  describe("FAQ section", () => {
    it("should render all FAQ questions", async () => {
      renderPage();
      await waitFor(() => {
        expect(screen.getByText(/Can I cancel anytime/i)).toBeInTheDocument();
        expect(screen.getByText(/refund policy/i)).toBeInTheDocument();
        expect(screen.getByText(/upgrade my plan/i)).toBeInTheDocument();
        expect(screen.getByText(/payment methods/i)).toBeInTheDocument();
      });
    });
  });
});
