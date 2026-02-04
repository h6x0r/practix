import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import React from "react";
import { MemoryRouter } from "react-router-dom";
import RoadmapPage from "./RoadmapPage";
import { AuthContext } from "@/components/Layout";

// Mock dependencies
vi.mock("../api/roadmapService", () => ({
  roadmapService: {
    getUserRoadmap: vi.fn(),
    generateVariants: vi.fn(),
    selectVariant: vi.fn(),
    deleteRoadmap: vi.fn(),
  },
}));

vi.mock("@/lib/storage", () => ({
  storage: {
    get: vi.fn(),
    set: vi.fn(),
    remove: vi.fn(),
  },
}));

vi.mock("@/contexts/LanguageContext", () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        "roadmap.introTitle": "AI-Powered Learning Roadmap",
        "roadmap.introDescription":
          "Answer a few questions and get a personalized learning path",
        "roadmap.featureAI": "AI-Generated",
        "roadmap.featureAIDesc": "Personalized recommendations",
        "roadmap.featurePersonal": "Personal Goals",
        "roadmap.featurePersonalDesc": "Tailored to your interests",
        "roadmap.featureProgress": "Track Progress",
        "roadmap.featureProgressDesc": "Monitor your journey",
        "roadmap.startButton": "Get Started",
        "roadmap.yourPersonalRoadmap": "Your Personal Roadmap",
        "roadmap.regenerate": "Regenerate",
        "roadmap.regeneratePremium": "Regenerate (Premium)",
        "roadmap.regeneratePremiumTitle": "Regeneration requires Premium",
        "roadmap.regeneratePremiumDesc":
          "Upgrade to create unlimited personalized roadmaps",
        "roadmap.regenerateModalTitle": "Regenerate Your Roadmap",
        "roadmap.regenerateModalDesc":
          "Create a new personalized learning path based on updated preferences",
        "roadmap.oneTimePayment": "One-time payment",
        "roadmap.regenerateFeature1": "AI-powered personalized path generation",
        "roadmap.regenerateFeature2": "Choose from multiple path variants",
        "roadmap.regenerateFeature3": "Adjust goals and time commitments",
        "roadmap.purchaseRegenerate": "Purchase Regeneration",
        "roadmap.unlimitedWith": "Want unlimited regenerations?",
        "roadmap.upgradeToPremium": "Upgrade to Premium →",
        "roadmap.interestsHint": "Select at least one area of interest",
        "roadmap.interestsError":
          "Please select at least one area of interest to continue",
        "common.cancel": "Cancel",
        "common.upgrade": "Upgrade",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("@/lib/logger", () => ({
  createLogger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

vi.mock("@/components/Toast", () => ({
  useToast: () => ({
    showToast: vi.fn(),
  }),
}));

vi.mock("./components/RoadmapVariantCard", () => ({
  RoadmapVariantCard: ({ variant, isSelected, onSelect }: any) => (
    <div
      data-testid={`variant-${variant.id}`}
      onClick={() => onSelect(variant)}
      className={isSelected ? "selected" : ""}
    >
      {variant.name}
    </div>
  ),
}));

import { roadmapService } from "../api/roadmapService";

describe("RoadmapPage", () => {
  const mockUser = {
    id: "user-1",
    email: "test@example.com",
    name: "Test User",
  };

  const mockRoadmap = {
    id: "roadmap-1",
    userId: "user-1",
    role: "backend",
    roleTitle: "Backend Developer",
    level: "intermediate",
    targetLevel: "Intermediate",
    canRegenerate: true,
    isPremium: false,
    phases: [
      {
        id: "phase-1",
        title: "Go Fundamentals",
        description: "Learn Go basics",
        colorTheme: "from-blue-500 to-cyan-500",
        progressPercentage: 50,
        steps: [
          {
            id: "step-1",
            title: "Variables",
            type: "task",
            status: "completed",
            deepLink: "/task/go-vars",
            durationEstimate: "15m",
          },
          {
            id: "step-2",
            title: "Functions",
            type: "task",
            status: "pending",
            deepLink: "/task/go-func",
            durationEstimate: "20m",
          },
        ],
      },
    ],
  };

  const mockVariants = [
    {
      id: "variant-1",
      name: "Backend Focus",
      description: "Focus on backend development",
      totalTasks: 50,
      estimatedHours: 100,
      estimatedMonths: 3,
      targetRole: "Backend Developer",
      difficulty: "intermediate" as const,
      phases: [],
    },
    {
      id: "variant-2",
      name: "Full Stack",
      description: "Complete full stack path",
      totalTasks: 80,
      estimatedHours: 160,
      estimatedMonths: 6,
      targetRole: "Full Stack Developer",
      difficulty: "advanced" as const,
      phases: [],
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);
    vi.mocked(roadmapService.generateVariants).mockResolvedValue({
      variants: mockVariants,
    });
    vi.mocked(roadmapService.selectVariant).mockResolvedValue(mockRoadmap);
  });

  const renderWithAuth = (user: typeof mockUser | null) => {
    return render(
      <AuthContext.Provider value={{ user, setUser: vi.fn() }}>
        <MemoryRouter>
          <RoadmapPage />
        </MemoryRouter>
      </AuthContext.Provider>,
    );
  };

  describe("initial state", () => {
    it("should check for existing roadmap when user is logged in", async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(roadmapService.getUserRoadmap).toHaveBeenCalled();
      });
    });

    it("should show intro screen when no roadmap exists", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(
          screen.getByText("AI-Powered Learning Roadmap"),
        ).toBeInTheDocument();
      });

      expect(screen.getByText("Get Started")).toBeInTheDocument();
    });
  });

  describe("existing roadmap", () => {
    it("should display existing roadmap", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Your Personal Roadmap")).toBeInTheDocument();
      });

      expect(screen.getByText("Go Fundamentals")).toBeInTheDocument();
    });

    it("should show progress for phases", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("50% Done")).toBeInTheDocument();
      });
    });

    it("should show regenerate button", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Regenerate")).toBeInTheDocument();
      });
    });
  });

  describe("intro screen", () => {
    it("should show features on intro screen", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("AI-Generated")).toBeInTheDocument();
      });

      expect(screen.getByText("Personal Goals")).toBeInTheDocument();
      expect(screen.getByText("Track Progress")).toBeInTheDocument();
    });
  });

  describe("unauthenticated user", () => {
    it("should show intro screen for unauthenticated users", async () => {
      renderWithAuth(null);

      await waitFor(() => {
        expect(
          screen.getByText("AI-Powered Learning Roadmap"),
        ).toBeInTheDocument();
      });

      expect(screen.getByText("Get Started")).toBeInTheDocument();
    });

    it("should not fetch roadmap for unauthenticated users", async () => {
      renderWithAuth(null);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      expect(roadmapService.getUserRoadmap).not.toHaveBeenCalled();
    });
  });

  describe("error handling", () => {
    it("should handle API error gracefully and show intro", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockRejectedValue(
        new Error("API Error"),
      );

      renderWithAuth(mockUser);

      // Should not crash and show intro screen
      await waitFor(() => {
        expect(
          screen.getByText("AI-Powered Learning Roadmap"),
        ).toBeInTheDocument();
      });
    });
  });

  describe("wizard flow", () => {
    it("should start wizard when clicking Get Started for authenticated user", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Step 1 of 5")).toBeInTheDocument();
        expect(screen.getByText("Your Background")).toBeInTheDocument();
      });
    });

    it("should show language selection on step 1", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Python")).toBeInTheDocument();
        expect(screen.getByText("JavaScript")).toBeInTheDocument();
        expect(screen.getByText("Java")).toBeInTheDocument();
        expect(screen.getByText("Go")).toBeInTheDocument();
      });
    });

    it("should allow selecting languages on step 1", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Python")).toBeInTheDocument();
      });

      // Click Python to select it
      fireEvent.click(screen.getByText("Python"));

      // Python button should now have selected styling (border-brand-500)
      const pythonButton = screen.getByText("Python").closest("button");
      expect(pythonButton).toHaveClass("border-brand-500");
    });

    it("should navigate through wizard steps with Continue button", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Step 1
      await waitFor(() => {
        expect(screen.getByText("Step 1 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue"));

      // Step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
        expect(screen.getByText("Experience Level")).toBeInTheDocument();
      });
    });

    it("should show Back button disabled on step 1", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Step 1 of 5")).toBeInTheDocument();
      });

      const backButton = screen.getByText("Back").closest("button");
      expect(backButton).toBeDisabled();
    });

    it("should navigate back with Back button", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Go to step 2
      await waitFor(() => {
        expect(screen.getByText("Step 1 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue"));

      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      // Go back to step 1
      fireEvent.click(screen.getByText("Back"));

      await waitFor(() => {
        expect(screen.getByText("Step 1 of 5")).toBeInTheDocument();
      });
    });

    it("should require interest selection on step 3", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Navigate to step 3
      fireEvent.click(screen.getByText("Continue")); // to step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue")); // to step 3
      await waitFor(() => {
        expect(screen.getByText("Step 3 of 5")).toBeInTheDocument();
        expect(screen.getByText("Your Interests")).toBeInTheDocument();
      });

      // Continue should be disabled without selecting interests
      const continueButton = screen.getByText("Continue").closest("button");
      expect(continueButton).toHaveClass("cursor-not-allowed");
    });

    it("should show hint message on interests step", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Navigate to step 3
      fireEvent.click(screen.getByText("Continue")); // to step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue")); // to step 3
      await waitFor(() => {
        expect(screen.getByText("Step 3 of 5")).toBeInTheDocument();
      });

      // Shows hint text
      expect(
        screen.getByText("Select at least one area of interest"),
      ).toBeInTheDocument();

      // Continue button should be disabled
      const continueButton = screen.getByText("Continue").closest("button");
      expect(continueButton).toBeDisabled();
    });

    it("should enable continue after selecting an interest", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Navigate to step 3
      fireEvent.click(screen.getByText("Continue")); // to step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue")); // to step 3
      await waitFor(() => {
        expect(screen.getByText("Step 3 of 5")).toBeInTheDocument();
      });

      // Continue button should be disabled initially
      let continueButton = screen.getByText("Continue").closest("button");
      expect(continueButton).toBeDisabled();

      // Select an interest
      fireEvent.click(screen.getByText("Backend Development"));

      // Continue button should now be enabled
      continueButton = screen.getByText("Continue").closest("button");
      expect(continueButton).not.toBeDisabled();

      // Still shows hint text (not error)
      expect(
        screen.getByText("Select at least one area of interest"),
      ).toBeInTheDocument();
    });

    it("should require goal selection on step 4", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Navigate to step 3 and select an interest
      fireEvent.click(screen.getByText("Continue")); // to step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue")); // to step 3
      await waitFor(() => {
        expect(screen.getByText("Step 3 of 5")).toBeInTheDocument();
      });

      // Select an interest
      fireEvent.click(screen.getByText("Backend Development"));

      fireEvent.click(screen.getByText("Continue")); // to step 4
      await waitFor(() => {
        expect(screen.getByText("Step 4 of 5")).toBeInTheDocument();
        expect(screen.getByText("Your Goal")).toBeInTheDocument();
      });

      // Continue should be disabled without selecting a goal
      const continueButton = screen.getByText("Continue").closest("button");
      expect(continueButton).toHaveClass("cursor-not-allowed");
    });

    it("should show Generate Roadmaps button on last step", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Navigate through all steps
      fireEvent.click(screen.getByText("Continue")); // step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue")); // step 3
      await waitFor(() => {
        expect(screen.getByText("Step 3 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Backend Development")); // select interest
      fireEvent.click(screen.getByText("Continue")); // step 4
      await waitFor(() => {
        expect(screen.getByText("Step 4 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Find a Job")); // select goal
      fireEvent.click(screen.getByText("Continue")); // step 5
      await waitFor(() => {
        expect(screen.getByText("Step 5 of 5")).toBeInTheDocument();
        expect(screen.getByText("Time Commitment")).toBeInTheDocument();
        expect(screen.getByText("Generate Roadmaps")).toBeInTheDocument();
      });
    });
  });

  describe("variant selection", () => {
    it("should display variants after generation", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Complete wizard quickly
      fireEvent.click(screen.getByText("Continue")); // step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue")); // step 3
      await waitFor(() => {
        expect(screen.getByText("Step 3 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Backend Development"));
      fireEvent.click(screen.getByText("Continue")); // step 4
      await waitFor(() => {
        expect(screen.getByText("Step 4 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Find a Job"));
      fireEvent.click(screen.getByText("Continue")); // step 5
      await waitFor(() => {
        expect(screen.getByText("Step 5 of 5")).toBeInTheDocument();
      });

      // Start generation
      fireEvent.click(screen.getByText("Generate Roadmaps"));

      // Wait for loading phases
      await vi.advanceTimersByTimeAsync(3500);

      // Wait for variants to appear
      await waitFor(
        () => {
          expect(screen.getByText("Choose Your Path")).toBeInTheDocument();
        },
        { timeout: 5000 },
      );

      expect(screen.getByTestId("variant-variant-1")).toBeInTheDocument();
      expect(screen.getByTestId("variant-variant-2")).toBeInTheDocument();

      vi.useRealTimers();
    });

    it("should allow selecting a variant", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(null);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Get Started")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Get Started"));

      // Complete wizard
      fireEvent.click(screen.getByText("Continue")); // step 2
      await waitFor(() => {
        expect(screen.getByText("Step 2 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Continue")); // step 3
      await waitFor(() => {
        expect(screen.getByText("Step 3 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Backend Development"));
      fireEvent.click(screen.getByText("Continue")); // step 4
      await waitFor(() => {
        expect(screen.getByText("Step 4 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Find a Job"));
      fireEvent.click(screen.getByText("Continue")); // step 5
      await waitFor(() => {
        expect(screen.getByText("Step 5 of 5")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Generate Roadmaps"));
      await vi.advanceTimersByTimeAsync(3500);

      await waitFor(
        () => {
          expect(screen.getByTestId("variant-variant-1")).toBeInTheDocument();
        },
        { timeout: 5000 },
      );

      // Select variant
      fireEvent.click(screen.getByTestId("variant-variant-1"));

      // Check it has selected class
      expect(screen.getByTestId("variant-variant-1")).toHaveClass("selected");

      vi.useRealTimers();
    });
  });

  describe("roadmap display", () => {
    it("should display phase steps with correct status", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Your Personal Roadmap")).toBeInTheDocument();
      });

      // Check step titles
      expect(screen.getByText("Variables")).toBeInTheDocument();
      expect(screen.getByText("Functions")).toBeInTheDocument();

      // Check step links
      const variablesLink = screen.getByText("Variables").closest("a");
      expect(variablesLink).toHaveAttribute("href", "/task/go-vars");

      const functionsLink = screen.getByText("Functions").closest("a");
      expect(functionsLink).toHaveAttribute("href", "/task/go-func");
    });

    it("should display role and level badges", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Backend Developer")).toBeInTheDocument();
        expect(screen.getByText("Intermediate")).toBeInTheDocument();
      });
    });

    it("should display Goal Achieved section", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Goal Achieved")).toBeInTheDocument();
        expect(
          screen.getByText(/Ready for Backend Developer/),
        ).toBeInTheDocument();
      });
    });

    it("should show step duration estimates", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("15m")).toBeInTheDocument();
        expect(screen.getByText("20m")).toBeInTheDocument();
      });
    });

    it("should show step types", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        const taskLabels = screen.getAllByText("task");
        expect(taskLabels.length).toBe(2);
      });
    });
  });

  describe("regenerate flow", () => {
    it("should start wizard when clicking Regenerate for user with canRegenerate", async () => {
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(mockRoadmap);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText("Regenerate")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Regenerate"));

      await waitFor(() => {
        expect(screen.getByText("Step 1 of 5")).toBeInTheDocument();
      });
    });

    it("should show premium gate modal when canRegenerate is false and not premium", async () => {
      const nonRegenerateRoadmap = {
        ...mockRoadmap,
        canRegenerate: false,
        isPremium: false,
      };
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(
        nonRegenerateRoadmap,
      );

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText(/Regenerate/)).toBeInTheDocument();
      });

      // Find the regenerate button that has the lock icon
      const regenerateButton = screen
        .getByText(/Regenerate \(Premium\)/)
        .closest("button");
      expect(regenerateButton).toBeInTheDocument();

      fireEvent.click(regenerateButton!);

      // Modal should appear
      await waitFor(() => {
        expect(screen.getByText(/Regenerate Your Roadmap/)).toBeInTheDocument();
        expect(screen.getByText("$4.99")).toBeInTheDocument();
      });
    });

    it("should show premium required banner when canRegenerate is false", async () => {
      const nonRegenerateRoadmap = {
        ...mockRoadmap,
        canRegenerate: false,
        isPremium: false,
      };
      vi.mocked(roadmapService.getUserRoadmap).mockResolvedValue(
        nonRegenerateRoadmap,
      );

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(
          screen.getByText(/Regeneration requires Premium/),
        ).toBeInTheDocument();
      });
    });
  });
});
