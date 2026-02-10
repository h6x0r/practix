import { render, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import OnboardingTour from "./OnboardingTour";
import { LanguageProvider } from "@/contexts/LanguageContext";

// Mock react-joyride
vi.mock("react-joyride", () => ({
  default: vi.fn(({ run, steps }) => {
    if (run && steps.length > 0) {
      return <div data-testid="joyride-mock">Joyride Running</div>;
    }
    return null;
  }),
  STATUS: { FINISHED: "finished", SKIPPED: "skipped" },
  ACTIONS: { PREV: "prev", NEXT: "next" },
  EVENTS: { STEP_AFTER: "step_after", TARGET_NOT_FOUND: "target_not_found" },
}));

const renderTour = (isNewUser: boolean, onComplete = vi.fn()) => {
  return render(
    <LanguageProvider>
      <OnboardingTour isNewUser={isNewUser} onComplete={onComplete} />
    </LanguageProvider>,
  );
};

describe("OnboardingTour", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    localStorage.clear();
  });

  describe("rendering", () => {
    it("should not render when user is not new", async () => {
      const { container } = renderTour(false);
      await act(async () => {
        vi.advanceTimersByTime(2000);
      });
      expect(
        container.querySelector('[data-testid="joyride-mock"]'),
      ).toBeNull();
    });

    it("should render joyride for new users after delay", async () => {
      const { queryByTestId } = renderTour(true);

      // Initially not rendered
      expect(queryByTestId("joyride-mock")).toBeNull();

      // After delay, should render
      await act(async () => {
        vi.advanceTimersByTime(1500);
      });
      expect(queryByTestId("joyride-mock")).toBeInTheDocument();
    });
  });

  describe("props", () => {
    it("should accept isNewUser prop", () => {
      const { container } = renderTour(false);
      expect(container).toBeDefined();
    });

    it("should accept onComplete callback", () => {
      const onComplete = vi.fn();
      const { container } = renderTour(true, onComplete);
      expect(container).toBeDefined();
    });
  });

  describe("tour behavior", () => {
    it("should start tour for new users after 1 second delay", async () => {
      const { queryByTestId } = renderTour(true);

      // Before delay - no tour
      expect(queryByTestId("joyride-mock")).toBeNull();

      // After delay - tour starts
      await act(async () => {
        vi.advanceTimersByTime(1100);
      });
      expect(queryByTestId("joyride-mock")).toBeInTheDocument();
    });

    it("should not start tour for existing users", async () => {
      const { queryByTestId } = renderTour(false);

      await act(async () => {
        vi.advanceTimersByTime(2000);
      });

      expect(queryByTestId("joyride-mock")).toBeNull();
    });
  });
});
