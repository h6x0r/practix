import React, { useState, useEffect, useCallback } from "react";
import Joyride, { CallBackProps, STATUS, Step, ACTIONS, EVENTS } from "react-joyride";
import { useLanguage } from "@/contexts/LanguageContext";

interface OnboardingTourProps {
  isNewUser: boolean;
  onComplete: () => void;
}

const OnboardingTour: React.FC<OnboardingTourProps> = ({ isNewUser, onComplete }) => {
  const { language } = useLanguage();
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);

  // Check if tour was already completed
  useEffect(() => {
    const tourCompleted = localStorage.getItem("practix_onboarding_completed");
    if (isNewUser && !tourCompleted) {
      // Delay start to allow page to render
      const timer = setTimeout(() => setRun(true), 1000);
      return () => clearTimeout(timer);
    }
  }, [isNewUser]);

  const getSteps = useCallback((): Step[] => {
    const steps: Step[] = [
      {
        target: "body",
        placement: "center",
        content: language === "ru"
          ? "Добро пожаловать в Practix! Давайте познакомимся с основными функциями платформы."
          : language === "uz"
          ? "Practix-ga xush kelibsiz! Platformaning asosiy funksiyalari bilan tanishamiz."
          : "Welcome to Practix! Let's explore the main features of the platform.",
        disableBeacon: true,
        title: language === "ru" ? "👋 Добро пожаловать!" : "👋 Welcome!",
      },
      {
        target: '[data-testid="nav-courses"]',
        content: language === "ru"
          ? "Здесь находятся все курсы. 18+ курсов по Java, Go, алгоритмам и многому другому."
          : language === "uz"
          ? "Bu yerda barcha kurslar. Java, Go, algoritmlar va boshqalar bo'yicha 18+ kurs."
          : "Here are all the courses. 18+ courses on Java, Go, algorithms, and more.",
        title: language === "ru" ? "📚 Каталог курсов" : "📚 Course Catalog",
      },
      {
        target: '[data-testid="nav-playground"]',
        content: language === "ru"
          ? "Playground - ваша личная IDE в браузере. Практикуйтесь на 8 языках программирования."
          : language === "uz"
          ? "Playground - brauzeringizda shaxsiy IDE. 8 ta dasturlash tilida mashq qiling."
          : "Playground is your personal IDE in the browser. Practice in 8 programming languages.",
        title: language === "ru" ? "🎮 Playground" : "🎮 Playground",
      },
      {
        target: '[data-testid="nav-roadmap"]',
        content: language === "ru"
          ? "AI создаст персональный roadmap обучения на основе ваших целей и опыта."
          : language === "uz"
          ? "AI sizning maqsadlaringiz va tajribangizga asoslangan shaxsiy o'quv yo'l xaritasini yaratadi."
          : "AI will create a personalized learning roadmap based on your goals and experience.",
        title: language === "ru" ? "🗺️ Roadmap" : "🗺️ Roadmap",
      },
      {
        target: '[data-testid="nav-leaderboard"]',
        content: language === "ru"
          ? "Соревнуйтесь с другими программистами и поднимайтесь в рейтинге!"
          : language === "uz"
          ? "Boshqa dasturchilar bilan raqobatlashing va reytingda ko'tariling!"
          : "Compete with other developers and climb the leaderboard!",
        title: language === "ru" ? "🏆 Лидерборд" : "🏆 Leaderboard",
      },
      {
        target: '[data-testid="nav-dashboard"]',
        content: language === "ru"
          ? "Dashboard показывает вашу статистику, streak и прогресс обучения."
          : language === "uz"
          ? "Dashboard sizning statistikangiz, streak va o'quv jarayoningizni ko'rsatadi."
          : "Dashboard shows your stats, streak, and learning progress.",
        title: language === "ru" ? "📊 Dashboard" : "📊 Dashboard",
      },
      {
        target: '[data-testid="theme-toggle"]',
        content: language === "ru"
          ? "Переключайте между светлой и тёмной темой."
          : language === "uz"
          ? "Yorug' va qorong'u mavzular o'rtasida almashing."
          : "Switch between light and dark theme.",
        title: language === "ru" ? "🌙 Тема" : "🌙 Theme",
      },
      {
        target: "body",
        placement: "center",
        content: language === "ru"
          ? "Вы готовы начать! Выберите курс и приступайте к обучению. Удачи! 🚀"
          : language === "uz"
          ? "Boshlashga tayyorsiz! Kursni tanlang va o'rganishni boshlang. Omad! 🚀"
          : "You're ready to start! Choose a course and begin learning. Good luck! 🚀",
        title: language === "ru" ? "🎉 Готово!" : "🎉 All Set!",
      },
    ];

    return steps;
  }, [language]);

  const handleCallback = (data: CallBackProps) => {
    const { status, action, type, index } = data;
    const finishedStatuses: string[] = [STATUS.FINISHED, STATUS.SKIPPED];

    if (finishedStatuses.includes(status)) {
      setRun(false);
      localStorage.setItem("practix_onboarding_completed", "true");
      onComplete();
    } else if (type === EVENTS.STEP_AFTER || type === EVENTS.TARGET_NOT_FOUND) {
      // Update step index on step change
      setStepIndex(index + (action === ACTIONS.PREV ? -1 : 1));
    }
  };

  if (!run) return null;

  return (
    <Joyride
      steps={getSteps()}
      run={run}
      stepIndex={stepIndex}
      continuous
      showSkipButton
      showProgress
      callback={handleCallback}
      scrollToFirstStep
      disableOverlayClose
      spotlightClicks
      styles={{
        options: {
          primaryColor: "#6366f1",
          zIndex: 10000,
          arrowColor: "#fff",
          backgroundColor: "#fff",
          overlayColor: "rgba(0, 0, 0, 0.5)",
          textColor: "#333",
        },
        tooltip: {
          borderRadius: 12,
          padding: 20,
        },
        tooltipTitle: {
          fontSize: 18,
          fontWeight: 700,
          marginBottom: 8,
        },
        tooltipContent: {
          fontSize: 14,
          lineHeight: 1.6,
        },
        buttonNext: {
          backgroundColor: "#6366f1",
          borderRadius: 8,
          padding: "10px 20px",
          fontSize: 14,
          fontWeight: 600,
        },
        buttonBack: {
          color: "#6366f1",
          marginRight: 10,
        },
        buttonSkip: {
          color: "#9ca3af",
        },
        spotlight: {
          borderRadius: 8,
        },
      }}
      locale={{
        back: language === "ru" ? "Назад" : language === "uz" ? "Orqaga" : "Back",
        close: language === "ru" ? "Закрыть" : language === "uz" ? "Yopish" : "Close",
        last: language === "ru" ? "Начать!" : language === "uz" ? "Boshlash!" : "Let's Go!",
        next: language === "ru" ? "Далее" : language === "uz" ? "Keyingi" : "Next",
        skip: language === "ru" ? "Пропустить" : language === "uz" ? "O'tkazib yuborish" : "Skip",
      }}
    />
  );
};

export default OnboardingTour;
