export interface UserPreferences {
  editorFontSize: number;
  editorFontFamily: string;
  editorMinimap: boolean;
  editorVimMode: boolean;
  editorLineNumbers: boolean;
  editorTheme: "vs-dark" | "light";
  notifications: {
    emailDigest: boolean;
    newCourses: boolean;
    marketing: boolean;
    securityAlerts: boolean;
  };
}

export interface User {
  id: string;
  name: string;
  email: string;
  avatarUrl: string;
  isPremium: boolean;
  role?: "USER" | "ADMIN"; // User role for access control
  plan?: {
    name: string;
    expiresAt: string;
  };
  preferences: UserPreferences;
}
