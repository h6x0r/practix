import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'interview-common-questions',
	title: 'Security Interview Questions',
	difficulty: 'medium',
	tags: ['security', 'interview', 'fundamentals', 'typescript'],
	estimatedTime: '30m',
	isPremium: true,
	youtubeUrl: '',
	description: `Master common security interview questions through interactive coding exercises.

**Key Interview Topics:**

1. **CIA Triad** - Confidentiality, Integrity, Availability
2. **Authentication vs Authorization** - Identity verification vs access control
3. **Encryption Types** - Symmetric vs Asymmetric
4. **Common Vulnerabilities** - OWASP Top 10 understanding
5. **Security Protocols** - TLS, OAuth, JWT

**Your Task:**

Implement a \`SecurityInterviewBot\` class that can quiz you on security concepts and evaluate your answers.`,
	initialCode: `interface Question {
  id: string;
  category: 'fundamentals' | 'cryptography' | 'web_security' | 'network' | 'compliance';
  question: string;
  keyPoints: string[];
  difficulty: 'easy' | 'medium' | 'hard';
  followUpQuestions?: string[];
}

interface Answer {
  questionId: string;
  response: string;
  timestamp: Date;
}

interface Evaluation {
  score: number; // 0-100
  matchedKeyPoints: string[];
  missedKeyPoints: string[];
  feedback: string;
  suggestedImprovement: string;
}

interface InterviewSession {
  id: string;
  startTime: Date;
  questions: Question[];
  answers: Answer[];
  evaluations: Evaluation[];
  overallScore: number;
  strengths: string[];
  weaknesses: string[];
}

class SecurityInterviewBot {
  private questionBank: Question[] = [];
  private sessions: Map<string, InterviewSession> = new Map();

  // Question Management
  addQuestion(question: Question): void {
    // TODO: Add a question to the bank
  }

  getQuestionsByCategory(category: Question['category']): Question[] {
    // TODO: Get questions by category
    return [];
  }

  getRandomQuestion(categories?: Question['category'][]): Question | null {
    // TODO: Get random question, optionally filtered by categories
    return null;
  }

  // Interview Session
  startSession(): string {
    // TODO: Start a new interview session
    return '';
  }

  askQuestion(sessionId: string, questionId?: string): Question | null {
    // TODO: Ask a question in the session
    return null;
  }

  submitAnswer(sessionId: string, questionId: string, response: string): Evaluation {
    // TODO: Submit and evaluate an answer
    return {} as Evaluation;
  }

  // Answer Evaluation
  evaluateAnswer(question: Question, response: string): Evaluation {
    // TODO: Evaluate how well the answer covers key points
    return {} as Evaluation;
  }

  calculateKeyPointCoverage(keyPoints: string[], response: string): {
    matched: string[];
    missed: string[];
    coverage: number;
  } {
    // TODO: Check which key points are covered in the response
    return { matched: [], missed: [], coverage: 0 };
  }

  // Session Analysis
  getSessionSummary(sessionId: string): {
    totalQuestions: number;
    averageScore: number;
    byCategory: Record<string, { score: number; count: number }>;
    recommendations: string[];
  } {
    // TODO: Generate session summary
    return {} as any;
  }

  identifyWeakAreas(sessionId: string): string[] {
    // TODO: Identify weak areas based on evaluations
    return [];
  }

  // Study Recommendations
  generateStudyPlan(sessionId: string): {
    topic: string;
    priority: 'high' | 'medium' | 'low';
    resources: string[];
  }[] {
    // TODO: Generate personalized study plan
    return [];
  }
}

// Test your implementation
const bot = new SecurityInterviewBot();

// Test 1: Add question
bot.addQuestion({
  id: 'q1',
  category: 'fundamentals',
  question: 'What is the CIA triad?',
  keyPoints: ['confidentiality', 'integrity', 'availability', 'security principles'],
  difficulty: 'easy',
});
console.log('Test 1 - Question added:', bot.getQuestionsByCategory('fundamentals').length === 1);

// Test 2: Get random question
bot.addQuestion({
  id: 'q2',
  category: 'cryptography',
  question: 'Explain symmetric vs asymmetric encryption',
  keyPoints: ['same key', 'public/private key pair', 'performance', 'key exchange'],
  difficulty: 'medium',
});
const randomQ = bot.getRandomQuestion();
console.log('Test 2 - Random question:', randomQ !== null);

// Test 3: Start session
const sessionId = bot.startSession();
console.log('Test 3 - Session started:', sessionId.length > 0);

// Test 4: Ask question
const question = bot.askQuestion(sessionId);
console.log('Test 4 - Question asked:', question !== null);

// Test 5: Evaluate answer
const eval1 = bot.evaluateAnswer(
  { id: 'q1', category: 'fundamentals', question: 'CIA triad?', keyPoints: ['confidentiality', 'integrity', 'availability'], difficulty: 'easy' },
  'The CIA triad consists of Confidentiality, Integrity, and Availability - the core principles of information security.'
);
console.log('Test 5 - Answer evaluated:', eval1.score > 50);

// Test 6: Submit answer in session
const eval2 = bot.submitAnswer(sessionId, question?.id || 'q1', 'Security answer here');
console.log('Test 6 - Answer submitted:', 'score' in eval2);

// Test 7: Key point coverage
const coverage = bot.calculateKeyPointCoverage(
  ['encryption', 'decryption', 'key management'],
  'Encryption transforms data using keys, decryption reverses it.'
);
console.log('Test 7 - Coverage calculated:', coverage.matched.length >= 2);

// Test 8: Session summary
const summary = bot.getSessionSummary(sessionId);
console.log('Test 8 - Summary generated:', 'totalQuestions' in summary);

// Test 9: Weak areas
const weakAreas = bot.identifyWeakAreas(sessionId);
console.log('Test 9 - Weak areas:', Array.isArray(weakAreas));

// Test 10: Study plan
const plan = bot.generateStudyPlan(sessionId);
console.log('Test 10 - Study plan:', Array.isArray(plan));`,
	solutionCode: `interface Question {
  id: string;
  category: 'fundamentals' | 'cryptography' | 'web_security' | 'network' | 'compliance';
  question: string;
  keyPoints: string[];
  difficulty: 'easy' | 'medium' | 'hard';
  followUpQuestions?: string[];
}

interface Answer {
  questionId: string;
  response: string;
  timestamp: Date;
}

interface Evaluation {
  score: number;
  matchedKeyPoints: string[];
  missedKeyPoints: string[];
  feedback: string;
  suggestedImprovement: string;
}

interface InterviewSession {
  id: string;
  startTime: Date;
  questions: Question[];
  answers: Answer[];
  evaluations: Evaluation[];
  overallScore: number;
  strengths: string[];
  weaknesses: string[];
}

class SecurityInterviewBot {
  private questionBank: Question[] = [];
  private sessions: Map<string, InterviewSession> = new Map();

  addQuestion(question: Question): void {
    // Avoid duplicates
    if (!this.questionBank.find(q => q.id === question.id)) {
      this.questionBank.push(question);
    }
  }

  getQuestionsByCategory(category: Question['category']): Question[] {
    return this.questionBank.filter(q => q.category === category);
  }

  getRandomQuestion(categories?: Question['category'][]): Question | null {
    let pool = this.questionBank;

    if (categories && categories.length > 0) {
      pool = this.questionBank.filter(q => categories.includes(q.category));
    }

    if (pool.length === 0) return null;

    const randomIndex = Math.floor(Math.random() * pool.length);
    return pool[randomIndex];
  }

  startSession(): string {
    const sessionId = \`session-\${Date.now()}-\${Math.random().toString(36).substring(2, 8)}\`;

    const session: InterviewSession = {
      id: sessionId,
      startTime: new Date(),
      questions: [],
      answers: [],
      evaluations: [],
      overallScore: 0,
      strengths: [],
      weaknesses: [],
    };

    this.sessions.set(sessionId, session);
    return sessionId;
  }

  askQuestion(sessionId: string, questionId?: string): Question | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    let question: Question | null = null;

    if (questionId) {
      question = this.questionBank.find(q => q.id === questionId) || null;
    } else {
      // Get a random question not already asked in this session
      const askedIds = new Set(session.questions.map(q => q.id));
      const available = this.questionBank.filter(q => !askedIds.has(q.id));

      if (available.length > 0) {
        const randomIndex = Math.floor(Math.random() * available.length);
        question = available[randomIndex];
      }
    }

    if (question) {
      session.questions.push(question);
    }

    return question;
  }

  submitAnswer(sessionId: string, questionId: string, response: string): Evaluation {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return {
        score: 0,
        matchedKeyPoints: [],
        missedKeyPoints: [],
        feedback: 'Session not found',
        suggestedImprovement: '',
      };
    }

    const question = session.questions.find(q => q.id === questionId);
    if (!question) {
      return {
        score: 0,
        matchedKeyPoints: [],
        missedKeyPoints: [],
        feedback: 'Question not found in session',
        suggestedImprovement: '',
      };
    }

    const answer: Answer = {
      questionId,
      response,
      timestamp: new Date(),
    };
    session.answers.push(answer);

    const evaluation = this.evaluateAnswer(question, response);
    session.evaluations.push(evaluation);

    // Update session stats
    this.updateSessionStats(session);

    return evaluation;
  }

  evaluateAnswer(question: Question, response: string): Evaluation {
    const coverage = this.calculateKeyPointCoverage(question.keyPoints, response);

    // Calculate base score from coverage
    let score = coverage.coverage * 100;

    // Adjust for difficulty
    const difficultyMultiplier = {
      easy: 0.9,
      medium: 1.0,
      hard: 1.1,
    };
    score = Math.min(100, score * difficultyMultiplier[question.difficulty]);

    // Bonus for thoroughness (longer, detailed responses)
    const wordCount = response.split(/\s+/).length;
    if (wordCount > 50) score = Math.min(100, score + 5);
    if (wordCount > 100) score = Math.min(100, score + 5);

    // Generate feedback
    let feedback: string;
    if (score >= 80) {
      feedback = 'Excellent answer! You covered most key points comprehensively.';
    } else if (score >= 60) {
      feedback = 'Good answer, but some important concepts were missing.';
    } else if (score >= 40) {
      feedback = 'Partial answer. You should study this topic more thoroughly.';
    } else {
      feedback = 'The answer needs significant improvement. Review the fundamentals.';
    }

    // Generate improvement suggestion
    const suggestedImprovement = coverage.missed.length > 0
      ? \`Focus on explaining: \${coverage.missed.slice(0, 3).join(', ')}\`
      : 'Consider adding real-world examples to strengthen your answer.';

    return {
      score: Math.round(score),
      matchedKeyPoints: coverage.matched,
      missedKeyPoints: coverage.missed,
      feedback,
      suggestedImprovement,
    };
  }

  calculateKeyPointCoverage(keyPoints: string[], response: string): {
    matched: string[];
    missed: string[];
    coverage: number;
  } {
    const responseLower = response.toLowerCase();
    const matched: string[] = [];
    const missed: string[] = [];

    for (const keyPoint of keyPoints) {
      const keyPointLower = keyPoint.toLowerCase();

      // Check for exact or partial match
      const words = keyPointLower.split(/\s+/);
      const hasMatch = words.some(word =>
        word.length > 3 && responseLower.includes(word)
      ) || responseLower.includes(keyPointLower);

      if (hasMatch) {
        matched.push(keyPoint);
      } else {
        missed.push(keyPoint);
      }
    }

    const coverage = keyPoints.length > 0 ? matched.length / keyPoints.length : 0;

    return { matched, missed, coverage };
  }

  private updateSessionStats(session: InterviewSession): void {
    if (session.evaluations.length === 0) {
      session.overallScore = 0;
      return;
    }

    // Calculate overall score
    const totalScore = session.evaluations.reduce((sum, e) => sum + e.score, 0);
    session.overallScore = Math.round(totalScore / session.evaluations.length);

    // Identify strengths and weaknesses by category
    const categoryScores: Record<string, { total: number; count: number }> = {};

    for (let i = 0; i < session.questions.length; i++) {
      const q = session.questions[i];
      const e = session.evaluations[i];
      if (!e) continue;

      if (!categoryScores[q.category]) {
        categoryScores[q.category] = { total: 0, count: 0 };
      }
      categoryScores[q.category].total += e.score;
      categoryScores[q.category].count++;
    }

    session.strengths = [];
    session.weaknesses = [];

    for (const [category, data] of Object.entries(categoryScores)) {
      const avgScore = data.total / data.count;
      if (avgScore >= 70) {
        session.strengths.push(category);
      } else if (avgScore < 50) {
        session.weaknesses.push(category);
      }
    }
  }

  getSessionSummary(sessionId: string): {
    totalQuestions: number;
    averageScore: number;
    byCategory: Record<string, { score: number; count: number }>;
    recommendations: string[];
  } {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return {
        totalQuestions: 0,
        averageScore: 0,
        byCategory: {},
        recommendations: [],
      };
    }

    const byCategory: Record<string, { score: number; count: number }> = {};

    for (let i = 0; i < session.questions.length; i++) {
      const q = session.questions[i];
      const e = session.evaluations[i];
      if (!e) continue;

      if (!byCategory[q.category]) {
        byCategory[q.category] = { score: 0, count: 0 };
      }
      byCategory[q.category].score += e.score;
      byCategory[q.category].count++;
    }

    // Calculate average scores per category
    for (const category of Object.keys(byCategory)) {
      byCategory[category].score = Math.round(
        byCategory[category].score / byCategory[category].count
      );
    }

    // Generate recommendations
    const recommendations: string[] = [];

    if (session.overallScore < 60) {
      recommendations.push('Review security fundamentals before continuing');
    }

    for (const weakness of session.weaknesses) {
      recommendations.push(\`Focus on improving \${weakness} knowledge\`);
    }

    if (session.evaluations.length < 5) {
      recommendations.push('Complete more questions for a comprehensive assessment');
    }

    if (recommendations.length === 0) {
      recommendations.push('Great progress! Continue practicing with harder questions');
    }

    return {
      totalQuestions: session.questions.length,
      averageScore: session.overallScore,
      byCategory,
      recommendations,
    };
  }

  identifyWeakAreas(sessionId: string): string[] {
    const session = this.sessions.get(sessionId);
    if (!session) return [];

    const weakAreas: string[] = [...session.weaknesses];

    // Also analyze missed key points
    const missedCounts: Record<string, number> = {};

    for (const evaluation of session.evaluations) {
      for (const missed of evaluation.missedKeyPoints) {
        missedCounts[missed] = (missedCounts[missed] || 0) + 1;
      }
    }

    // Add frequently missed topics
    const sortedMissed = Object.entries(missedCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([topic]) => topic);

    for (const topic of sortedMissed) {
      if (!weakAreas.includes(topic)) {
        weakAreas.push(topic);
      }
    }

    return weakAreas;
  }

  generateStudyPlan(sessionId: string): {
    topic: string;
    priority: 'high' | 'medium' | 'low';
    resources: string[];
  }[] {
    const session = this.sessions.get(sessionId);
    if (!session) return [];

    const plan: { topic: string; priority: 'high' | 'medium' | 'low'; resources: string[] }[] = [];

    const resourceMap: Record<string, string[]> = {
      fundamentals: ['NIST Cybersecurity Framework', 'CompTIA Security+ Study Guide', 'OWASP Top 10'],
      cryptography: ['Applied Cryptography by Bruce Schneier', 'Cryptopals Challenges', 'Khan Academy Cryptography'],
      web_security: ['OWASP Web Security Testing Guide', 'PortSwigger Web Security Academy', 'HackTheBox'],
      network: ['Network+ Study Guide', 'Wireshark Documentation', 'TCP/IP Illustrated'],
      compliance: ['GDPR Guidelines', 'HIPAA Security Rule', 'PCI-DSS Quick Reference'],
    };

    // Add high priority items for weaknesses
    for (const weakness of session.weaknesses) {
      plan.push({
        topic: weakness,
        priority: 'high',
        resources: resourceMap[weakness] || ['General security documentation'],
      });
    }

    // Add medium priority for categories with moderate scores
    const summary = this.getSessionSummary(sessionId);
    for (const [category, data] of Object.entries(summary.byCategory)) {
      if (data.score >= 50 && data.score < 70 && !session.weaknesses.includes(category)) {
        plan.push({
          topic: category,
          priority: 'medium',
          resources: resourceMap[category] || ['General security documentation'],
        });
      }
    }

    // Add low priority for areas not yet covered
    const allCategories: Question['category'][] = ['fundamentals', 'cryptography', 'web_security', 'network', 'compliance'];
    const coveredCategories = new Set(Object.keys(summary.byCategory));

    for (const category of allCategories) {
      if (!coveredCategories.has(category)) {
        plan.push({
          topic: category,
          priority: 'low',
          resources: resourceMap[category] || ['General security documentation'],
        });
      }
    }

    return plan;
  }
}

// Test your implementation
const bot = new SecurityInterviewBot();

// Test 1: Add question
bot.addQuestion({
  id: 'q1',
  category: 'fundamentals',
  question: 'What is the CIA triad?',
  keyPoints: ['confidentiality', 'integrity', 'availability', 'security principles'],
  difficulty: 'easy',
});
console.log('Test 1 - Question added:', bot.getQuestionsByCategory('fundamentals').length === 1);

// Test 2: Get random question
bot.addQuestion({
  id: 'q2',
  category: 'cryptography',
  question: 'Explain symmetric vs asymmetric encryption',
  keyPoints: ['same key', 'public/private key pair', 'performance', 'key exchange'],
  difficulty: 'medium',
});
const randomQ = bot.getRandomQuestion();
console.log('Test 2 - Random question:', randomQ !== null);

// Test 3: Start session
const sessionId = bot.startSession();
console.log('Test 3 - Session started:', sessionId.length > 0);

// Test 4: Ask question
const question = bot.askQuestion(sessionId);
console.log('Test 4 - Question asked:', question !== null);

// Test 5: Evaluate answer
const eval1 = bot.evaluateAnswer(
  { id: 'q1', category: 'fundamentals', question: 'CIA triad?', keyPoints: ['confidentiality', 'integrity', 'availability'], difficulty: 'easy' },
  'The CIA triad consists of Confidentiality, Integrity, and Availability - the core principles of information security.'
);
console.log('Test 5 - Answer evaluated:', eval1.score > 50);

// Test 6: Submit answer in session
const eval2 = bot.submitAnswer(sessionId, question?.id || 'q1', 'Security answer here');
console.log('Test 6 - Answer submitted:', 'score' in eval2);

// Test 7: Key point coverage
const coverage = bot.calculateKeyPointCoverage(
  ['encryption', 'decryption', 'key management'],
  'Encryption transforms data using keys, decryption reverses it.'
);
console.log('Test 7 - Coverage calculated:', coverage.matched.length >= 2);

// Test 8: Session summary
const summary = bot.getSessionSummary(sessionId);
console.log('Test 8 - Summary generated:', 'totalQuestions' in summary);

// Test 9: Weak areas
const weakAreas = bot.identifyWeakAreas(sessionId);
console.log('Test 9 - Weak areas:', Array.isArray(weakAreas));

// Test 10: Study plan
const plan = bot.generateStudyPlan(sessionId);
console.log('Test 10 - Study plan:', Array.isArray(plan));`,
	testCode: `import { describe, it, expect, beforeEach } from 'vitest';

interface Question {
  id: string;
  category: 'fundamentals' | 'cryptography' | 'web_security' | 'network' | 'compliance';
  question: string;
  keyPoints: string[];
  difficulty: 'easy' | 'medium' | 'hard';
  followUpQuestions?: string[];
}

interface Answer {
  questionId: string;
  response: string;
  timestamp: Date;
}

interface Evaluation {
  score: number;
  matchedKeyPoints: string[];
  missedKeyPoints: string[];
  feedback: string;
  suggestedImprovement: string;
}

interface InterviewSession {
  id: string;
  startTime: Date;
  questions: Question[];
  answers: Answer[];
  evaluations: Evaluation[];
  overallScore: number;
  strengths: string[];
  weaknesses: string[];
}

class SecurityInterviewBot {
  private questionBank: Question[] = [];
  private sessions: Map<string, InterviewSession> = new Map();

  addQuestion(question: Question): void {
    if (!this.questionBank.find(q => q.id === question.id)) {
      this.questionBank.push(question);
    }
  }

  getQuestionsByCategory(category: Question['category']): Question[] {
    return this.questionBank.filter(q => q.category === category);
  }

  getRandomQuestion(categories?: Question['category'][]): Question | null {
    let pool = this.questionBank;
    if (categories && categories.length > 0) {
      pool = this.questionBank.filter(q => categories.includes(q.category));
    }
    if (pool.length === 0) return null;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  startSession(): string {
    const sessionId = \`session-\${Date.now()}-\${Math.random().toString(36).substring(2, 8)}\`;
    const session: InterviewSession = {
      id: sessionId, startTime: new Date(), questions: [], answers: [],
      evaluations: [], overallScore: 0, strengths: [], weaknesses: [],
    };
    this.sessions.set(sessionId, session);
    return sessionId;
  }

  askQuestion(sessionId: string, questionId?: string): Question | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    let question: Question | null = null;
    if (questionId) {
      question = this.questionBank.find(q => q.id === questionId) || null;
    } else {
      const askedIds = new Set(session.questions.map(q => q.id));
      const available = this.questionBank.filter(q => !askedIds.has(q.id));
      if (available.length > 0) {
        question = available[Math.floor(Math.random() * available.length)];
      }
    }
    if (question) session.questions.push(question);
    return question;
  }

  submitAnswer(sessionId: string, questionId: string, response: string): Evaluation {
    const session = this.sessions.get(sessionId);
    if (!session) return { score: 0, matchedKeyPoints: [], missedKeyPoints: [], feedback: 'Session not found', suggestedImprovement: '' };

    const question = session.questions.find(q => q.id === questionId);
    if (!question) return { score: 0, matchedKeyPoints: [], missedKeyPoints: [], feedback: 'Question not found', suggestedImprovement: '' };

    session.answers.push({ questionId, response, timestamp: new Date() });
    const evaluation = this.evaluateAnswer(question, response);
    session.evaluations.push(evaluation);
    this.updateSessionStats(session);
    return evaluation;
  }

  evaluateAnswer(question: Question, response: string): Evaluation {
    const coverage = this.calculateKeyPointCoverage(question.keyPoints, response);
    let score = coverage.coverage * 100;
    const multiplier = { easy: 0.9, medium: 1.0, hard: 1.1 };
    score = Math.min(100, score * multiplier[question.difficulty]);

    const wordCount = response.split(/\s+/).length;
    if (wordCount > 50) score = Math.min(100, score + 5);
    if (wordCount > 100) score = Math.min(100, score + 5);

    let feedback = score >= 80 ? 'Excellent!' : score >= 60 ? 'Good but missing points' : score >= 40 ? 'Partial answer' : 'Needs improvement';
    const suggestedImprovement = coverage.missed.length > 0 ? \`Focus on: \${coverage.missed.slice(0, 3).join(', ')}\` : 'Add examples';

    return { score: Math.round(score), matchedKeyPoints: coverage.matched, missedKeyPoints: coverage.missed, feedback, suggestedImprovement };
  }

  calculateKeyPointCoverage(keyPoints: string[], response: string): { matched: string[]; missed: string[]; coverage: number } {
    const responseLower = response.toLowerCase();
    const matched: string[] = [];
    const missed: string[] = [];

    for (const keyPoint of keyPoints) {
      const words = keyPoint.toLowerCase().split(/\s+/);
      const hasMatch = words.some(w => w.length > 3 && responseLower.includes(w)) || responseLower.includes(keyPoint.toLowerCase());
      if (hasMatch) matched.push(keyPoint);
      else missed.push(keyPoint);
    }
    return { matched, missed, coverage: keyPoints.length > 0 ? matched.length / keyPoints.length : 0 };
  }

  private updateSessionStats(session: InterviewSession): void {
    if (session.evaluations.length === 0) { session.overallScore = 0; return; }
    session.overallScore = Math.round(session.evaluations.reduce((sum, e) => sum + e.score, 0) / session.evaluations.length);

    const categoryScores: Record<string, { total: number; count: number }> = {};
    for (let i = 0; i < session.questions.length; i++) {
      const q = session.questions[i], e = session.evaluations[i];
      if (!e) continue;
      if (!categoryScores[q.category]) categoryScores[q.category] = { total: 0, count: 0 };
      categoryScores[q.category].total += e.score;
      categoryScores[q.category].count++;
    }

    session.strengths = [];
    session.weaknesses = [];
    for (const [cat, data] of Object.entries(categoryScores)) {
      const avg = data.total / data.count;
      if (avg >= 70) session.strengths.push(cat);
      else if (avg < 50) session.weaknesses.push(cat);
    }
  }

  getSessionSummary(sessionId: string): { totalQuestions: number; averageScore: number; byCategory: Record<string, { score: number; count: number }>; recommendations: string[] } {
    const session = this.sessions.get(sessionId);
    if (!session) return { totalQuestions: 0, averageScore: 0, byCategory: {}, recommendations: [] };

    const byCategory: Record<string, { score: number; count: number }> = {};
    for (let i = 0; i < session.questions.length; i++) {
      const q = session.questions[i], e = session.evaluations[i];
      if (!e) continue;
      if (!byCategory[q.category]) byCategory[q.category] = { score: 0, count: 0 };
      byCategory[q.category].score += e.score;
      byCategory[q.category].count++;
    }
    for (const cat of Object.keys(byCategory)) {
      byCategory[cat].score = Math.round(byCategory[cat].score / byCategory[cat].count);
    }

    const recommendations: string[] = [];
    if (session.overallScore < 60) recommendations.push('Review fundamentals');
    for (const w of session.weaknesses) recommendations.push(\`Improve \${w}\`);
    if (recommendations.length === 0) recommendations.push('Keep practicing');

    return { totalQuestions: session.questions.length, averageScore: session.overallScore, byCategory, recommendations };
  }

  identifyWeakAreas(sessionId: string): string[] {
    const session = this.sessions.get(sessionId);
    if (!session) return [];
    return [...session.weaknesses];
  }

  generateStudyPlan(sessionId: string): { topic: string; priority: 'high' | 'medium' | 'low'; resources: string[] }[] {
    const session = this.sessions.get(sessionId);
    if (!session) return [];

    const plan: { topic: string; priority: 'high' | 'medium' | 'low'; resources: string[] }[] = [];
    const resources: Record<string, string[]> = {
      fundamentals: ['CompTIA Security+', 'OWASP'],
      cryptography: ['Applied Cryptography', 'Cryptopals'],
      web_security: ['PortSwigger Academy', 'OWASP'],
      network: ['Network+', 'Wireshark'],
      compliance: ['GDPR', 'PCI-DSS'],
    };

    for (const w of session.weaknesses) {
      plan.push({ topic: w, priority: 'high', resources: resources[w] || ['General docs'] });
    }
    return plan;
  }
}

describe('SecurityInterviewBot', () => {
  let bot: SecurityInterviewBot;

  beforeEach(() => {
    bot = new SecurityInterviewBot();
  });

  it('should add and retrieve questions', () => {
    bot.addQuestion({
      id: 'q1',
      category: 'fundamentals',
      question: 'What is the CIA triad?',
      keyPoints: ['confidentiality', 'integrity', 'availability'],
      difficulty: 'easy',
    });

    expect(bot.getQuestionsByCategory('fundamentals')).toHaveLength(1);
    expect(bot.getQuestionsByCategory('cryptography')).toHaveLength(0);
  });

  it('should get random questions', () => {
    bot.addQuestion({ id: 'q1', category: 'fundamentals', question: 'Q1', keyPoints: [], difficulty: 'easy' });
    bot.addQuestion({ id: 'q2', category: 'cryptography', question: 'Q2', keyPoints: [], difficulty: 'medium' });

    const random = bot.getRandomQuestion();
    expect(random).not.toBeNull();

    const filtered = bot.getRandomQuestion(['cryptography']);
    expect(filtered?.category).toBe('cryptography');
  });

  it('should start and manage sessions', () => {
    const sessionId = bot.startSession();
    expect(sessionId).toMatch(/^session-/);
  });

  it('should evaluate answers based on key points', () => {
    const question: Question = {
      id: 'q1',
      category: 'fundamentals',
      question: 'CIA triad?',
      keyPoints: ['confidentiality', 'integrity', 'availability'],
      difficulty: 'easy',
    };

    const goodAnswer = bot.evaluateAnswer(question, 'Confidentiality protects data, integrity ensures accuracy, availability means accessible.');
    const badAnswer = bot.evaluateAnswer(question, 'Security is important.');

    expect(goodAnswer.score).toBeGreaterThan(badAnswer.score);
    expect(goodAnswer.matchedKeyPoints.length).toBeGreaterThan(badAnswer.matchedKeyPoints.length);
  });

  it('should calculate key point coverage', () => {
    const coverage = bot.calculateKeyPointCoverage(
      ['encryption', 'decryption', 'key management'],
      'Encryption transforms data, decryption reverses it.'
    );

    expect(coverage.matched).toContain('encryption');
    expect(coverage.matched).toContain('decryption');
    expect(coverage.missed).toContain('key management');
    expect(coverage.coverage).toBeGreaterThan(0.5);
  });

  it('should track answers in session', () => {
    bot.addQuestion({ id: 'q1', category: 'fundamentals', question: 'CIA?', keyPoints: ['confidentiality'], difficulty: 'easy' });

    const sessionId = bot.startSession();
    const q = bot.askQuestion(sessionId, 'q1');
    const evaluation = bot.submitAnswer(sessionId, 'q1', 'Confidentiality protects data');

    expect(q?.id).toBe('q1');
    expect(evaluation.score).toBeGreaterThan(0);
  });

  it('should generate session summary', () => {
    bot.addQuestion({ id: 'q1', category: 'fundamentals', question: 'Q1', keyPoints: ['point1'], difficulty: 'easy' });

    const sessionId = bot.startSession();
    bot.askQuestion(sessionId, 'q1');
    bot.submitAnswer(sessionId, 'q1', 'Answer covering point1');

    const summary = bot.getSessionSummary(sessionId);

    expect(summary.totalQuestions).toBe(1);
    expect(summary.byCategory.fundamentals).toBeDefined();
    expect(summary.recommendations.length).toBeGreaterThan(0);
  });

  it('should identify weak areas', () => {
    bot.addQuestion({ id: 'q1', category: 'fundamentals', question: 'Q1', keyPoints: ['point1', 'point2', 'point3'], difficulty: 'easy' });

    const sessionId = bot.startSession();
    bot.askQuestion(sessionId, 'q1');
    bot.submitAnswer(sessionId, 'q1', 'Very poor answer');

    const weakAreas = bot.identifyWeakAreas(sessionId);
    expect(Array.isArray(weakAreas)).toBe(true);
  });

  it('should generate study plan', () => {
    bot.addQuestion({ id: 'q1', category: 'cryptography', question: 'Encryption?', keyPoints: ['symmetric', 'asymmetric'], difficulty: 'medium' });

    const sessionId = bot.startSession();
    bot.askQuestion(sessionId, 'q1');
    bot.submitAnswer(sessionId, 'q1', 'Bad answer');

    const plan = bot.generateStudyPlan(sessionId);

    expect(Array.isArray(plan)).toBe(true);
  });

  it('should not ask same question twice', () => {
    bot.addQuestion({ id: 'q1', category: 'fundamentals', question: 'Q1', keyPoints: [], difficulty: 'easy' });

    const sessionId = bot.startSession();
    const first = bot.askQuestion(sessionId);
    const second = bot.askQuestion(sessionId);

    expect(first?.id).toBe('q1');
    expect(second).toBeNull();
  });
});`,
	hint1:
		'For key point coverage, convert both the response and key points to lowercase, then check if each key point word (longer than 3 characters) appears in the response. Calculate coverage as matched / total.',
	hint2:
		'For session management, use a Map to store sessions by ID. Track questions asked to avoid repetition. Update overall score after each answer submission.',
	whyItMatters: `Security interview preparation is essential for landing security roles:

**Common Interview Questions:**
- "What is the CIA triad?" - Tests fundamental understanding
- "Explain the difference between encryption and hashing" - Technical depth
- "How would you respond to a data breach?" - Practical application
- "Describe your experience with OWASP Top 10" - Industry knowledge

**Interview Success Factors:**
- **Clear explanations** - Can you explain complex concepts simply?
- **Real-world examples** - Can you relate theory to practice?
- **Problem-solving approach** - How do you think through security challenges?

**Industry Statistics:**
- Security job openings grew 35% in 2024
- Average security engineer salary: $120k-$180k
- Most common certification requirements: Security+, CISSP, CEH

Structured interview practice significantly improves performance and confidence.`,
	order: 1,
	translations: {
		ru: {
			title: 'Вопросы собеседования по безопасности',
			description: `Освойте распространённые вопросы собеседования по безопасности через интерактивные упражнения.

**Ключевые темы собеседования:**

1. **Триада CIA** - Конфиденциальность, Целостность, Доступность
2. **Аутентификация vs Авторизация** - Проверка личности vs контроль доступа
3. **Типы шифрования** - Симметричное vs Асимметричное
4. **Распространённые уязвимости** - Понимание OWASP Top 10
5. **Протоколы безопасности** - TLS, OAuth, JWT

**Ваша задача:**

Реализуйте класс \`SecurityInterviewBot\`, который может проверять ваши знания по концепциям безопасности.`,
			hint1:
				'Для покрытия ключевых точек переведите ответ и точки в нижний регистр, проверьте присутствие каждого слова (длиннее 3 символов) в ответе.',
			hint2:
				'Для управления сессиями используйте Map для хранения по ID. Отслеживайте заданные вопросы, чтобы избежать повторений.',
			whyItMatters: `Подготовка к собеседованию по безопасности необходима для получения роли в безопасности:

**Распространённые вопросы:**
- "Что такое триада CIA?" - Тест на фундаментальное понимание
- "Объясните разницу между шифрованием и хешированием" - Техническая глубина
- "Как бы вы отреагировали на утечку данных?" - Практическое применение

**Факторы успеха:**
- **Чёткие объяснения** - Можете ли вы объяснить сложные концепции просто?
- **Реальные примеры** - Можете ли вы связать теорию с практикой?
- **Подход к решению проблем** - Как вы думаете о проблемах безопасности?

Структурированная практика значительно улучшает результаты и уверенность.`,
		},
		uz: {
			title: 'Xavfsizlik intervyu savollari',
			description: `Keng tarqalgan xavfsizlik intervyu savollarini interaktiv mashqlar orqali o'rganing.

**Asosiy intervyu mavzulari:**

1. **CIA Triada** - Maxfiylik, Yaxlitlik, Mavjudlik
2. **Autentifikatsiya vs Avtorizatsiya** - Shaxsni tekshirish vs kirish nazorati
3. **Shifrlash turlari** - Simmetrik vs Asimmetrik
4. **Keng tarqalgan zaifliklar** - OWASP Top 10 tushunish
5. **Xavfsizlik protokollari** - TLS, OAuth, JWT

**Vazifangiz:**

Xavfsizlik tushunchalari bo'yicha bilimingizni tekshira oladigan \`SecurityInterviewBot\` klassini yarating.`,
			hint1:
				"Kalit nuqtalarni qoplash uchun javob va nuqtalarni kichik harflarga o'tkazing, har bir so'zning (3 belgidan uzun) javobda mavjudligini tekshiring.",
			hint2:
				"Sessiyalarni boshqarish uchun ID bo'yicha saqlash uchun Map dan foydalaning. Takrorlanishning oldini olish uchun berilgan savollarni kuzatib boring.",
			whyItMatters: `Xavfsizlik intervyusiga tayyorgarlik xavfsizlik rolini olish uchun zarur:

**Keng tarqalgan savollar:**
- "CIA triada nima?" - Asosiy tushunishni tekshirish
- "Shifrlash va heshlash o'rtasidagi farqni tushuntiring" - Texnik chuqurlik
- "Ma'lumotlar sizib chiqishiga qanday javob berardingiz?" - Amaliy qo'llash

**Muvaffaqiyat omillari:**
- **Aniq tushuntirishlar** - Murakkab tushunchalarni sodda tushuntira olasizmi?
- **Haqiqiy misollar** - Nazariyani amaliyot bilan bog'lay olasizmi?
- **Muammoni hal qilish yondashuvi** - Xavfsizlik muammolari haqida qanday o'ylaysiz?

Tuzilgan amaliyot natijalar va ishonchni sezilarli darajada yaxshilaydi.`,
		},
	},
};

export default task;
