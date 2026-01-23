import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'cert-cissp-prep',
	title: 'CISSP Certification Practice',
	difficulty: 'hard',
	tags: ['security', 'certification', 'cissp', 'typescript'],
	estimatedTime: '40m',
	isPremium: true,
	youtubeUrl: '',
	description: `Practice CISSP (Certified Information Systems Security Professional) concepts through coding.

**CISSP Overview:**

The CISSP covers 8 domains of security:

1. **Security and Risk Management** - Governance, compliance, risk
2. **Asset Security** - Data classification, handling
3. **Security Architecture** - Design principles, models
4. **Communication & Network Security** - Network architecture
5. **Identity & Access Management** - Authentication, authorization
6. **Security Assessment & Testing** - Audits, testing
7. **Security Operations** - Incident response, monitoring
8. **Software Development Security** - SDLC security

**Your Task:**

Implement a \`CISSPQuizEngine\` class that tests knowledge across all 8 domains.`,
	initialCode: `interface Question {
  id: string;
  domain: number; // 1-8
  question: string;
  options: string[];
  correctAnswer: number; // 0-based index
  explanation: string;
}

interface QuizResult {
  totalQuestions: number;
  correctAnswers: number;
  scorePercentage: number;
  domainScores: Record<number, { correct: number; total: number }>;
  passed: boolean; // 70% to pass
  weakDomains: number[];
}

interface StudyRecommendation {
  domain: number;
  domainName: string;
  priority: 'high' | 'medium' | 'low';
  topics: string[];
}

class CISSPQuizEngine {
  private questions: Question[] = [];
  private domainNames: Record<number, string> = {
    1: 'Security and Risk Management',
    2: 'Asset Security',
    3: 'Security Architecture and Engineering',
    4: 'Communication and Network Security',
    5: 'Identity and Access Management',
    6: 'Security Assessment and Testing',
    7: 'Security Operations',
    8: 'Software Development Security',
  };

  addQuestion(question: Question): void {
    // TODO: Add a question to the quiz bank
  }

  getQuestionsByDomain(domain: number): Question[] {
    // TODO: Get all questions for a specific domain
    return [];
  }

  generateQuiz(numQuestions: number, domains?: number[]): Question[] {
    // TODO: Generate a random quiz
    // If domains specified, only include those
    return [];
  }

  evaluateAnswers(questions: Question[], answers: number[]): QuizResult {
    // TODO: Evaluate quiz answers and calculate scores
    return {} as QuizResult;
  }

  getStudyRecommendations(result: QuizResult): StudyRecommendation[] {
    // TODO: Generate study recommendations based on weak areas
    return [];
  }

  getDomainWeight(domain: number): number {
    // TODO: Return CISSP exam domain weights
    // Domain 1: 15%, Domain 2: 10%, etc.
    return 0;
  }

  calculateWeightedScore(domainScores: Record<number, { correct: number; total: number }>): number {
    // TODO: Calculate weighted score based on domain weights
    return 0;
  }

  shuffleOptions(question: Question): Question {
    // TODO: Shuffle options while tracking correct answer
    return question;
  }
}

export { CISSPQuizEngine, Question, QuizResult, StudyRecommendation };`,
	solutionCode: `interface Question {
  id: string;
  domain: number;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

interface QuizResult {
  totalQuestions: number;
  correctAnswers: number;
  scorePercentage: number;
  domainScores: Record<number, { correct: number; total: number }>;
  passed: boolean;
  weakDomains: number[];
}

interface StudyRecommendation {
  domain: number;
  domainName: string;
  priority: 'high' | 'medium' | 'low';
  topics: string[];
}

class CISSPQuizEngine {
  private questions: Question[] = [];
  private domainNames: Record<number, string> = {
    1: 'Security and Risk Management',
    2: 'Asset Security',
    3: 'Security Architecture and Engineering',
    4: 'Communication and Network Security',
    5: 'Identity and Access Management',
    6: 'Security Assessment and Testing',
    7: 'Security Operations',
    8: 'Software Development Security',
  };

  private domainWeights: Record<number, number> = {
    1: 15, 2: 10, 3: 13, 4: 13,
    5: 13, 6: 12, 7: 13, 8: 11,
  };

  private domainTopics: Record<number, string[]> = {
    1: ['Risk assessment', 'Security governance', 'Compliance', 'Business continuity'],
    2: ['Data classification', 'Data retention', 'Privacy protection'],
    3: ['Security models', 'Cryptography', 'Physical security'],
    4: ['Network protocols', 'Firewalls', 'VPNs', 'Wireless security'],
    5: ['Authentication methods', 'Authorization', 'Identity management'],
    6: ['Vulnerability assessment', 'Penetration testing', 'Auditing'],
    7: ['Incident response', 'Disaster recovery', 'Logging and monitoring'],
    8: ['SDLC', 'Secure coding', 'Code review', 'DevSecOps'],
  };

  addQuestion(question: Question): void {
    if (question.domain < 1 || question.domain > 8) {
      throw new Error('Domain must be between 1 and 8');
    }
    this.questions.push(question);
  }

  getQuestionsByDomain(domain: number): Question[] {
    return this.questions.filter(q => q.domain === domain);
  }

  generateQuiz(numQuestions: number, domains?: number[]): Question[] {
    let pool = this.questions;

    if (domains && domains.length > 0) {
      pool = pool.filter(q => domains.includes(q.domain));
    }

    // Shuffle and take requested number
    const shuffled = [...pool].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(numQuestions, shuffled.length));
  }

  evaluateAnswers(questions: Question[], answers: number[]): QuizResult {
    const domainScores: Record<number, { correct: number; total: number }> = {};

    // Initialize domain scores
    for (let d = 1; d <= 8; d++) {
      domainScores[d] = { correct: 0, total: 0 };
    }

    let correctAnswers = 0;

    for (let i = 0; i < questions.length; i++) {
      const question = questions[i];
      const answer = answers[i];

      domainScores[question.domain].total++;

      if (answer === question.correctAnswer) {
        correctAnswers++;
        domainScores[question.domain].correct++;
      }
    }

    const scorePercentage = questions.length > 0
      ? (correctAnswers / questions.length) * 100
      : 0;

    // Find weak domains (less than 60% correct)
    const weakDomains: number[] = [];
    for (const [domain, scores] of Object.entries(domainScores)) {
      if (scores.total > 0 && (scores.correct / scores.total) < 0.6) {
        weakDomains.push(parseInt(domain, 10));
      }
    }

    return {
      totalQuestions: questions.length,
      correctAnswers,
      scorePercentage: Math.round(scorePercentage * 10) / 10,
      domainScores,
      passed: scorePercentage >= 70,
      weakDomains,
    };
  }

  getStudyRecommendations(result: QuizResult): StudyRecommendation[] {
    const recommendations: StudyRecommendation[] = [];

    for (const [domainStr, scores] of Object.entries(result.domainScores)) {
      const domain = parseInt(domainStr, 10);
      if (scores.total === 0) continue;

      const percentage = (scores.correct / scores.total) * 100;
      let priority: 'high' | 'medium' | 'low';

      if (percentage < 50) {
        priority = 'high';
      } else if (percentage < 70) {
        priority = 'medium';
      } else {
        priority = 'low';
      }

      if (priority !== 'low') {
        recommendations.push({
          domain,
          domainName: this.domainNames[domain],
          priority,
          topics: this.domainTopics[domain] || [],
        });
      }
    }

    // Sort by priority (high first)
    const priorityOrder = { 'high': 0, 'medium': 1, 'low': 2 };
    return recommendations.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);
  }

  getDomainWeight(domain: number): number {
    return this.domainWeights[domain] || 0;
  }

  calculateWeightedScore(domainScores: Record<number, { correct: number; total: number }>): number {
    let weightedSum = 0;
    let totalWeight = 0;

    for (const [domainStr, scores] of Object.entries(domainScores)) {
      const domain = parseInt(domainStr, 10);
      if (scores.total === 0) continue;

      const weight = this.domainWeights[domain];
      const domainScore = scores.correct / scores.total;

      weightedSum += domainScore * weight;
      totalWeight += weight;
    }

    return totalWeight > 0 ? (weightedSum / totalWeight) * 100 : 0;
  }

  shuffleOptions(question: Question): Question {
    const indices = question.options.map((_, i) => i);

    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const newOptions = indices.map(i => question.options[i]);
    const newCorrectAnswer = indices.indexOf(question.correctAnswer);

    return {
      ...question,
      options: newOptions,
      correctAnswer: newCorrectAnswer,
    };
  }
}

export { CISSPQuizEngine, Question, QuizResult, StudyRecommendation };`,
	hint1: `For evaluateAnswers, iterate through questions and answers, track correct/total per domain, calculate percentage, and determine if passed (≥70%).`,
	hint2: `For calculateWeightedScore, multiply each domain's score percentage by its weight, sum up, and divide by total weight.`,
	testCode: `import { CISSPQuizEngine } from './solution';

const sampleQuestions = [
  { id: 'q1', domain: 1, question: 'What is the primary goal of risk management?', options: ['A', 'B', 'C', 'D'], correctAnswer: 0, explanation: '' },
  { id: 'q2', domain: 1, question: 'BCP stands for?', options: ['A', 'B', 'C', 'D'], correctAnswer: 1, explanation: '' },
  { id: 'q3', domain: 5, question: 'What is MFA?', options: ['A', 'B', 'C', 'D'], correctAnswer: 2, explanation: '' },
  { id: 'q4', domain: 7, question: 'What is SIEM?', options: ['A', 'B', 'C', 'D'], correctAnswer: 0, explanation: '' },
];

// Test1: addQuestion adds to bank
test('Test1', () => {
  const engine = new CISSPQuizEngine();
  engine.addQuestion(sampleQuestions[0]);
  expect(engine.getQuestionsByDomain(1).length).toBe(1);
});

// Test2: getQuestionsByDomain filters correctly
test('Test2', () => {
  const engine = new CISSPQuizEngine();
  sampleQuestions.forEach(q => engine.addQuestion(q));
  expect(engine.getQuestionsByDomain(1).length).toBe(2);
  expect(engine.getQuestionsByDomain(5).length).toBe(1);
});

// Test3: generateQuiz returns requested number
test('Test3', () => {
  const engine = new CISSPQuizEngine();
  sampleQuestions.forEach(q => engine.addQuestion(q));
  const quiz = engine.generateQuiz(2);
  expect(quiz.length).toBe(2);
});

// Test4: evaluateAnswers calculates correct score
test('Test4', () => {
  const engine = new CISSPQuizEngine();
  const questions = sampleQuestions.slice(0, 2);
  const answers = [0, 1]; // Both correct
  const result = engine.evaluateAnswers(questions, answers);
  expect(result.scorePercentage).toBe(100);
  expect(result.passed).toBe(true);
});

// Test5: evaluateAnswers identifies failures
test('Test5', () => {
  const engine = new CISSPQuizEngine();
  const questions = sampleQuestions.slice(0, 4);
  const answers = [0, 0, 0, 0]; // 2 correct out of 4 = 50%
  const result = engine.evaluateAnswers(questions, answers);
  expect(result.passed).toBe(false);
});

// Test6: getDomainWeight returns correct values
test('Test6', () => {
  const engine = new CISSPQuizEngine();
  expect(engine.getDomainWeight(1)).toBe(15);
  expect(engine.getDomainWeight(2)).toBe(10);
});

// Test7: getStudyRecommendations prioritizes weak areas
test('Test7', () => {
  const engine = new CISSPQuizEngine();
  const result = {
    totalQuestions: 4,
    correctAnswers: 1,
    scorePercentage: 25,
    domainScores: { 1: { correct: 0, total: 2 }, 5: { correct: 1, total: 1 }, 7: { correct: 0, total: 1 } },
    passed: false,
    weakDomains: [1, 7],
  };
  const recommendations = engine.getStudyRecommendations(result as any);
  expect(recommendations[0].priority).toBe('high');
});

// Test8: calculateWeightedScore works
test('Test8', () => {
  const engine = new CISSPQuizEngine();
  const domainScores = { 1: { correct: 10, total: 10 }, 2: { correct: 5, total: 10 } };
  const score = engine.calculateWeightedScore(domainScores);
  expect(score).toBeGreaterThan(0);
  expect(score).toBeLessThan(100);
});

// Test9: shuffleOptions changes order
test('Test9', () => {
  const engine = new CISSPQuizEngine();
  const original = { ...sampleQuestions[0], options: ['A', 'B', 'C', 'D'] };
  // Run multiple times to ensure shuffling happens
  let changed = false;
  for (let i = 0; i < 10; i++) {
    const shuffled = engine.shuffleOptions({ ...original });
    if (shuffled.options.join('') !== 'ABCD') {
      changed = true;
      break;
    }
  }
  // Can't guarantee change, but structure should be valid
  expect(engine.shuffleOptions(original).options.length).toBe(4);
});

// Test10: generateQuiz filters by domain
test('Test10', () => {
  const engine = new CISSPQuizEngine();
  sampleQuestions.forEach(q => engine.addQuestion(q));
  const quiz = engine.generateQuiz(10, [1]);
  expect(quiz.every(q => q.domain === 1)).toBe(true);
});`,
	whyItMatters: `CISSP is the gold standard for security professionals and opens doors to senior roles.

**CISSP Requirements:**

- 5 years experience in 2+ domains (or 4 years + degree)
- Pass the exam (125-175 adaptive questions, 4 hours)
- Endorsement by CISSP holder
- CPE credits for maintenance

**Career Impact:**

| Role | Avg Salary (US) |
|------|-----------------|
| Security Analyst | $75K |
| Security Engineer | $95K |
| CISSP holder | $120K+ |
| Security Architect | $150K+ |
| CISO | $200K+ |

**Study Strategy:**

1. **Official Study Guide** - Read cover to cover
2. **Practice Tests** - At least 1000 questions
3. **Mind Maps** - Visualize domain relationships
4. **Flashcards** - Key terms and definitions
5. **Study Groups** - Learn from others

**Exam Tips:**

- Think like a manager, not a technician
- Security > Availability in most scenarios
- Human safety always comes first
- Read questions carefully for keywords`,
	order: 0,
	translations: {
		ru: {
			title: 'Практика сертификации CISSP',
			description: `Практикуйте концепции CISSP через программирование.

**Обзор CISSP:**

CISSP охватывает 8 доменов безопасности:

1. Управление безопасностью и рисками
2. Безопасность активов
3. Архитектура безопасности
4. Безопасность коммуникаций и сетей
5. Управление идентификацией и доступом
6. Оценка и тестирование безопасности
7. Операции безопасности
8. Безопасность разработки ПО

**Ваша задача:**

Реализуйте класс \`CISSPQuizEngine\`.`,
			hint1: `Для evaluateAnswers пройдите по вопросам и ответам, отслеживайте верные/всего по доменам, вычислите процент и определите прошёл ли (≥70%).`,
			hint2: `Для calculateWeightedScore умножьте процент каждого домена на его вес, суммируйте и разделите на общий вес.`,
			whyItMatters: `CISSP - золотой стандарт для специалистов по безопасности и открывает двери к старшим позициям.`
		},
		uz: {
			title: 'CISSP sertifikatsiyasi amaliyoti',
			description: `CISSP tushunchalarini dasturlash orqali mashq qiling.

**CISSP haqida:**

CISSP 8 ta xavfsizlik domenini qamrab oladi.

**Sizning vazifangiz:**

\`CISSPQuizEngine\` klassini amalga oshiring.`,
			hint1: `evaluateAnswers uchun savollar va javoblar bo'ylab o'ting, domenlar bo'yicha to'g'ri/jami ni kuzating, foizni hisoblang va o'tdimi aniqland (≥70%).`,
			hint2: `calculateWeightedScore uchun har bir domen foizini uning og'irligiga ko'paytiring, yig'ing va umumiy og'irlikka bo'ling.`,
			whyItMatters: `CISSP xavfsizlik mutaxassislari uchun oltin standart va yuqori lavozimlarga eshik ochadi.`
		}
	}
};

export default task;
