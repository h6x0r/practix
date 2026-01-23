import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-risk-assessment',
	title: 'Security Risk Assessment',
	difficulty: 'medium',
	tags: ['security', 'fundamentals', 'risk-assessment', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to perform security risk assessments using the Risk = Likelihood × Impact formula.

**What is Risk Assessment?**

Risk assessment is the process of identifying, analyzing, and evaluating risks. It helps prioritize security efforts by focusing on the most significant risks.

**Risk Formula:**

\`\`\`
Risk Score = Likelihood × Impact

Likelihood: Probability of threat occurring (1-5)
Impact: Damage if threat occurs (1-5)
Risk: Overall risk level (1-25)
\`\`\`

**Risk Matrix:**

|  | Impact: 1 | Impact: 2 | Impact: 3 | Impact: 4 | Impact: 5 |
|--|-----------|-----------|-----------|-----------|-----------|
| **Likelihood: 5** | 5 | 10 | 15 | 20 | 25 |
| **Likelihood: 4** | 4 | 8 | 12 | 16 | 20 |
| **Likelihood: 3** | 3 | 6 | 9 | 12 | 15 |
| **Likelihood: 2** | 2 | 4 | 6 | 8 | 10 |
| **Likelihood: 1** | 1 | 2 | 3 | 4 | 5 |

**Your Task:**

Implement a \`RiskAssessor\` class that:

1. Calculates risk scores from likelihood and impact
2. Categorizes risks by severity level
3. Prioritizes risks for treatment
4. Tracks risk treatment status`,
	initialCode: `interface Risk {
  id: string;
  name: string;
  description: string;
  likelihood: number; // 1-5
  impact: number;     // 1-5
  category: string;
  status: 'identified' | 'analyzing' | 'treating' | 'accepted' | 'mitigated';
}

type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

interface RiskAssessment {
  risk: Risk;
  score: number;
  level: RiskLevel;
  priority: number;
}

class RiskAssessor {
  private risks: Map<string, Risk> = new Map();
  private riskCounter = 0;

  addRisk(name: string, description: string, likelihood: number, impact: number, category: string): Risk {
    // TODO: Create and store a new risk
    // Validate likelihood and impact are 1-5
    return {} as Risk;
  }

  calculateScore(risk: Risk): number {
    // TODO: Calculate risk score = likelihood * impact
    return 0;
  }

  determineLevel(score: number): RiskLevel {
    // TODO: Determine risk level from score
    // 1-4: low, 5-9: medium, 10-15: high, 16-25: critical
    return 'low';
  }

  assessRisk(riskId: string): RiskAssessment | null {
    // TODO: Get full risk assessment
    return null;
  }

  getAllAssessments(): RiskAssessment[] {
    // TODO: Get assessments for all risks, sorted by score descending
    return [];
  }

  getPriorityRisks(minLevel: RiskLevel): Risk[] {
    // TODO: Get risks at or above specified level
    return [];
  }

  updateRiskStatus(riskId: string, status: Risk['status']): boolean {
    // TODO: Update risk treatment status
    return false;
  }

  getRisksByCategory(category: string): Risk[] {
    // TODO: Filter risks by category
    return [];
  }

  getAverageRiskScore(): number {
    // TODO: Calculate average risk score across all risks
    return 0;
  }

  generateRiskMatrix(): number[][] {
    // TODO: Generate 5x5 matrix showing risk count at each likelihood/impact
    return [];
  }
}

export { RiskAssessor, Risk, RiskLevel, RiskAssessment };`,
	solutionCode: `interface Risk {
  id: string;
  name: string;
  description: string;
  likelihood: number; // 1-5
  impact: number;     // 1-5
  category: string;
  status: 'identified' | 'analyzing' | 'treating' | 'accepted' | 'mitigated';
}

type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

interface RiskAssessment {
  risk: Risk;
  score: number;
  level: RiskLevel;
  priority: number;
}

class RiskAssessor {
  private risks: Map<string, Risk> = new Map();
  private riskCounter = 0;

  addRisk(name: string, description: string, likelihood: number, impact: number, category: string): Risk {
    const validLikelihood = Math.max(1, Math.min(5, Math.round(likelihood)));
    const validImpact = Math.max(1, Math.min(5, Math.round(impact)));

    const risk: Risk = {
      id: \`R-\${++this.riskCounter}\`,
      name,
      description,
      likelihood: validLikelihood,
      impact: validImpact,
      category,
      status: 'identified',
    };

    this.risks.set(risk.id, risk);
    return risk;
  }

  calculateScore(risk: Risk): number {
    return risk.likelihood * risk.impact;
  }

  determineLevel(score: number): RiskLevel {
    if (score >= 16) return 'critical';
    if (score >= 10) return 'high';
    if (score >= 5) return 'medium';
    return 'low';
  }

  assessRisk(riskId: string): RiskAssessment | null {
    const risk = this.risks.get(riskId);
    if (!risk) return null;

    const score = this.calculateScore(risk);
    const level = this.determineLevel(score);

    return {
      risk,
      score,
      level,
      priority: score,
    };
  }

  getAllAssessments(): RiskAssessment[] {
    const assessments: RiskAssessment[] = [];

    for (const risk of this.risks.values()) {
      const assessment = this.assessRisk(risk.id);
      if (assessment) {
        assessments.push(assessment);
      }
    }

    return assessments.sort((a, b) => b.score - a.score);
  }

  getPriorityRisks(minLevel: RiskLevel): Risk[] {
    const levelPriority: Record<RiskLevel, number> = {
      'low': 1,
      'medium': 2,
      'high': 3,
      'critical': 4,
    };

    const minPriority = levelPriority[minLevel];

    return Array.from(this.risks.values()).filter(risk => {
      const score = this.calculateScore(risk);
      const level = this.determineLevel(score);
      return levelPriority[level] >= minPriority;
    });
  }

  updateRiskStatus(riskId: string, status: Risk['status']): boolean {
    const risk = this.risks.get(riskId);
    if (!risk) return false;

    risk.status = status;
    return true;
  }

  getRisksByCategory(category: string): Risk[] {
    return Array.from(this.risks.values()).filter(
      risk => risk.category.toLowerCase() === category.toLowerCase()
    );
  }

  getAverageRiskScore(): number {
    const risks = Array.from(this.risks.values());
    if (risks.length === 0) return 0;

    const totalScore = risks.reduce((sum, risk) => sum + this.calculateScore(risk), 0);
    return totalScore / risks.length;
  }

  generateRiskMatrix(): number[][] {
    const matrix: number[][] = Array(5).fill(null).map(() => Array(5).fill(0));

    for (const risk of this.risks.values()) {
      const row = 5 - risk.likelihood; // Invert for display (5 at top)
      const col = risk.impact - 1;
      matrix[row][col]++;
    }

    return matrix;
  }
}

export { RiskAssessor, Risk, RiskLevel, RiskAssessment };`,
	hint1: `For addRisk, clamp likelihood and impact to 1-5 range using Math.max/min. Generate unique ID and set initial status to 'identified'.`,
	hint2: `For determineLevel, use thresholds: critical >= 16, high >= 10, medium >= 5, low < 5.`,
	testCode: `import { RiskAssessor } from './solution';

// Test1: addRisk creates risk with ID
test('Test1', () => {
  const assessor = new RiskAssessor();
  const risk = assessor.addRisk('SQL Injection', 'Database attack', 4, 5, 'Technical');
  expect(risk.id).toContain('R-');
  expect(risk.name).toBe('SQL Injection');
});

// Test2: calculateScore returns product
test('Test2', () => {
  const assessor = new RiskAssessor();
  const risk = assessor.addRisk('Test', 'Test', 3, 4, 'Test');
  expect(assessor.calculateScore(risk)).toBe(12);
});

// Test3: determineLevel returns correct levels
test('Test3', () => {
  const assessor = new RiskAssessor();
  expect(assessor.determineLevel(25)).toBe('critical');
  expect(assessor.determineLevel(12)).toBe('high');
  expect(assessor.determineLevel(6)).toBe('medium');
  expect(assessor.determineLevel(3)).toBe('low');
});

// Test4: assessRisk returns full assessment
test('Test4', () => {
  const assessor = new RiskAssessor();
  const risk = assessor.addRisk('Test', 'Test', 5, 5, 'Test');
  const assessment = assessor.assessRisk(risk.id);
  expect(assessment?.score).toBe(25);
  expect(assessment?.level).toBe('critical');
});

// Test5: getAllAssessments sorted by score
test('Test5', () => {
  const assessor = new RiskAssessor();
  assessor.addRisk('Low', 'Low', 1, 1, 'Test');
  assessor.addRisk('High', 'High', 5, 5, 'Test');
  const assessments = assessor.getAllAssessments();
  expect(assessments[0].score).toBeGreaterThan(assessments[1].score);
});

// Test6: getPriorityRisks filters by level
test('Test6', () => {
  const assessor = new RiskAssessor();
  assessor.addRisk('Low', 'Low', 1, 1, 'Test');
  assessor.addRisk('Critical', 'Critical', 5, 5, 'Test');
  const high = assessor.getPriorityRisks('high');
  expect(high.length).toBe(1);
  expect(high[0].name).toBe('Critical');
});

// Test7: updateRiskStatus changes status
test('Test7', () => {
  const assessor = new RiskAssessor();
  const risk = assessor.addRisk('Test', 'Test', 3, 3, 'Test');
  assessor.updateRiskStatus(risk.id, 'mitigated');
  const assessment = assessor.assessRisk(risk.id);
  expect(assessment?.risk.status).toBe('mitigated');
});

// Test8: getRisksByCategory filters
test('Test8', () => {
  const assessor = new RiskAssessor();
  assessor.addRisk('R1', 'R1', 3, 3, 'Technical');
  assessor.addRisk('R2', 'R2', 3, 3, 'Operational');
  const technical = assessor.getRisksByCategory('Technical');
  expect(technical.length).toBe(1);
});

// Test9: getAverageRiskScore calculates correctly
test('Test9', () => {
  const assessor = new RiskAssessor();
  assessor.addRisk('R1', 'R1', 2, 2, 'Test'); // Score: 4
  assessor.addRisk('R2', 'R2', 4, 4, 'Test'); // Score: 16
  expect(assessor.getAverageRiskScore()).toBe(10);
});

// Test10: generateRiskMatrix returns 5x5 array
test('Test10', () => {
  const assessor = new RiskAssessor();
  assessor.addRisk('R1', 'R1', 3, 3, 'Test');
  const matrix = assessor.generateRiskMatrix();
  expect(matrix.length).toBe(5);
  expect(matrix[0].length).toBe(5);
});`,
	whyItMatters: `Risk assessment helps organizations focus limited resources on the most critical threats.

**Real-World Application:**

Equifax (2017) failed to patch a known vulnerability for months. A proper risk assessment would have shown:

\`\`\`
Risk: Unpatched Apache Struts vulnerability
Likelihood: 5 (Known exploit, publicly disclosed)
Impact: 5 (Access to sensitive data)
Score: 25 (CRITICAL)

Treatment: Immediate patching required

Actual outcome: 147 million records stolen
Cost: $700+ million in settlements
\`\`\`

**Risk Treatment Options:**

| Strategy | When to Use | Example |
|----------|-------------|---------|
| **Mitigate** | Cost-effective controls exist | Add firewall rule |
| **Transfer** | Insurance can cover loss | Cyber insurance |
| **Avoid** | Risk outweighs benefits | Don't collect SSNs |
| **Accept** | Low risk, high mitigation cost | Minor info disclosure |

**NIST Risk Assessment Process:**

1. **Prepare** - Establish context
2. **Conduct** - Identify and analyze risks
3. **Communicate** - Report findings
4. **Maintain** - Monitor and review`,
	order: 4,
	translations: {
		ru: {
			title: 'Оценка рисков безопасности',
			description: `Научитесь выполнять оценку рисков безопасности по формуле Риск = Вероятность × Воздействие.

**Что такое оценка рисков?**

Оценка рисков - процесс выявления, анализа и оценки рисков. Помогает приоритизировать усилия по безопасности.

**Формула риска:**

Балл риска = Вероятность × Воздействие

**Ваша задача:**

Реализуйте класс \`RiskAssessor\`.`,
			hint1: `Для addRisk ограничьте likelihood и impact диапазоном 1-5 через Math.max/min.`,
			hint2: `Для determineLevel используйте пороги: critical >= 16, high >= 10, medium >= 5, low < 5.`,
			whyItMatters: `Оценка рисков помогает организациям сфокусировать ограниченные ресурсы на самых критичных угрозах.`
		},
		uz: {
			title: 'Xavfsizlik riskini baholash',
			description: `Risk = Ehtimollik × Ta'sir formulasi yordamida xavfsizlik riskini baholashni o'rganing.

**Risk baholash nima?**

Risk baholash - risklarni aniqlash, tahlil qilish va baholash jarayoni.

**Sizning vazifangiz:**

\`RiskAssessor\` klassini amalga oshiring.`,
			hint1: `addRisk uchun likelihood va impact ni Math.max/min orqali 1-5 oralig'iga cheklang.`,
			hint2: `determineLevel uchun chegaralardan foydalaning: critical >= 16, high >= 10, medium >= 5, low < 5.`,
			whyItMatters: `Risk baholash tashkilotlarga cheklangan resurslarni eng muhim tahdidlarga yo'naltirish imkonini beradi.`
		}
	}
};

export default task;
