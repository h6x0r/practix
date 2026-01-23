import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'interview-scenario-analysis',
	title: 'Scenario-Based Security Analysis',
	difficulty: 'medium',
	tags: ['security', 'interview', 'scenarios', 'typescript'],
	estimatedTime: '35m',
	isPremium: true,
	youtubeUrl: '',
	description: `Practice analyzing security scenarios - a key skill for security interviews.

**Scenario Types:**

1. **Breach Response** - What would you do if X happened?
2. **Architecture Review** - Is this design secure?
3. **Policy Decision** - How would you implement this policy?
4. **Risk Assessment** - Evaluate the risks of this change
5. **Investigation** - Analyze these logs for anomalies

**Your Task:**

Implement a \`ScenarioAnalyzer\` class that can evaluate your responses to security scenarios.`,
	initialCode: `interface Scenario {
  id: string;
  type: 'breach' | 'architecture' | 'policy' | 'risk' | 'investigation';
  title: string;
  description: string;
  context: {
    environment: string;
    stakeholders: string[];
    constraints: string[];
    timeline?: string;
  };
  expectedActions: string[];
  redFlags: string[]; // Things that should NOT be done
  optimalOutcome: string;
}

interface AnalysisResponse {
  scenarioId: string;
  immediateActions: string[];
  investigationSteps: string[];
  stakeholderCommunication: string[];
  longTermRemediation: string[];
  lessonsLearned: string[];
}

interface ScenarioEvaluation {
  score: number;
  actionsMatched: string[];
  actionsMissed: string[];
  redFlagsTriggered: string[];
  feedback: {
    strengths: string[];
    improvements: string[];
  };
  interviewerNotes: string;
}

class ScenarioAnalyzer {
  private scenarios: Map<string, Scenario> = new Map();

  // Scenario Management
  addScenario(scenario: Scenario): void {
    // TODO: Add a scenario to the database
  }

  getScenariosByType(type: Scenario['type']): Scenario[] {
    // TODO: Get scenarios by type
    return [];
  }

  getRandomScenario(types?: Scenario['type'][]): Scenario | null {
    // TODO: Get random scenario, optionally filtered by type
    return null;
  }

  // Response Analysis
  analyzeResponse(scenario: Scenario, response: AnalysisResponse): ScenarioEvaluation {
    // TODO: Analyze how well the response addresses the scenario
    return {} as ScenarioEvaluation;
  }

  checkActionCoverage(expectedActions: string[], providedActions: string[]): {
    matched: string[];
    missed: string[];
    coverage: number;
  } {
    // TODO: Check how many expected actions were covered
    return { matched: [], missed: [], coverage: 0 };
  }

  detectRedFlags(response: AnalysisResponse, redFlags: string[]): string[] {
    // TODO: Check if response contains any red flag behaviors
    return [];
  }

  // Interview Simulation
  generateFollowUpQuestions(scenario: Scenario, response: AnalysisResponse): string[] {
    // TODO: Generate follow-up questions based on the response
    return [];
  }

  simulateInterviewerPushback(response: AnalysisResponse): {
    challenge: string;
    suggestedResponse: string;
  }[] {
    // TODO: Simulate challenging questions an interviewer might ask
    return [];
  }

  // Best Practices
  getExpectedFramework(scenarioType: Scenario['type']): {
    phase: string;
    actions: string[];
    considerations: string[];
  }[] {
    // TODO: Return the expected response framework for scenario type
    return [];
  }

  compareToOptimal(response: AnalysisResponse, scenario: Scenario): {
    similarity: number;
    gaps: string[];
    extras: string[];
  } {
    // TODO: Compare response to optimal outcome
    return { similarity: 0, gaps: [], extras: [] };
  }
}

// Test your implementation
const analyzer = new ScenarioAnalyzer();

// Test 1: Add scenario
analyzer.addScenario({
  id: 's1',
  type: 'breach',
  title: 'Ransomware Attack',
  description: 'Your company servers have been encrypted by ransomware demanding 50 BTC.',
  context: {
    environment: 'Healthcare organization with patient data',
    stakeholders: ['CEO', 'Legal', 'IT', 'Patients'],
    constraints: ['HIPAA compliance', '48-hour deadline'],
  },
  expectedActions: ['isolate systems', 'notify stakeholders', 'assess backup status', 'contact law enforcement'],
  redFlags: ['pay ransom immediately', 'hide incident', 'blame employees publicly'],
  optimalOutcome: 'Systems restored from backup, incident reported, security improved',
});
console.log('Test 1 - Scenario added:', analyzer.getScenariosByType('breach').length === 1);

// Test 2: Get random scenario
analyzer.addScenario({
  id: 's2',
  type: 'architecture',
  title: 'API Security Review',
  description: 'Review the security of a new public API design.',
  context: { environment: 'E-commerce', stakeholders: ['Dev', 'Security'], constraints: ['Launch deadline'] },
  expectedActions: ['authentication check', 'rate limiting', 'input validation'],
  redFlags: ['skip security review', 'use HTTP'],
  optimalOutcome: 'Secure API with proper controls',
});
const randomScenario = analyzer.getRandomScenario();
console.log('Test 2 - Random scenario:', randomScenario !== null);

// Test 3: Analyze response
const response: AnalysisResponse = {
  scenarioId: 's1',
  immediateActions: ['Isolate affected systems', 'Notify IT security team'],
  investigationSteps: ['Analyze malware sample', 'Check backup status'],
  stakeholderCommunication: ['Brief CEO', 'Prepare legal notification'],
  longTermRemediation: ['Improve backup procedures', 'Security training'],
  lessonsLearned: ['Need better endpoint protection'],
};
const evaluation = analyzer.analyzeResponse(analyzer.getScenariosByType('breach')[0], response);
console.log('Test 3 - Response analyzed:', evaluation.score >= 0);

// Test 4: Check action coverage
const coverage = analyzer.checkActionCoverage(
  ['isolate systems', 'notify stakeholders', 'assess backup'],
  ['Isolate affected systems', 'Brief stakeholders', 'Check backup status']
);
console.log('Test 4 - Action coverage:', coverage.coverage > 0.5);

// Test 5: Detect red flags
const flags = analyzer.detectRedFlags(response, ['pay ransom', 'hide incident']);
console.log('Test 5 - Red flags check:', flags.length === 0);

// Test 6: Generate follow-up questions
const followUps = analyzer.generateFollowUpQuestions(
  analyzer.getScenariosByType('breach')[0],
  response
);
console.log('Test 6 - Follow-ups generated:', followUps.length > 0);

// Test 7: Simulate pushback
const pushback = analyzer.simulateInterviewerPushback(response);
console.log('Test 7 - Pushback simulated:', pushback.length >= 0);

// Test 8: Get expected framework
const framework = analyzer.getExpectedFramework('breach');
console.log('Test 8 - Framework retrieved:', framework.length > 0);

// Test 9: Compare to optimal
const comparison = analyzer.compareToOptimal(response, analyzer.getScenariosByType('breach')[0]);
console.log('Test 9 - Comparison done:', 'similarity' in comparison);

// Test 10: Filter by type
const archScenarios = analyzer.getScenariosByType('architecture');
console.log('Test 10 - Filter by type:', archScenarios.length === 1);`,
	solutionCode: `interface Scenario {
  id: string;
  type: 'breach' | 'architecture' | 'policy' | 'risk' | 'investigation';
  title: string;
  description: string;
  context: {
    environment: string;
    stakeholders: string[];
    constraints: string[];
    timeline?: string;
  };
  expectedActions: string[];
  redFlags: string[];
  optimalOutcome: string;
}

interface AnalysisResponse {
  scenarioId: string;
  immediateActions: string[];
  investigationSteps: string[];
  stakeholderCommunication: string[];
  longTermRemediation: string[];
  lessonsLearned: string[];
}

interface ScenarioEvaluation {
  score: number;
  actionsMatched: string[];
  actionsMissed: string[];
  redFlagsTriggered: string[];
  feedback: {
    strengths: string[];
    improvements: string[];
  };
  interviewerNotes: string;
}

class ScenarioAnalyzer {
  private scenarios: Map<string, Scenario> = new Map();

  addScenario(scenario: Scenario): void {
    this.scenarios.set(scenario.id, scenario);
  }

  getScenariosByType(type: Scenario['type']): Scenario[] {
    return Array.from(this.scenarios.values()).filter(s => s.type === type);
  }

  getRandomScenario(types?: Scenario['type'][]): Scenario | null {
    let pool = Array.from(this.scenarios.values());

    if (types && types.length > 0) {
      pool = pool.filter(s => types.includes(s.type));
    }

    if (pool.length === 0) return null;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  analyzeResponse(scenario: Scenario, response: AnalysisResponse): ScenarioEvaluation {
    // Collect all provided actions
    const allProvidedActions = [
      ...response.immediateActions,
      ...response.investigationSteps,
      ...response.stakeholderCommunication,
      ...response.longTermRemediation,
      ...response.lessonsLearned,
    ];

    // Check action coverage
    const coverage = this.checkActionCoverage(scenario.expectedActions, allProvidedActions);

    // Check for red flags
    const redFlagsTriggered = this.detectRedFlags(response, scenario.redFlags);

    // Calculate score
    let score = coverage.coverage * 80; // Up to 80 points for coverage

    // Bonus for comprehensiveness
    if (response.immediateActions.length >= 2) score += 5;
    if (response.investigationSteps.length >= 2) score += 5;
    if (response.stakeholderCommunication.length >= 1) score += 5;
    if (response.longTermRemediation.length >= 1) score += 5;

    // Penalty for red flags
    score -= redFlagsTriggered.length * 15;
    score = Math.max(0, Math.min(100, score));

    // Generate feedback
    const strengths: string[] = [];
    const improvements: string[] = [];

    if (coverage.coverage >= 0.7) {
      strengths.push('Good coverage of expected actions');
    }
    if (response.stakeholderCommunication.length > 0) {
      strengths.push('Considered stakeholder communication');
    }
    if (response.lessonsLearned.length > 0) {
      strengths.push('Included lessons learned for improvement');
    }

    if (coverage.missed.length > 0) {
      improvements.push(\`Consider adding: \${coverage.missed.slice(0, 2).join(', ')}\`);
    }
    if (response.immediateActions.length < 2) {
      improvements.push('Expand on immediate actions to take');
    }
    if (redFlagsTriggered.length > 0) {
      improvements.push(\`Avoid: \${redFlagsTriggered.join(', ')}\`);
    }

    // Generate interviewer notes
    let interviewerNotes: string;
    if (score >= 80) {
      interviewerNotes = 'Strong response demonstrating security incident handling expertise.';
    } else if (score >= 60) {
      interviewerNotes = 'Solid foundation but could benefit from more comprehensive approach.';
    } else if (score >= 40) {
      interviewerNotes = 'Basic understanding shown but significant gaps in response framework.';
    } else {
      interviewerNotes = 'Response needs substantial improvement in incident handling methodology.';
    }

    return {
      score: Math.round(score),
      actionsMatched: coverage.matched,
      actionsMissed: coverage.missed,
      redFlagsTriggered,
      feedback: { strengths, improvements },
      interviewerNotes,
    };
  }

  checkActionCoverage(expectedActions: string[], providedActions: string[]): {
    matched: string[];
    missed: string[];
    coverage: number;
  } {
    const matched: string[] = [];
    const missed: string[] = [];

    const providedLower = providedActions.map(a => a.toLowerCase());

    for (const expected of expectedActions) {
      const expectedLower = expected.toLowerCase();
      const words = expectedLower.split(/\s+/).filter(w => w.length > 3);

      // Check if any provided action covers this expected action
      const isMatched = providedLower.some(provided => {
        return words.some(word => provided.includes(word)) ||
          provided.includes(expectedLower);
      });

      if (isMatched) {
        matched.push(expected);
      } else {
        missed.push(expected);
      }
    }

    const coverage = expectedActions.length > 0 ? matched.length / expectedActions.length : 0;

    return { matched, missed, coverage };
  }

  detectRedFlags(response: AnalysisResponse, redFlags: string[]): string[] {
    const triggered: string[] = [];

    const allText = [
      ...response.immediateActions,
      ...response.investigationSteps,
      ...response.stakeholderCommunication,
      ...response.longTermRemediation,
      ...response.lessonsLearned,
    ].join(' ').toLowerCase();

    for (const flag of redFlags) {
      const flagLower = flag.toLowerCase();
      const words = flagLower.split(/\s+/).filter(w => w.length > 3);

      // Check if the response contains red flag keywords
      const isTriggered = words.some(word => allText.includes(word)) ||
        allText.includes(flagLower);

      if (isTriggered) {
        triggered.push(flag);
      }
    }

    return triggered;
  }

  generateFollowUpQuestions(scenario: Scenario, response: AnalysisResponse): string[] {
    const questions: string[] = [];

    // Questions about immediate actions
    if (response.immediateActions.length > 0) {
      questions.push(\`What's your priority order for these immediate actions?\`);
      questions.push(\`How would you handle this if \${scenario.context.constraints[0] || 'resources were limited'}?\`);
    }

    // Questions about stakeholders
    if (scenario.context.stakeholders.length > 0) {
      const stakeholder = scenario.context.stakeholders[0];
      questions.push(\`How would you communicate this to the \${stakeholder}?\`);
    }

    // Questions about investigation
    if (response.investigationSteps.length > 0) {
      questions.push(\`What evidence would you preserve and why?\`);
    }

    // Scenario-specific questions
    if (scenario.type === 'breach') {
      questions.push(\`What if the attacker is still in the system?\`);
      questions.push(\`How would you determine the extent of data exposure?\`);
    } else if (scenario.type === 'architecture') {
      questions.push(\`What trade-offs did you consider in your recommendations?\`);
    }

    // Questions about gaps
    const coverage = this.checkActionCoverage(
      scenario.expectedActions,
      [...response.immediateActions, ...response.investigationSteps]
    );
    if (coverage.missed.length > 0) {
      questions.push(\`What about \${coverage.missed[0]}? Would you consider that?\`);
    }

    return questions.slice(0, 5); // Return top 5 questions
  }

  simulateInterviewerPushback(response: AnalysisResponse): {
    challenge: string;
    suggestedResponse: string;
  }[] {
    const pushbacks: { challenge: string; suggestedResponse: string }[] = [];

    // Challenge immediate actions
    if (response.immediateActions.length > 0) {
      pushbacks.push({
        challenge: 'What if your first action causes more damage?',
        suggestedResponse: 'I would assess the impact before acting. Having a rollback plan and testing in a safe environment when possible.',
      });
    }

    // Challenge stakeholder communication
    if (response.stakeholderCommunication.length > 0) {
      pushbacks.push({
        challenge: 'The CEO wants to keep this quiet. How do you handle that?',
        suggestedResponse: 'I would explain the legal and regulatory requirements for disclosure, and the risks of non-compliance.',
      });
    }

    // Challenge timeline
    pushbacks.push({
      challenge: 'What if you only had 1 hour instead of 24 hours?',
      suggestedResponse: 'I would focus on containment first, then quick assessment of critical assets, while delegating documentation.',
    });

    // Challenge resources
    pushbacks.push({
      challenge: 'Your team is unavailable. What do you do?',
      suggestedResponse: 'I would follow documented incident response procedures, engage external IR support if available, and prioritize the most critical actions I can perform alone.',
    });

    return pushbacks;
  }

  getExpectedFramework(scenarioType: Scenario['type']): {
    phase: string;
    actions: string[];
    considerations: string[];
  }[] {
    const frameworks: Record<Scenario['type'], { phase: string; actions: string[]; considerations: string[] }[]> = {
      breach: [
        {
          phase: 'Preparation',
          actions: ['Activate incident response team', 'Document timeline'],
          considerations: ['Who has decision authority?', 'What are notification requirements?'],
        },
        {
          phase: 'Identification',
          actions: ['Determine scope of breach', 'Identify affected systems'],
          considerations: ['What data was potentially accessed?', 'How did the attacker gain access?'],
        },
        {
          phase: 'Containment',
          actions: ['Isolate affected systems', 'Preserve evidence'],
          considerations: ['Short-term vs long-term containment', 'Business continuity needs'],
        },
        {
          phase: 'Eradication',
          actions: ['Remove malware/threat', 'Patch vulnerabilities'],
          considerations: ['Is the root cause addressed?', 'Are there backdoors?'],
        },
        {
          phase: 'Recovery',
          actions: ['Restore from backup', 'Monitor for recurrence'],
          considerations: ['Validation of system integrity', 'Gradual restoration'],
        },
        {
          phase: 'Lessons Learned',
          actions: ['Post-incident review', 'Update procedures'],
          considerations: ['What could be improved?', 'Were controls adequate?'],
        },
      ],
      architecture: [
        {
          phase: 'Threat Modeling',
          actions: ['Identify assets', 'Map attack surface'],
          considerations: ['What are we protecting?', 'Who are potential attackers?'],
        },
        {
          phase: 'Control Design',
          actions: ['Select appropriate controls', 'Defense in depth'],
          considerations: ['Cost vs benefit', 'Usability impact'],
        },
        {
          phase: 'Review',
          actions: ['Security testing', 'Code review'],
          considerations: ['Coverage of OWASP Top 10', 'Compliance requirements'],
        },
      ],
      policy: [
        {
          phase: 'Assessment',
          actions: ['Identify requirements', 'Stakeholder input'],
          considerations: ['Regulatory requirements', 'Business needs'],
        },
        {
          phase: 'Development',
          actions: ['Draft policy', 'Review cycle'],
          considerations: ['Enforceability', 'Clarity'],
        },
        {
          phase: 'Implementation',
          actions: ['Training', 'Technical controls'],
          considerations: ['User acceptance', 'Monitoring'],
        },
      ],
      risk: [
        {
          phase: 'Identification',
          actions: ['List potential risks', 'Categorize threats'],
          considerations: ['Internal vs external threats', 'Likelihood'],
        },
        {
          phase: 'Analysis',
          actions: ['Assess impact', 'Calculate risk score'],
          considerations: ['Quantitative vs qualitative', 'Risk appetite'],
        },
        {
          phase: 'Mitigation',
          actions: ['Select treatment', 'Implement controls'],
          considerations: ['Accept, mitigate, transfer, avoid', 'Cost effectiveness'],
        },
      ],
      investigation: [
        {
          phase: 'Evidence Collection',
          actions: ['Preserve logs', 'Chain of custody'],
          considerations: ['Legal admissibility', 'Volatility order'],
        },
        {
          phase: 'Analysis',
          actions: ['Timeline reconstruction', 'Indicator analysis'],
          considerations: ['Tools available', 'Scope limitations'],
        },
        {
          phase: 'Reporting',
          actions: ['Document findings', 'Present conclusions'],
          considerations: ['Audience', 'Actionable recommendations'],
        },
      ],
    };

    return frameworks[scenarioType] || [];
  }

  compareToOptimal(response: AnalysisResponse, scenario: Scenario): {
    similarity: number;
    gaps: string[];
    extras: string[];
  } {
    const allProvided = [
      ...response.immediateActions,
      ...response.investigationSteps,
      ...response.stakeholderCommunication,
      ...response.longTermRemediation,
    ];

    // Compare to expected actions
    const coverage = this.checkActionCoverage(scenario.expectedActions, allProvided);

    // Identify extras (actions beyond expected)
    const extras: string[] = [];
    const providedLower = allProvided.map(a => a.toLowerCase());
    const expectedLower = scenario.expectedActions.map(a => a.toLowerCase());

    for (const provided of providedLower) {
      const isExpected = expectedLower.some(expected => {
        const words = expected.split(/\s+/).filter(w => w.length > 3);
        return words.some(w => provided.includes(w));
      });

      if (!isExpected) {
        const original = allProvided[providedLower.indexOf(provided)];
        extras.push(original);
      }
    }

    return {
      similarity: coverage.coverage,
      gaps: coverage.missed,
      extras: extras.slice(0, 5),
    };
  }
}

// Test your implementation
const analyzer = new ScenarioAnalyzer();

// Test 1: Add scenario
analyzer.addScenario({
  id: 's1',
  type: 'breach',
  title: 'Ransomware Attack',
  description: 'Your company servers have been encrypted by ransomware demanding 50 BTC.',
  context: {
    environment: 'Healthcare organization with patient data',
    stakeholders: ['CEO', 'Legal', 'IT', 'Patients'],
    constraints: ['HIPAA compliance', '48-hour deadline'],
  },
  expectedActions: ['isolate systems', 'notify stakeholders', 'assess backup status', 'contact law enforcement'],
  redFlags: ['pay ransom immediately', 'hide incident', 'blame employees publicly'],
  optimalOutcome: 'Systems restored from backup, incident reported, security improved',
});
console.log('Test 1 - Scenario added:', analyzer.getScenariosByType('breach').length === 1);

// Test 2: Get random scenario
analyzer.addScenario({
  id: 's2',
  type: 'architecture',
  title: 'API Security Review',
  description: 'Review the security of a new public API design.',
  context: { environment: 'E-commerce', stakeholders: ['Dev', 'Security'], constraints: ['Launch deadline'] },
  expectedActions: ['authentication check', 'rate limiting', 'input validation'],
  redFlags: ['skip security review', 'use HTTP'],
  optimalOutcome: 'Secure API with proper controls',
});
const randomScenario = analyzer.getRandomScenario();
console.log('Test 2 - Random scenario:', randomScenario !== null);

// Test 3: Analyze response
const response: AnalysisResponse = {
  scenarioId: 's1',
  immediateActions: ['Isolate affected systems', 'Notify IT security team'],
  investigationSteps: ['Analyze malware sample', 'Check backup status'],
  stakeholderCommunication: ['Brief CEO', 'Prepare legal notification'],
  longTermRemediation: ['Improve backup procedures', 'Security training'],
  lessonsLearned: ['Need better endpoint protection'],
};
const evaluation = analyzer.analyzeResponse(analyzer.getScenariosByType('breach')[0], response);
console.log('Test 3 - Response analyzed:', evaluation.score >= 0);

// Test 4: Check action coverage
const coverage = analyzer.checkActionCoverage(
  ['isolate systems', 'notify stakeholders', 'assess backup'],
  ['Isolate affected systems', 'Brief stakeholders', 'Check backup status']
);
console.log('Test 4 - Action coverage:', coverage.coverage > 0.5);

// Test 5: Detect red flags
const flags = analyzer.detectRedFlags(response, ['pay ransom', 'hide incident']);
console.log('Test 5 - Red flags check:', flags.length === 0);

// Test 6: Generate follow-up questions
const followUps = analyzer.generateFollowUpQuestions(
  analyzer.getScenariosByType('breach')[0],
  response
);
console.log('Test 6 - Follow-ups generated:', followUps.length > 0);

// Test 7: Simulate pushback
const pushback = analyzer.simulateInterviewerPushback(response);
console.log('Test 7 - Pushback simulated:', pushback.length >= 0);

// Test 8: Get expected framework
const framework = analyzer.getExpectedFramework('breach');
console.log('Test 8 - Framework retrieved:', framework.length > 0);

// Test 9: Compare to optimal
const comparison = analyzer.compareToOptimal(response, analyzer.getScenariosByType('breach')[0]);
console.log('Test 9 - Comparison done:', 'similarity' in comparison);

// Test 10: Filter by type
const archScenarios = analyzer.getScenariosByType('architecture');
console.log('Test 10 - Filter by type:', archScenarios.length === 1);`,
	testCode: `import { describe, it, expect, beforeEach } from 'vitest';

interface Scenario {
  id: string;
  type: 'breach' | 'architecture' | 'policy' | 'risk' | 'investigation';
  title: string;
  description: string;
  context: { environment: string; stakeholders: string[]; constraints: string[]; timeline?: string };
  expectedActions: string[];
  redFlags: string[];
  optimalOutcome: string;
}

interface AnalysisResponse {
  scenarioId: string;
  immediateActions: string[];
  investigationSteps: string[];
  stakeholderCommunication: string[];
  longTermRemediation: string[];
  lessonsLearned: string[];
}

interface ScenarioEvaluation {
  score: number;
  actionsMatched: string[];
  actionsMissed: string[];
  redFlagsTriggered: string[];
  feedback: { strengths: string[]; improvements: string[] };
  interviewerNotes: string;
}

class ScenarioAnalyzer {
  private scenarios: Map<string, Scenario> = new Map();

  addScenario(scenario: Scenario): void { this.scenarios.set(scenario.id, scenario); }
  getScenariosByType(type: Scenario['type']): Scenario[] { return Array.from(this.scenarios.values()).filter(s => s.type === type); }
  getRandomScenario(types?: Scenario['type'][]): Scenario | null {
    let pool = Array.from(this.scenarios.values());
    if (types && types.length > 0) pool = pool.filter(s => types.includes(s.type));
    if (pool.length === 0) return null;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  analyzeResponse(scenario: Scenario, response: AnalysisResponse): ScenarioEvaluation {
    const allProvided = [...response.immediateActions, ...response.investigationSteps, ...response.stakeholderCommunication, ...response.longTermRemediation, ...response.lessonsLearned];
    const coverage = this.checkActionCoverage(scenario.expectedActions, allProvided);
    const redFlagsTriggered = this.detectRedFlags(response, scenario.redFlags);

    let score = coverage.coverage * 80;
    if (response.immediateActions.length >= 2) score += 5;
    if (response.investigationSteps.length >= 2) score += 5;
    if (response.stakeholderCommunication.length >= 1) score += 5;
    if (response.longTermRemediation.length >= 1) score += 5;
    score -= redFlagsTriggered.length * 15;
    score = Math.max(0, Math.min(100, score));

    const strengths: string[] = coverage.coverage >= 0.7 ? ['Good action coverage'] : [];
    const improvements: string[] = coverage.missed.length > 0 ? [\`Add: \${coverage.missed[0]}\`] : [];

    return {
      score: Math.round(score), actionsMatched: coverage.matched, actionsMissed: coverage.missed,
      redFlagsTriggered, feedback: { strengths, improvements }, interviewerNotes: 'Evaluated'
    };
  }

  checkActionCoverage(expectedActions: string[], providedActions: string[]): { matched: string[]; missed: string[]; coverage: number } {
    const matched: string[] = [], missed: string[] = [];
    const providedLower = providedActions.map(a => a.toLowerCase());
    for (const expected of expectedActions) {
      const words = expected.toLowerCase().split(/\s+/).filter(w => w.length > 3);
      const isMatched = providedLower.some(p => words.some(w => p.includes(w)));
      if (isMatched) matched.push(expected);
      else missed.push(expected);
    }
    return { matched, missed, coverage: expectedActions.length > 0 ? matched.length / expectedActions.length : 0 };
  }

  detectRedFlags(response: AnalysisResponse, redFlags: string[]): string[] {
    const allText = [...response.immediateActions, ...response.investigationSteps, ...response.stakeholderCommunication].join(' ').toLowerCase();
    return redFlags.filter(flag => flag.toLowerCase().split(/\s+/).filter(w => w.length > 3).some(w => allText.includes(w)));
  }

  generateFollowUpQuestions(scenario: Scenario, response: AnalysisResponse): string[] {
    const questions = ['What is your priority order?'];
    if (scenario.context.stakeholders.length > 0) questions.push(\`How would you communicate to \${scenario.context.stakeholders[0]}?\`);
    return questions;
  }

  simulateInterviewerPushback(response: AnalysisResponse): { challenge: string; suggestedResponse: string }[] {
    return [{ challenge: 'What if resources are limited?', suggestedResponse: 'Prioritize critical actions' }];
  }

  getExpectedFramework(scenarioType: Scenario['type']): { phase: string; actions: string[]; considerations: string[] }[] {
    if (scenarioType === 'breach') return [
      { phase: 'Containment', actions: ['Isolate systems'], considerations: ['Business continuity'] },
      { phase: 'Recovery', actions: ['Restore from backup'], considerations: ['Validation'] }
    ];
    return [{ phase: 'Assessment', actions: ['Identify risks'], considerations: ['Impact'] }];
  }

  compareToOptimal(response: AnalysisResponse, scenario: Scenario): { similarity: number; gaps: string[]; extras: string[] } {
    const coverage = this.checkActionCoverage(scenario.expectedActions, [...response.immediateActions, ...response.investigationSteps]);
    return { similarity: coverage.coverage, gaps: coverage.missed, extras: [] };
  }
}

describe('ScenarioAnalyzer', () => {
  let analyzer: ScenarioAnalyzer;

  beforeEach(() => {
    analyzer = new ScenarioAnalyzer();
  });

  it('should add and retrieve scenarios', () => {
    analyzer.addScenario({
      id: 's1', type: 'breach', title: 'Ransomware', description: 'Attack scenario',
      context: { environment: 'Healthcare', stakeholders: ['CEO'], constraints: ['HIPAA'] },
      expectedActions: ['isolate', 'notify'], redFlags: ['pay ransom'], optimalOutcome: 'Recovered'
    });

    expect(analyzer.getScenariosByType('breach')).toHaveLength(1);
    expect(analyzer.getScenariosByType('architecture')).toHaveLength(0);
  });

  it('should get random scenario', () => {
    analyzer.addScenario({ id: 's1', type: 'breach', title: 'Test', description: '', context: { environment: '', stakeholders: [], constraints: [] }, expectedActions: [], redFlags: [], optimalOutcome: '' });
    analyzer.addScenario({ id: 's2', type: 'architecture', title: 'Test', description: '', context: { environment: '', stakeholders: [], constraints: [] }, expectedActions: [], redFlags: [], optimalOutcome: '' });

    const random = analyzer.getRandomScenario();
    expect(random).not.toBeNull();

    const filtered = analyzer.getRandomScenario(['architecture']);
    expect(filtered?.type).toBe('architecture');
  });

  it('should check action coverage', () => {
    const coverage = analyzer.checkActionCoverage(
      ['isolate systems', 'notify stakeholders'],
      ['Isolate the affected systems', 'Send notification to stakeholders']
    );

    expect(coverage.matched.length).toBe(2);
    expect(coverage.coverage).toBe(1);
  });

  it('should detect red flags', () => {
    const response: AnalysisResponse = {
      scenarioId: 's1',
      immediateActions: ['Pay the ransom immediately'],
      investigationSteps: [],
      stakeholderCommunication: [],
      longTermRemediation: [],
      lessonsLearned: []
    };

    const flags = analyzer.detectRedFlags(response, ['pay ransom', 'hide incident']);
    expect(flags).toContain('pay ransom');
  });

  it('should analyze response and calculate score', () => {
    analyzer.addScenario({
      id: 's1', type: 'breach', title: 'Test',
      description: '', context: { environment: '', stakeholders: [], constraints: [] },
      expectedActions: ['isolate', 'notify', 'backup'],
      redFlags: ['pay ransom'], optimalOutcome: ''
    });

    const response: AnalysisResponse = {
      scenarioId: 's1',
      immediateActions: ['Isolate systems', 'Check backup'],
      investigationSteps: ['Analyze logs'],
      stakeholderCommunication: ['Notify management'],
      longTermRemediation: ['Improve security'],
      lessonsLearned: ['Better monitoring']
    };

    const evaluation = analyzer.analyzeResponse(analyzer.getScenariosByType('breach')[0], response);

    expect(evaluation.score).toBeGreaterThan(50);
    expect(evaluation.actionsMatched.length).toBeGreaterThan(0);
  });

  it('should penalize red flag responses', () => {
    analyzer.addScenario({
      id: 's1', type: 'breach', title: 'Test',
      description: '', context: { environment: '', stakeholders: [], constraints: [] },
      expectedActions: ['isolate'], redFlags: ['pay ransom'], optimalOutcome: ''
    });

    const goodResponse: AnalysisResponse = {
      scenarioId: 's1', immediateActions: ['Isolate systems'],
      investigationSteps: [], stakeholderCommunication: [], longTermRemediation: [], lessonsLearned: []
    };

    const badResponse: AnalysisResponse = {
      scenarioId: 's1', immediateActions: ['Pay the ransom'],
      investigationSteps: [], stakeholderCommunication: [], longTermRemediation: [], lessonsLearned: []
    };

    const goodEval = analyzer.analyzeResponse(analyzer.getScenariosByType('breach')[0], goodResponse);
    const badEval = analyzer.analyzeResponse(analyzer.getScenariosByType('breach')[0], badResponse);

    expect(goodEval.score).toBeGreaterThan(badEval.score);
    expect(badEval.redFlagsTriggered.length).toBeGreaterThan(0);
  });

  it('should generate follow-up questions', () => {
    analyzer.addScenario({
      id: 's1', type: 'breach', title: 'Test',
      description: '', context: { environment: '', stakeholders: ['CEO'], constraints: [] },
      expectedActions: [], redFlags: [], optimalOutcome: ''
    });

    const questions = analyzer.generateFollowUpQuestions(
      analyzer.getScenariosByType('breach')[0],
      { scenarioId: 's1', immediateActions: ['action'], investigationSteps: [], stakeholderCommunication: [], longTermRemediation: [], lessonsLearned: [] }
    );

    expect(questions.length).toBeGreaterThan(0);
  });

  it('should provide expected framework', () => {
    const framework = analyzer.getExpectedFramework('breach');

    expect(framework.length).toBeGreaterThan(0);
    expect(framework[0].phase).toBeDefined();
    expect(framework[0].actions.length).toBeGreaterThan(0);
  });

  it('should compare response to optimal', () => {
    analyzer.addScenario({
      id: 's1', type: 'breach', title: 'Test',
      description: '', context: { environment: '', stakeholders: [], constraints: [] },
      expectedActions: ['isolate', 'notify', 'backup'],
      redFlags: [], optimalOutcome: ''
    });

    const comparison = analyzer.compareToOptimal(
      { scenarioId: 's1', immediateActions: ['isolate systems'], investigationSteps: [], stakeholderCommunication: [], longTermRemediation: [], lessonsLearned: [] },
      analyzer.getScenariosByType('breach')[0]
    );

    expect(comparison.similarity).toBeGreaterThan(0);
    expect(comparison.gaps.length).toBeGreaterThan(0);
  });
});`,
	hint1:
		'For action coverage, convert both expected and provided actions to lowercase, then check if key words (longer than 3 characters) from expected actions appear in provided actions.',
	hint2:
		'For red flag detection, search for keywords from each red flag in all response fields. Use case-insensitive matching to catch variations.',
	whyItMatters: `Scenario-based questions test how you apply security knowledge in practice:

**Why Interviewers Use Scenarios:**
- **Real-world application** - Tests practical problem-solving, not just theory
- **Communication skills** - How you explain your thought process
- **Prioritization** - Can you identify what matters most under pressure?
- **Judgment** - Understanding trade-offs and making decisions with incomplete information

**Common Scenario Types:**
- "We discovered a breach. What do you do first?"
- "This architecture design has a deadline. What would you prioritize?"
- "An executive wants to bypass security controls. How do you handle it?"

**Key Success Factors:**
1. **Structure your response** - Use frameworks like PICERL (Preparation, Identification, Containment, Eradication, Recovery, Lessons Learned)
2. **Consider stakeholders** - Security decisions affect business operations
3. **Avoid red flags** - Never suggest hiding incidents, skipping due process, or acting rashly

These scenarios mirror actual security incidents where structured thinking prevents costly mistakes.`,
	order: 2,
	translations: {
		ru: {
			title: 'Анализ сценариев безопасности',
			description: `Практикуйте анализ сценариев безопасности - ключевой навык для собеседований.

**Типы сценариев:**

1. **Реагирование на инцидент** - Что бы вы сделали, если произошло X?
2. **Обзор архитектуры** - Безопасен ли этот дизайн?
3. **Решение по политике** - Как бы вы реализовали эту политику?
4. **Оценка рисков** - Оцените риски этого изменения
5. **Расследование** - Проанализируйте логи на аномалии

**Ваша задача:**

Реализуйте класс \`ScenarioAnalyzer\`, который может оценивать ваши ответы на сценарии безопасности.`,
			hint1:
				'Для покрытия действий переведите ожидаемые и предоставленные действия в нижний регистр, затем проверьте наличие ключевых слов.',
			hint2:
				'Для обнаружения красных флагов ищите ключевые слова каждого флага во всех полях ответа. Используйте регистронезависимое сравнение.',
			whyItMatters: `Сценарные вопросы проверяют применение знаний на практике:

**Почему интервьюеры используют сценарии:**
- **Практическое применение** - Тестирует решение проблем, а не только теорию
- **Коммуникативные навыки** - Как вы объясняете ход мыслей
- **Приоритизация** - Можете ли вы определить главное под давлением?
- **Суждение** - Понимание компромиссов при неполной информации

**Ключевые факторы успеха:**
1. **Структурируйте ответ** - Используйте фреймворки как PICERL
2. **Учитывайте заинтересованные стороны** - Решения по безопасности влияют на бизнес
3. **Избегайте красных флагов** - Никогда не предлагайте скрывать инциденты`,
		},
		uz: {
			title: 'Xavfsizlik stsenariylarini tahlil qilish',
			description: `Xavfsizlik stsenariylarini tahlil qilishni mashq qiling - intervyular uchun asosiy ko'nikma.

**Stsenariy turlari:**

1. **Hodisaga javob** - Agar X sodir bo'lsa, nima qilardingiz?
2. **Arxitektura ko'rigi** - Bu dizayn xavfsizmi?
3. **Siyosat qarori** - Bu siyosatni qanday amalga oshirardingiz?
4. **Xavfni baholash** - Bu o'zgarishning xavflarini baholang
5. **Tekshiruv** - Loglarni anomaliyalar uchun tahlil qiling

**Vazifangiz:**

Xavfsizlik stsenariylariga javoblaringizni baholay oladigan \`ScenarioAnalyzer\` klassini yarating.`,
			hint1:
				"Harakatlar qoplanishini tekshirish uchun kutilgan va taqdim etilgan harakatlarni kichik harflarga o'tkazing, so'ngra kalit so'zlarni tekshiring.",
			hint2:
				"Qizil bayroqlarni aniqlash uchun har bir bayroqning kalit so'zlarini barcha javob maydonlarida qidiring. Registrga sezgir bo'lmagan solishtirish ishlating.",
			whyItMatters: `Stsenariy asosidagi savollar bilimlarni amalda qo'llashni tekshiradi:

**Nima uchun intervyuerlar stsenariylardan foydalanadi:**
- **Amaliy qo'llash** - Faqat nazariya emas, amaliy muammolarni hal qilishni tekshiradi
- **Kommunikatsiya ko'nikmalari** - Fikrlash jarayonini qanday tushuntirasiz
- **Ustuvorlash** - Bosim ostida eng muhimni aniqlay olasizmi?
- **Baholash** - To'liq bo'lmagan ma'lumot bilan murosakorlikni tushunish

**Muvaffaqiyat omillari:**
1. **Javobni tuzilmalang** - PICERL kabi freymvorklardan foydalaning
2. **Manfaatdor tomonlarni hisobga oling** - Xavfsizlik qarorlari biznesga ta'sir qiladi
3. **Qizil bayroqlardan qoching** - Hech qachon hodisalarni yashirishni taklif qilmang`,
		},
	},
};

export default task;
