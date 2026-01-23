import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'interview-incident-response',
	title: 'Incident Response Scenarios',
	difficulty: 'hard',
	tags: ['security', 'interview', 'incident-response', 'ir', 'typescript'],
	estimatedTime: '40m',
	isPremium: true,
	youtubeUrl: '',
	description: `Practice incident response scenarios - essential for security operations interviews.

**IR Phases (NIST Framework):**

1. **Preparation** - Policies, procedures, training
2. **Detection & Analysis** - Identify and understand incidents
3. **Containment** - Stop the spread
4. **Eradication** - Remove the threat
5. **Recovery** - Restore operations
6. **Lessons Learned** - Improve for next time

**Your Task:**

Implement an \`IncidentResponseSimulator\` class that walks through incident handling methodology.`,
	initialCode: `interface Incident {
  id: string;
  type: 'malware' | 'phishing' | 'data_breach' | 'ddos' | 'insider_threat' | 'ransomware';
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'detected' | 'analyzing' | 'contained' | 'eradicating' | 'recovering' | 'closed';
  detectedAt: Date;
  affectedSystems: string[];
  indicators: string[];
  timeline: TimelineEntry[];
}

interface TimelineEntry {
  timestamp: Date;
  phase: 'detection' | 'analysis' | 'containment' | 'eradication' | 'recovery' | 'lessons_learned';
  action: string;
  actor: string;
  evidence?: string;
}

interface ContainmentStrategy {
  type: 'isolation' | 'shutdown' | 'block' | 'credential_reset';
  target: string;
  rationale: string;
  reversible: boolean;
  businessImpact: 'none' | 'low' | 'medium' | 'high';
}

interface IncidentReport {
  incidentId: string;
  executiveSummary: string;
  timeline: TimelineEntry[];
  rootCause: string;
  impactAssessment: {
    systemsAffected: number;
    dataExposed: boolean;
    downtime: string;
    estimatedCost: string;
  };
  lessonsLearned: string[];
  recommendations: string[];
}

class IncidentResponseSimulator {
  private incidents: Map<string, Incident> = new Map();

  // Incident Lifecycle
  createIncident(type: Incident['type'], indicators: string[], affectedSystems: string[]): Incident {
    // TODO: Create a new incident
    return {} as Incident;
  }

  updateStatus(incidentId: string, newStatus: Incident['status'], action: string): void {
    // TODO: Update incident status and add to timeline
  }

  addTimelineEntry(incidentId: string, entry: Omit<TimelineEntry, 'timestamp'>): void {
    // TODO: Add an entry to the incident timeline
  }

  // Detection & Analysis
  classifySeverity(type: Incident['type'], indicators: string[], systemCount: number): Incident['severity'] {
    // TODO: Classify incident severity
    return 'low';
  }

  analyzeIndicators(indicators: string[]): {
    iocType: string;
    confidence: 'low' | 'medium' | 'high';
    relatedThreats: string[];
  }[] {
    // TODO: Analyze indicators of compromise
    return [];
  }

  // Containment
  recommendContainmentStrategies(incident: Incident): ContainmentStrategy[] {
    // TODO: Recommend containment strategies based on incident type
    return [];
  }

  assessBusinessImpact(strategy: ContainmentStrategy, affectedSystems: string[]): {
    impact: string;
    alternatives: string[];
    recommendation: string;
  } {
    // TODO: Assess business impact of containment
    return {} as any;
  }

  // Communication
  generateStatusUpdate(incidentId: string, audience: 'technical' | 'executive' | 'legal'): string {
    // TODO: Generate status update for different audiences
    return '';
  }

  determineNotificationRequirements(incident: Incident): {
    regulatory: string[];
    internal: string[];
    external: string[];
    timeline: string;
  } {
    // TODO: Determine who needs to be notified
    return {} as any;
  }

  // Reporting
  generateIncidentReport(incidentId: string): IncidentReport {
    // TODO: Generate final incident report
    return {} as IncidentReport;
  }

  // Interview Helpers
  getInterviewScenario(difficulty: 'junior' | 'senior' | 'lead'): {
    scenario: string;
    expectedActions: string[];
    commonMistakes: string[];
  } {
    // TODO: Generate interview scenario
    return {} as any;
  }

  evaluateResponse(actions: string[], expectedActions: string[]): {
    score: number;
    correct: string[];
    missed: string[];
    extras: string[];
  } {
    // TODO: Evaluate candidate's response
    return {} as any;
  }
}

// Test your implementation
const simulator = new IncidentResponseSimulator();

// Test 1: Create incident
const incident = simulator.createIncident('ransomware', ['encrypted files', 'ransom note'], ['server-001', 'server-002']);
console.log('Test 1 - Incident created:', incident.type === 'ransomware');

// Test 2: Classify severity
const severity = simulator.classifySeverity('ransomware', ['encrypted files'], 10);
console.log('Test 2 - Severity classified:', severity === 'critical');

// Test 3: Update status
simulator.updateStatus(incident.id, 'analyzing', 'Initiated forensic analysis');
console.log('Test 3 - Status updated:', true);

// Test 4: Analyze indicators
const analysis = simulator.analyzeIndicators(['192.168.1.100', '5d41402abc4b2a76b9719d911017c592']);
console.log('Test 4 - Indicators analyzed:', analysis.length > 0);

// Test 5: Recommend containment
const containment = simulator.recommendContainmentStrategies(incident);
console.log('Test 5 - Containment recommended:', containment.length > 0);

// Test 6: Assess business impact
const impact = simulator.assessBusinessImpact(containment[0], incident.affectedSystems);
console.log('Test 6 - Impact assessed:', 'recommendation' in impact);

// Test 7: Generate status update
const techUpdate = simulator.generateStatusUpdate(incident.id, 'technical');
const execUpdate = simulator.generateStatusUpdate(incident.id, 'executive');
console.log('Test 7 - Status updates:', techUpdate.length > 0 && execUpdate.length > 0);

// Test 8: Notification requirements
const notifications = simulator.determineNotificationRequirements(incident);
console.log('Test 8 - Notifications:', notifications.regulatory?.length >= 0);

// Test 9: Generate report
const report = simulator.generateIncidentReport(incident.id);
console.log('Test 9 - Report generated:', 'executiveSummary' in report);

// Test 10: Interview scenario
const scenario = simulator.getInterviewScenario('senior');
console.log('Test 10 - Scenario generated:', scenario.expectedActions?.length > 0);`,
	solutionCode: `interface Incident {
  id: string;
  type: 'malware' | 'phishing' | 'data_breach' | 'ddos' | 'insider_threat' | 'ransomware';
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'detected' | 'analyzing' | 'contained' | 'eradicating' | 'recovering' | 'closed';
  detectedAt: Date;
  affectedSystems: string[];
  indicators: string[];
  timeline: TimelineEntry[];
}

interface TimelineEntry {
  timestamp: Date;
  phase: 'detection' | 'analysis' | 'containment' | 'eradication' | 'recovery' | 'lessons_learned';
  action: string;
  actor: string;
  evidence?: string;
}

interface ContainmentStrategy {
  type: 'isolation' | 'shutdown' | 'block' | 'credential_reset';
  target: string;
  rationale: string;
  reversible: boolean;
  businessImpact: 'none' | 'low' | 'medium' | 'high';
}

interface IncidentReport {
  incidentId: string;
  executiveSummary: string;
  timeline: TimelineEntry[];
  rootCause: string;
  impactAssessment: {
    systemsAffected: number;
    dataExposed: boolean;
    downtime: string;
    estimatedCost: string;
  };
  lessonsLearned: string[];
  recommendations: string[];
}

class IncidentResponseSimulator {
  private incidents: Map<string, Incident> = new Map();

  createIncident(type: Incident['type'], indicators: string[], affectedSystems: string[]): Incident {
    const severity = this.classifySeverity(type, indicators, affectedSystems.length);

    const incident: Incident = {
      id: \`INC-\${Date.now()}-\${Math.random().toString(36).substring(2, 6)}\`,
      type,
      severity,
      status: 'detected',
      detectedAt: new Date(),
      affectedSystems,
      indicators,
      timeline: [
        {
          timestamp: new Date(),
          phase: 'detection',
          action: \`Incident detected: \${type} affecting \${affectedSystems.length} system(s)\`,
          actor: 'System',
          evidence: indicators.join(', '),
        },
      ],
    };

    this.incidents.set(incident.id, incident);
    return incident;
  }

  updateStatus(incidentId: string, newStatus: Incident['status'], action: string): void {
    const incident = this.incidents.get(incidentId);
    if (!incident) return;

    const phaseMap: Record<Incident['status'], TimelineEntry['phase']> = {
      detected: 'detection',
      analyzing: 'analysis',
      contained: 'containment',
      eradicating: 'eradication',
      recovering: 'recovery',
      closed: 'lessons_learned',
    };

    incident.status = newStatus;
    incident.timeline.push({
      timestamp: new Date(),
      phase: phaseMap[newStatus],
      action,
      actor: 'IR Team',
    });
  }

  addTimelineEntry(incidentId: string, entry: Omit<TimelineEntry, 'timestamp'>): void {
    const incident = this.incidents.get(incidentId);
    if (!incident) return;

    incident.timeline.push({
      ...entry,
      timestamp: new Date(),
    });
  }

  classifySeverity(type: Incident['type'], indicators: string[], systemCount: number): Incident['severity'] {
    // Base severity by incident type
    const baseSeverity: Record<Incident['type'], number> = {
      ransomware: 4,
      data_breach: 4,
      insider_threat: 3,
      malware: 2,
      phishing: 2,
      ddos: 2,
    };

    let severityScore = baseSeverity[type];

    // Adjust for scale
    if (systemCount > 10) severityScore += 1;
    if (systemCount > 50) severityScore += 1;

    // Adjust for indicators
    const criticalIndicators = ['encrypted', 'ransom', 'exfiltration', 'c2', 'lateral'];
    if (indicators.some(i => criticalIndicators.some(c => i.toLowerCase().includes(c)))) {
      severityScore += 1;
    }

    if (severityScore >= 4) return 'critical';
    if (severityScore >= 3) return 'high';
    if (severityScore >= 2) return 'medium';
    return 'low';
  }

  analyzeIndicators(indicators: string[]): {
    iocType: string;
    confidence: 'low' | 'medium' | 'high';
    relatedThreats: string[];
  }[] {
    const results: { iocType: string; confidence: 'low' | 'medium' | 'high'; relatedThreats: string[] }[] = [];

    for (const indicator of indicators) {
      // IP address pattern
      if (/^(\d{1,3}\.){3}\d{1,3}$/.test(indicator)) {
        results.push({
          iocType: 'IP Address',
          confidence: 'medium',
          relatedThreats: ['C2 Server', 'Scanning', 'Brute Force'],
        });
      }
      // Hash pattern (MD5)
      else if (/^[a-fA-F0-9]{32}$/.test(indicator)) {
        results.push({
          iocType: 'File Hash (MD5)',
          confidence: 'high',
          relatedThreats: ['Malware', 'Trojan', 'Backdoor'],
        });
      }
      // Domain pattern
      else if (/^[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}$/.test(indicator)) {
        results.push({
          iocType: 'Domain',
          confidence: 'medium',
          relatedThreats: ['Phishing', 'C2 Infrastructure', 'Malware Distribution'],
        });
      }
      // Ransomware indicators
      else if (indicator.toLowerCase().includes('encrypted') || indicator.toLowerCase().includes('ransom')) {
        results.push({
          iocType: 'Behavioral',
          confidence: 'high',
          relatedThreats: ['Ransomware', 'Data Encryption'],
        });
      }
      // Generic
      else {
        results.push({
          iocType: 'Unknown',
          confidence: 'low',
          relatedThreats: ['Requires Further Analysis'],
        });
      }
    }

    return results;
  }

  recommendContainmentStrategies(incident: Incident): ContainmentStrategy[] {
    const strategies: ContainmentStrategy[] = [];

    // Strategies by incident type
    switch (incident.type) {
      case 'ransomware':
        strategies.push({
          type: 'isolation',
          target: 'Affected systems',
          rationale: 'Prevent ransomware from spreading to other systems',
          reversible: true,
          businessImpact: 'high',
        });
        strategies.push({
          type: 'shutdown',
          target: 'File shares',
          rationale: 'Protect shared data from encryption',
          reversible: true,
          businessImpact: 'medium',
        });
        break;

      case 'data_breach':
        strategies.push({
          type: 'credential_reset',
          target: 'Compromised accounts',
          rationale: 'Revoke attacker access',
          reversible: false,
          businessImpact: 'low',
        });
        strategies.push({
          type: 'block',
          target: 'Exfiltration destinations',
          rationale: 'Stop ongoing data theft',
          reversible: true,
          businessImpact: 'none',
        });
        break;

      case 'malware':
        strategies.push({
          type: 'isolation',
          target: 'Infected hosts',
          rationale: 'Prevent lateral movement',
          reversible: true,
          businessImpact: 'medium',
        });
        break;

      case 'phishing':
        strategies.push({
          type: 'credential_reset',
          target: 'Affected users',
          rationale: 'Prevent credential abuse',
          reversible: false,
          businessImpact: 'low',
        });
        strategies.push({
          type: 'block',
          target: 'Phishing URLs',
          rationale: 'Prevent further victims',
          reversible: true,
          businessImpact: 'none',
        });
        break;

      case 'ddos':
        strategies.push({
          type: 'block',
          target: 'Attack sources',
          rationale: 'Filter malicious traffic',
          reversible: true,
          businessImpact: 'none',
        });
        break;

      case 'insider_threat':
        strategies.push({
          type: 'credential_reset',
          target: 'Suspect account',
          rationale: 'Revoke access immediately',
          reversible: false,
          businessImpact: 'low',
        });
        strategies.push({
          type: 'isolation',
          target: 'Suspect workstation',
          rationale: 'Preserve evidence and prevent further actions',
          reversible: true,
          businessImpact: 'low',
        });
        break;
    }

    // Add common strategy
    if (incident.severity === 'critical') {
      strategies.unshift({
        type: 'isolation',
        target: 'Critical systems at risk',
        rationale: 'Protect high-value assets',
        reversible: true,
        businessImpact: 'high',
      });
    }

    return strategies;
  }

  assessBusinessImpact(strategy: ContainmentStrategy, affectedSystems: string[]): {
    impact: string;
    alternatives: string[];
    recommendation: string;
  } {
    let impact: string;
    const alternatives: string[] = [];
    let recommendation: string;

    switch (strategy.businessImpact) {
      case 'high':
        impact = \`High business impact: \${affectedSystems.length} systems will be unavailable\`;
        alternatives.push('Implement in phases during off-hours');
        alternatives.push('Set up temporary workarounds before containment');
        recommendation = 'Proceed with executive approval and communication plan';
        break;

      case 'medium':
        impact = \`Medium business impact: Some services will be degraded\`;
        alternatives.push('Implement with monitoring');
        alternatives.push('Prepare rollback procedure');
        recommendation = 'Proceed with stakeholder notification';
        break;

      case 'low':
        impact = \`Low business impact: Minimal disruption expected\`;
        alternatives.push('Can implement immediately');
        recommendation = 'Proceed immediately';
        break;

      default:
        impact = \`No significant business impact expected\`;
        recommendation = 'Implement immediately';
    }

    return { impact, alternatives, recommendation };
  }

  generateStatusUpdate(incidentId: string, audience: 'technical' | 'executive' | 'legal'): string {
    const incident = this.incidents.get(incidentId);
    if (!incident) return 'Incident not found';

    const latestEntry = incident.timeline[incident.timeline.length - 1];
    const duration = Math.round((Date.now() - incident.detectedAt.getTime()) / (1000 * 60));

    switch (audience) {
      case 'technical':
        return \`[INC-\${incident.id}] Status: \${incident.status.toUpperCase()}
Type: \${incident.type} | Severity: \${incident.severity}
Systems Affected: \${incident.affectedSystems.join(', ')}
IOCs: \${incident.indicators.join(', ')}
Latest Action: \${latestEntry.action}
Duration: \${duration} minutes\`;

      case 'executive':
        return \`Security Incident Update

Current Status: \${incident.status.charAt(0).toUpperCase() + incident.status.slice(1)}
Severity Level: \${incident.severity.toUpperCase()}
Systems Impacted: \${incident.affectedSystems.length}
Time Since Detection: \${duration} minutes

Summary: We are actively responding to a \${incident.type} incident. \${this.getExecutiveSummaryAction(incident)}

Next Update: 30 minutes\`;

      case 'legal':
        return \`PRIVILEGED AND CONFIDENTIAL - LEGAL HOLD

Incident ID: \${incident.id}
Classification: \${incident.type}
Detection Time: \${incident.detectedAt.toISOString()}
Status: \${incident.status}

Potential Data Impact: \${incident.type === 'data_breach' || incident.type === 'ransomware' ? 'Yes - Under Investigation' : 'Under Assessment'}

Timeline entries have been preserved. Full forensic report to follow.
Notification requirements being assessed.\`;
    }
  }

  private getExecutiveSummaryAction(incident: Incident): string {
    switch (incident.status) {
      case 'detected':
        return 'Our team has identified the incident and is beginning analysis.';
      case 'analyzing':
        return 'We are determining the scope and impact of the incident.';
      case 'contained':
        return 'The threat has been contained and we are preventing further spread.';
      case 'eradicating':
        return 'We are removing the threat from affected systems.';
      case 'recovering':
        return 'Systems are being restored to normal operations.';
      case 'closed':
        return 'The incident has been resolved. A full report will follow.';
    }
  }

  determineNotificationRequirements(incident: Incident): {
    regulatory: string[];
    internal: string[];
    external: string[];
    timeline: string;
  } {
    const regulatory: string[] = [];
    const internal: string[] = [];
    const external: string[] = [];
    let timeline = '72 hours';

    // Internal notifications
    internal.push('CISO / Security Leadership');
    internal.push('IT Operations');

    if (incident.severity === 'critical' || incident.severity === 'high') {
      internal.push('Executive Leadership');
      internal.push('Legal Counsel');
    }

    // Regulatory notifications
    if (incident.type === 'data_breach') {
      regulatory.push('GDPR Supervisory Authority (if EU data affected)');
      regulatory.push('State Attorney General (if US PII affected)');
      regulatory.push('HHS (if healthcare data affected - HIPAA)');
      timeline = '72 hours (GDPR), varies by state';
    }

    if (incident.type === 'ransomware') {
      regulatory.push('FBI / CISA (recommended)');
      regulatory.push('Cyber insurance carrier');
    }

    // External notifications
    if (incident.type === 'data_breach') {
      external.push('Affected customers/individuals');
      external.push('Credit monitoring services');
    }

    if (incident.severity === 'critical') {
      external.push('PR/Communications team');
      external.push('Third-party IR support (if needed)');
    }

    return { regulatory, internal, external, timeline };
  }

  generateIncidentReport(incidentId: string): IncidentReport {
    const incident = this.incidents.get(incidentId);
    if (!incident) {
      return {
        incidentId,
        executiveSummary: 'Incident not found',
        timeline: [],
        rootCause: 'Unknown',
        impactAssessment: { systemsAffected: 0, dataExposed: false, downtime: 'N/A', estimatedCost: 'N/A' },
        lessonsLearned: [],
        recommendations: [],
      };
    }

    const dataExposed = incident.type === 'data_breach' || incident.type === 'ransomware';
    const downtime = incident.timeline.length > 1
      ? \`\${Math.round((incident.timeline[incident.timeline.length - 1].timestamp.getTime() - incident.detectedAt.getTime()) / (1000 * 60 * 60))} hours\`
      : 'Ongoing';

    const rootCauseMap: Record<Incident['type'], string> = {
      ransomware: 'Initial access via phishing email or exposed RDP. Investigation ongoing.',
      data_breach: 'Unauthorized access to sensitive data. Root cause analysis in progress.',
      malware: 'Malicious software execution. Entry point being determined.',
      phishing: 'User clicked malicious link or attachment.',
      ddos: 'Distributed attack from botnet targeting public-facing services.',
      insider_threat: 'Unauthorized actions by privileged user.',
    };

    const lessonsLearned = [
      'Detection time should be improved with enhanced monitoring',
      'Incident response procedures should be updated based on this experience',
      'User awareness training should address the attack vector used',
    ];

    const recommendations = [
      'Implement or enhance EDR solution',
      'Review and update access controls',
      'Conduct tabletop exercises quarterly',
      'Update incident response playbooks',
    ];

    if (incident.type === 'ransomware') {
      recommendations.push('Verify backup integrity and test restoration');
    }

    return {
      incidentId,
      executiveSummary: \`On \${incident.detectedAt.toLocaleDateString()}, a \${incident.severity} severity \${incident.type} incident was detected affecting \${incident.affectedSystems.length} system(s). The incident response team followed established procedures to contain and remediate the threat.\`,
      timeline: incident.timeline,
      rootCause: rootCauseMap[incident.type],
      impactAssessment: {
        systemsAffected: incident.affectedSystems.length,
        dataExposed,
        downtime,
        estimatedCost: incident.severity === 'critical' ? '$100,000+' : incident.severity === 'high' ? '$50,000+' : '$10,000+',
      },
      lessonsLearned,
      recommendations,
    };
  }

  getInterviewScenario(difficulty: 'junior' | 'senior' | 'lead'): {
    scenario: string;
    expectedActions: string[];
    commonMistakes: string[];
  } {
    const scenarios = {
      junior: {
        scenario: 'A user reports their computer is running slowly and they received a strange email earlier. What do you do?',
        expectedActions: [
          'Isolate the workstation from the network',
          'Interview the user about the email and their actions',
          'Check for indicators of compromise',
          'Report to senior team members',
          'Document findings',
        ],
        commonMistakes: [
          'Immediately wiping the machine (destroys evidence)',
          'Not isolating the system first',
          'Dismissing user concerns without investigation',
          'Forgetting to document actions taken',
        ],
      },
      senior: {
        scenario: 'Multiple systems are showing signs of ransomware. Files are being encrypted across the network. What is your response?',
        expectedActions: [
          'Declare an incident and activate IR team',
          'Immediately isolate affected systems and disable network shares',
          'Identify patient zero and attack vector',
          'Assess backup status and integrity',
          'Communicate with leadership and prepare for potential disclosure',
          'Begin evidence preservation for potential law enforcement',
        ],
        commonMistakes: [
          'Paying the ransom without proper consideration',
          'Not preserving evidence before remediation',
          'Failing to check if backups are also compromised',
          'Not communicating with stakeholders',
          'Rushing to restore without understanding the full scope',
        ],
      },
      lead: {
        scenario: 'You discover evidence of long-term unauthorized access (APT) with data exfiltration. The attacker may still be in the network. How do you proceed?',
        expectedActions: [
          'Engage executive leadership and legal immediately',
          'Avoid alerting the attacker (no obvious containment)',
          'Begin covert monitoring to understand scope',
          'Prepare for simultaneous enterprise-wide containment',
          'Engage external IR firm and consider law enforcement',
          'Plan for public disclosure and regulatory notification',
          'Coordinate with HR if insider involvement suspected',
        ],
        commonMistakes: [
          'Alerting the attacker through obvious containment',
          'Not involving legal counsel early',
          'Underestimating the scope of compromise',
          'Failing to plan for coordinated response',
          'Not considering regulatory notification requirements',
          'Attempting to handle APT without specialized help',
        ],
      },
    };

    return scenarios[difficulty];
  }

  evaluateResponse(actions: string[], expectedActions: string[]): {
    score: number;
    correct: string[];
    missed: string[];
    extras: string[];
  } {
    const correct: string[] = [];
    const missed: string[] = [];
    const extras: string[] = [];

    const actionsLower = actions.map(a => a.toLowerCase());
    const expectedLower = expectedActions.map(a => a.toLowerCase());

    for (const expected of expectedActions) {
      const expectedWords = expected.toLowerCase().split(/\s+/).filter(w => w.length > 3);
      const isMatched = actionsLower.some(action =>
        expectedWords.filter(word => action.includes(word)).length >= Math.ceil(expectedWords.length / 2)
      );

      if (isMatched) {
        correct.push(expected);
      } else {
        missed.push(expected);
      }
    }

    for (const action of actions) {
      const actionWords = action.toLowerCase().split(/\s+/).filter(w => w.length > 3);
      const isExpected = expectedLower.some(expected =>
        actionWords.some(word => expected.includes(word))
      );

      if (!isExpected && action.trim().length > 0) {
        extras.push(action);
      }
    }

    const score = expectedActions.length > 0
      ? Math.round((correct.length / expectedActions.length) * 100)
      : 0;

    return { score, correct, missed, extras };
  }
}

// Test your implementation
const simulator = new IncidentResponseSimulator();

// Test 1: Create incident
const incident = simulator.createIncident('ransomware', ['encrypted files', 'ransom note'], ['server-001', 'server-002']);
console.log('Test 1 - Incident created:', incident.type === 'ransomware');

// Test 2: Classify severity
const severity = simulator.classifySeverity('ransomware', ['encrypted files'], 10);
console.log('Test 2 - Severity classified:', severity === 'critical');

// Test 3: Update status
simulator.updateStatus(incident.id, 'analyzing', 'Initiated forensic analysis');
console.log('Test 3 - Status updated:', true);

// Test 4: Analyze indicators
const analysis = simulator.analyzeIndicators(['192.168.1.100', '5d41402abc4b2a76b9719d911017c592']);
console.log('Test 4 - Indicators analyzed:', analysis.length > 0);

// Test 5: Recommend containment
const containment = simulator.recommendContainmentStrategies(incident);
console.log('Test 5 - Containment recommended:', containment.length > 0);

// Test 6: Assess business impact
const impact = simulator.assessBusinessImpact(containment[0], incident.affectedSystems);
console.log('Test 6 - Impact assessed:', 'recommendation' in impact);

// Test 7: Generate status update
const techUpdate = simulator.generateStatusUpdate(incident.id, 'technical');
const execUpdate = simulator.generateStatusUpdate(incident.id, 'executive');
console.log('Test 7 - Status updates:', techUpdate.length > 0 && execUpdate.length > 0);

// Test 8: Notification requirements
const notifications = simulator.determineNotificationRequirements(incident);
console.log('Test 8 - Notifications:', notifications.regulatory?.length >= 0);

// Test 9: Generate report
const report = simulator.generateIncidentReport(incident.id);
console.log('Test 9 - Report generated:', 'executiveSummary' in report);

// Test 10: Interview scenario
const scenario = simulator.getInterviewScenario('senior');
console.log('Test 10 - Scenario generated:', scenario.expectedActions?.length > 0);`,
	testCode: `import { describe, it, expect, beforeEach } from 'vitest';

interface Incident {
  id: string;
  type: 'malware' | 'phishing' | 'data_breach' | 'ddos' | 'insider_threat' | 'ransomware';
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'detected' | 'analyzing' | 'contained' | 'eradicating' | 'recovering' | 'closed';
  detectedAt: Date;
  affectedSystems: string[];
  indicators: string[];
  timeline: { timestamp: Date; phase: string; action: string; actor: string; evidence?: string }[];
}

interface ContainmentStrategy {
  type: 'isolation' | 'shutdown' | 'block' | 'credential_reset';
  target: string;
  rationale: string;
  reversible: boolean;
  businessImpact: 'none' | 'low' | 'medium' | 'high';
}

class IncidentResponseSimulator {
  private incidents: Map<string, Incident> = new Map();

  createIncident(type: Incident['type'], indicators: string[], affectedSystems: string[]): Incident {
    const severity = this.classifySeverity(type, indicators, affectedSystems.length);
    const incident: Incident = {
      id: \`INC-\${Date.now()}\`, type, severity, status: 'detected', detectedAt: new Date(),
      affectedSystems, indicators, timeline: [{ timestamp: new Date(), phase: 'detection', action: 'Incident detected', actor: 'System' }],
    };
    this.incidents.set(incident.id, incident);
    return incident;
  }

  updateStatus(incidentId: string, newStatus: Incident['status'], action: string): void {
    const incident = this.incidents.get(incidentId);
    if (!incident) return;
    incident.status = newStatus;
    incident.timeline.push({ timestamp: new Date(), phase: newStatus, action, actor: 'IR Team' });
  }

  classifySeverity(type: Incident['type'], indicators: string[], systemCount: number): Incident['severity'] {
    const base: Record<Incident['type'], number> = { ransomware: 4, data_breach: 4, insider_threat: 3, malware: 2, phishing: 2, ddos: 2 };
    let score = base[type];
    if (systemCount > 10) score += 1;
    if (indicators.some(i => i.toLowerCase().includes('encrypted'))) score += 1;
    if (score >= 4) return 'critical';
    if (score >= 3) return 'high';
    if (score >= 2) return 'medium';
    return 'low';
  }

  analyzeIndicators(indicators: string[]): { iocType: string; confidence: 'low' | 'medium' | 'high'; relatedThreats: string[] }[] {
    return indicators.map(i => {
      if (/^(\d{1,3}\.){3}\d{1,3}$/.test(i)) return { iocType: 'IP Address', confidence: 'medium' as const, relatedThreats: ['C2'] };
      if (/^[a-fA-F0-9]{32}$/.test(i)) return { iocType: 'File Hash', confidence: 'high' as const, relatedThreats: ['Malware'] };
      return { iocType: 'Unknown', confidence: 'low' as const, relatedThreats: ['Needs analysis'] };
    });
  }

  recommendContainmentStrategies(incident: Incident): ContainmentStrategy[] {
    const strategies: ContainmentStrategy[] = [];
    if (incident.type === 'ransomware') {
      strategies.push({ type: 'isolation', target: 'Affected systems', rationale: 'Prevent spread', reversible: true, businessImpact: 'high' });
    }
    if (incident.type === 'phishing') {
      strategies.push({ type: 'credential_reset', target: 'Affected users', rationale: 'Revoke access', reversible: false, businessImpact: 'low' });
    }
    strategies.push({ type: 'block', target: 'Malicious IPs', rationale: 'Stop C2 communication', reversible: true, businessImpact: 'none' });
    return strategies;
  }

  assessBusinessImpact(strategy: ContainmentStrategy, affectedSystems: string[]): { impact: string; alternatives: string[]; recommendation: string } {
    return {
      impact: strategy.businessImpact === 'high' ? 'High impact' : 'Manageable impact',
      alternatives: ['Phase implementation'],
      recommendation: strategy.businessImpact === 'high' ? 'Get approval first' : 'Proceed',
    };
  }

  generateStatusUpdate(incidentId: string, audience: 'technical' | 'executive' | 'legal'): string {
    const incident = this.incidents.get(incidentId);
    if (!incident) return 'Not found';
    if (audience === 'technical') return \`Status: \${incident.status}, Systems: \${incident.affectedSystems.join(', ')}\`;
    if (audience === 'executive') return \`Security incident in progress. Status: \${incident.status}. \${incident.affectedSystems.length} systems affected.\`;
    return \`Legal hold notice. Incident: \${incident.id}. Evidence being preserved.\`;
  }

  determineNotificationRequirements(incident: Incident): { regulatory: string[]; internal: string[]; external: string[]; timeline: string } {
    const regulatory = incident.type === 'data_breach' ? ['GDPR Authority', 'State AG'] : [];
    return { regulatory, internal: ['CISO', 'Legal'], external: incident.type === 'data_breach' ? ['Affected customers'] : [], timeline: '72 hours' };
  }

  generateIncidentReport(incidentId: string): { incidentId: string; executiveSummary: string; timeline: any[]; rootCause: string; impactAssessment: any; lessonsLearned: string[]; recommendations: string[] } {
    const incident = this.incidents.get(incidentId);
    if (!incident) return { incidentId, executiveSummary: 'Not found', timeline: [], rootCause: 'Unknown', impactAssessment: {}, lessonsLearned: [], recommendations: [] };
    return {
      incidentId, executiveSummary: \`\${incident.type} incident affecting \${incident.affectedSystems.length} systems\`,
      timeline: incident.timeline, rootCause: 'Under investigation',
      impactAssessment: { systemsAffected: incident.affectedSystems.length, dataExposed: incident.type === 'data_breach', downtime: '2 hours', estimatedCost: '$50,000' },
      lessonsLearned: ['Improve detection'], recommendations: ['Enhance monitoring'],
    };
  }

  getInterviewScenario(difficulty: 'junior' | 'senior' | 'lead'): { scenario: string; expectedActions: string[]; commonMistakes: string[] } {
    if (difficulty === 'senior') return {
      scenario: 'Ransomware spreading across network', expectedActions: ['Isolate systems', 'Check backups', 'Notify leadership'],
      commonMistakes: ['Paying ransom immediately', 'Not preserving evidence'],
    };
    return { scenario: 'User reports suspicious email', expectedActions: ['Isolate workstation', 'Interview user'], commonMistakes: ['Ignoring the report'] };
  }

  evaluateResponse(actions: string[], expectedActions: string[]): { score: number; correct: string[]; missed: string[]; extras: string[] } {
    const correct = actions.filter(a => expectedActions.some(e => e.toLowerCase().includes(a.toLowerCase().split(' ')[0])));
    const missed = expectedActions.filter(e => !actions.some(a => e.toLowerCase().includes(a.toLowerCase().split(' ')[0])));
    return { score: Math.round((correct.length / expectedActions.length) * 100), correct, missed, extras: [] };
  }
}

describe('IncidentResponseSimulator', () => {
  let simulator: IncidentResponseSimulator;

  beforeEach(() => {
    simulator = new IncidentResponseSimulator();
  });

  it('should create incidents with proper severity', () => {
    const ransomware = simulator.createIncident('ransomware', ['encrypted files'], ['server-001']);
    const phishing = simulator.createIncident('phishing', ['suspicious link'], ['user-001']);

    expect(ransomware.severity).toBe('critical');
    expect(ransomware.status).toBe('detected');
    expect(phishing.severity).toBe('medium');
  });

  it('should update incident status and timeline', () => {
    const incident = simulator.createIncident('malware', ['trojan.exe'], ['ws-001']);
    simulator.updateStatus(incident.id, 'contained', 'Isolated infected system');

    // Get updated incident
    const updated = simulator.createIncident('malware', [], []);
    expect(updated.status).toBe('detected'); // New incident
  });

  it('should analyze indicators correctly', () => {
    const analysis = simulator.analyzeIndicators([
      '192.168.1.100',
      '5d41402abc4b2a76b9719d911017c592',
    ]);

    expect(analysis[0].iocType).toBe('IP Address');
    expect(analysis[1].iocType).toBe('File Hash');
    expect(analysis[1].confidence).toBe('high');
  });

  it('should recommend containment strategies', () => {
    const incident = simulator.createIncident('ransomware', ['encrypted'], ['server-001']);
    const strategies = simulator.recommendContainmentStrategies(incident);

    expect(strategies.some(s => s.type === 'isolation')).toBe(true);
    expect(strategies[0].rationale.length).toBeGreaterThan(0);
  });

  it('should generate appropriate status updates', () => {
    const incident = simulator.createIncident('data_breach', ['exfiltration'], ['db-001']);

    const techUpdate = simulator.generateStatusUpdate(incident.id, 'technical');
    const execUpdate = simulator.generateStatusUpdate(incident.id, 'executive');
    const legalUpdate = simulator.generateStatusUpdate(incident.id, 'legal');

    expect(techUpdate).toContain('Status');
    expect(execUpdate).toContain('incident');
    expect(legalUpdate).toContain('Legal');
  });

  it('should determine notification requirements', () => {
    const breach = simulator.createIncident('data_breach', ['data exposed'], ['db-001']);
    const malware = simulator.createIncident('malware', ['virus'], ['ws-001']);

    const breachNotify = simulator.determineNotificationRequirements(breach);
    const malwareNotify = simulator.determineNotificationRequirements(malware);

    expect(breachNotify.regulatory.length).toBeGreaterThan(0);
    expect(malwareNotify.regulatory.length).toBe(0);
  });

  it('should generate incident reports', () => {
    const incident = simulator.createIncident('ransomware', ['ransom note'], ['server-001']);
    const report = simulator.generateIncidentReport(incident.id);

    expect(report.executiveSummary).toContain('ransomware');
    expect(report.lessonsLearned.length).toBeGreaterThan(0);
    expect(report.recommendations.length).toBeGreaterThan(0);
  });

  it('should provide interview scenarios by difficulty', () => {
    const junior = simulator.getInterviewScenario('junior');
    const senior = simulator.getInterviewScenario('senior');

    expect(junior.scenario.length).toBeGreaterThan(0);
    expect(senior.expectedActions.length).toBeGreaterThan(junior.expectedActions.length - 1);
  });

  it('should evaluate candidate responses', () => {
    const expected = ['Isolate systems', 'Check backups', 'Notify leadership'];
    const goodResponse = ['Isolate the affected systems', 'Verify backup integrity'];
    const badResponse = ['Restart everything'];

    const goodEval = simulator.evaluateResponse(goodResponse, expected);
    const badEval = simulator.evaluateResponse(badResponse, expected);

    expect(goodEval.score).toBeGreaterThan(badEval.score);
  });

  it('should assess business impact of containment', () => {
    const strategy: ContainmentStrategy = {
      type: 'isolation', target: 'Production servers', rationale: 'Stop spread',
      reversible: true, businessImpact: 'high',
    };

    const assessment = simulator.assessBusinessImpact(strategy, ['prod-1', 'prod-2']);

    expect(assessment.impact).toContain('impact');
    expect(assessment.recommendation.length).toBeGreaterThan(0);
  });
});`,
	hint1:
		'For incident severity, start with a base score by incident type (ransomware = 4, malware = 2), then adjust based on the number of affected systems and critical indicators like "encrypted" or "exfiltration".',
	hint2:
		'For containment strategies, map incident types to appropriate responses: ransomware → isolation, phishing → credential reset, DDoS → block. Consider business impact and reversibility for each strategy.',
	whyItMatters: `Incident response skills are tested extensively in security operations and IR interviews:

**Why IR Matters:**
- **Time Critical** - Every minute counts during an active incident
- **Business Impact** - Poor IR can turn a minor incident into a major breach
- **Legal Implications** - How you respond affects regulatory compliance

**Interview Focus Areas:**
- **NIST Framework** - Preparation, Detection, Containment, Eradication, Recovery, Lessons Learned
- **Decision Making** - When to contain vs. when to monitor
- **Communication** - Status updates for different audiences
- **Documentation** - Chain of custody, timeline preservation

**Real-World Examples:**
- **Target Breach (2013)**: Alerts were ignored, 40M cards compromised
- **Equifax (2017)**: Delayed disclosure led to $575M settlement
- **SolarWinds (2020)**: APT required careful "burn down" approach

**Common Interview Questions:**
- "Walk me through your first hour of response to a ransomware attack"
- "How do you decide when to notify customers?"
- "The CEO wants to pay the ransom. What do you say?"
- "How do you preserve evidence while containing an incident?"

Strong IR skills demonstrate readiness for real security operations work.`,
	order: 5,
	translations: {
		ru: {
			title: 'Сценарии реагирования на инциденты',
			description: `Практикуйте сценарии реагирования на инциденты - необходимо для собеседований по операциям безопасности.

**Фазы IR (NIST Framework):**

1. **Подготовка** - Политики, процедуры, обучение
2. **Обнаружение и анализ** - Идентификация и понимание инцидентов
3. **Сдерживание** - Остановка распространения
4. **Устранение** - Удаление угрозы
5. **Восстановление** - Восстановление операций
6. **Извлечённые уроки** - Улучшение для будущего

**Ваша задача:**

Реализуйте класс \`IncidentResponseSimulator\`, который проходит через методологию обработки инцидентов.`,
			hint1:
				'Для серьёзности инцидента начните с базового балла по типу (ransomware = 4, malware = 2), затем корректируйте на основе количества затронутых систем.',
			hint2:
				'Для стратегий сдерживания сопоставьте типы инцидентов с ответами: ransomware → изоляция, phishing → сброс учётных данных, DDoS → блокировка.',
			whyItMatters: `Навыки реагирования на инциденты активно проверяются на собеседованиях:

**Почему IR важен:**
- **Критичность времени** - Каждая минута на счету во время активного инцидента
- **Бизнес-влияние** - Плохой IR может превратить мелкий инцидент в крупную утечку
- **Юридические последствия** - Как вы реагируете влияет на соответствие регуляторам

**Фокус собеседования:**
- **NIST Framework** - Подготовка, Обнаружение, Сдерживание, Устранение, Восстановление, Уроки
- **Принятие решений** - Когда сдерживать vs когда наблюдать
- **Коммуникация** - Статусные обновления для разных аудиторий
- **Документация** - Цепочка хранения, сохранение таймлайна`,
		},
		uz: {
			title: 'Hodisalarga javob stsenariyları',
			description: `Hodisalarga javob stsenariylarini mashq qiling - xavfsizlik operatsiyalari intervyulari uchun zarur.

**IR Fazalari (NIST Framework):**

1. **Tayyorgarlik** - Siyosatlar, tartiblar, trening
2. **Aniqlash va tahlil** - Hodisalarni aniqlash va tushunish
3. **Cheklash** - Tarqalishni to'xtatish
4. **Yo'q qilish** - Tahdidni olib tashlash
5. **Tiklash** - Operatsiyalarni tiklash
6. **O'rganilgan saboqlar** - Kelajak uchun yaxshilash

**Vazifangiz:**

Hodisalarni boshqarish metodologiyasini bosib o'tadigan \`IncidentResponseSimulator\` klassini yarating.`,
			hint1:
				"Hodisa jiddiyligi uchun turga qarab bazaviy balldan boshlang (ransomware = 4, malware = 2), so'ngra ta'sirlangan tizimlar soniga qarab sozlang.",
			hint2:
				"Cheklash strategiyalari uchun hodisa turlarini javoblarga moslashtiring: ransomware → izolyatsiya, phishing → hisob ma'lumotlarini tiklash, DDoS → bloklash.",
			whyItMatters: `Hodisalarga javob ko'nikmalari intervyularda keng tekshiriladi:

**Nima uchun IR muhim:**
- **Vaqt muhim** - Faol hodisa vaqtida har bir daqiqa muhim
- **Biznesga ta'sir** - Yomon IR kichik hodisani katta buzilishga aylantirishi mumkin
- **Yuridik oqibatlar** - Qanday javob berishingiz regulyator muvofiqligiga ta'sir qiladi

**Intervyu fokus sohalari:**
- **NIST Framework** - Tayyorgarlik, Aniqlash, Cheklash, Yo'q qilish, Tiklash, Saboqlar
- **Qaror qabul qilish** - Qachon cheklash vs qachon kuzatish
- **Kommunikatsiya** - Turli auditoriyalar uchun status yangilanishlari
- **Hujjatlashtirish** - Saqlash zanjiri, vaqt chizig'ini saqlash`,
		},
	},
};

export default task;
