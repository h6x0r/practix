// OWASP Top 10 Tasks
import sqlInjection from './01-sql-injection';
import xssPrevention from './02-xss-prevention';
import csrf from './03-csrf';
import brokenAuth from './04-broken-auth';
import idor from './05-idor';
import ssrf from './06-ssrf';
import cryptoFailures from './07-cryptographic-failures';
import securityMisconfig from './08-security-misconfig';
import securityLogging from './09-security-logging';
import accessControl from './10-access-control';

export const tasks = [
	sqlInjection,
	xssPrevention,
	csrf,
	brokenAuth,
	idor,
	ssrf,
	cryptoFailures,
	securityMisconfig,
	securityLogging,
	accessControl,
];
