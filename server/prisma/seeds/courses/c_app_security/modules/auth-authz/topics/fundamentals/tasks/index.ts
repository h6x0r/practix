// Auth & Authorization Tasks
import jwtSecurity from './01-jwt-security';
import oauth2Security from './02-oauth2-security';
import sessionManagement from './03-session-management';
import mfaImplementation from './04-mfa-implementation';
import rbacAbac from './05-rbac-abac';
import apiKeySecurity from './06-api-key-security';

export const tasks = [
	jwtSecurity,
	oauth2Security,
	sessionManagement,
	mfaImplementation,
	rbacAbac,
	apiKeySecurity,
];
