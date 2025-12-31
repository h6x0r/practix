import { task as resourceNamingBasic } from './01-resource-naming-basic';
import { task as resourceNamingNested } from './02-resource-naming-nested';
import { task as httpMethods } from './03-http-methods';
import { task as statusCodes } from './04-status-codes';
import { task as requestValidation } from './05-request-validation';
import { task as responseFormatting } from './06-response-formatting';
import { task as errorHandlingBasic } from './07-error-handling-basic';
import { task as errorHandlingAdvanced } from './08-error-handling-advanced';
import { task as versioningUrl } from './09-versioning-url';
import { task as versioningHeader } from './10-versioning-header';

export const tasks = [
	resourceNamingBasic,
	resourceNamingNested,
	httpMethods,
	statusCodes,
	requestValidation,
	responseFormatting,
	errorHandlingBasic,
	errorHandlingAdvanced,
	versioningUrl,
	versioningHeader,
];
