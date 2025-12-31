/**
 * HTTP Middleware Advanced - Tasks Index
 * Advanced body manipulation, concurrency control, and composition
 */

import { task as bodyPreview } from './10-body-preview';
import { task as captureStatus } from './11-capture-status';
import { task as teeBody } from './12-tee-body';
import { task as prependBody } from './13-prepend-body';
import { task as decompressGzip } from './14-decompress-gzip';
import { task as concurrencyLimit } from './15-concurrency-limit';
import { task as timeout } from './16-timeout';
import { task as maxBytes } from './17-max-bytes';
import { task as chain } from './18-chain';

export const tasks = [
	bodyPreview,
	captureStatus,
	teeBody,
	prependBody,
	decompressGzip,
	concurrencyLimit,
	timeout,
	maxBytes,
	chain,
];
