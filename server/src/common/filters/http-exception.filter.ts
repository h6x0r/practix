import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import { Request, Response } from 'express';
import { HttpExceptionResponse } from '../types';

@Catch()
export class GlobalExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(GlobalExceptionFilter.name);

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let message = 'Internal server error';
    let errors: string[] = [];

    if (exception instanceof HttpException) {
      status = exception.getStatus();
      const exceptionResponse = exception.getResponse();

      if (typeof exceptionResponse === 'string') {
        message = exceptionResponse;
      } else if (typeof exceptionResponse === 'object') {
        const res = exceptionResponse as HttpExceptionResponse;

        // Handle message field which can be string, string[], or object
        if (typeof res.message === 'string') {
          message = res.message;
        } else if (Array.isArray(res.message)) {
          message = res.message[0] || message;
          errors = res.message;
        } else if (res.message && typeof res.message === 'object') {
          message = JSON.stringify(res.message);
        }

        // Handle errors array
        if (res.errors) {
          errors = res.errors;
        }
      }
    } else if (exception instanceof Error) {
      message = exception.message;
      this.logger.error(`Unhandled error: ${exception.message}`, exception.stack);
    }

    const errorResponse = {
      statusCode: status,
      message,
      errors: errors.length > 0 ? errors : undefined,
      timestamp: new Date().toISOString(),
      path: request.url,
    };

    response.status(status).json(errorResponse);
  }
}
