import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import * as Sentry from '@sentry/node';

@Injectable()
export class SentryInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    return next.handle().pipe(
      catchError((error) => {
        // Skip common HTTP exceptions that aren't real errors
        const skipStatuses = [
          HttpStatus.BAD_REQUEST,
          HttpStatus.UNAUTHORIZED,
          HttpStatus.FORBIDDEN,
          HttpStatus.NOT_FOUND,
          HttpStatus.CONFLICT,
          HttpStatus.TOO_MANY_REQUESTS,
        ];

        if (error instanceof HttpException) {
          const status = error.getStatus();
          if (skipStatuses.includes(status)) {
            return throwError(() => error);
          }
        }

        // Capture the error in Sentry
        const request = context.switchToHttp().getRequest();

        Sentry.withScope((scope) => {
          // Add request context
          scope.setExtra('url', request.url);
          scope.setExtra('method', request.method);
          scope.setExtra('body', request.body);
          scope.setExtra('params', request.params);
          scope.setExtra('query', request.query);

          // Add user context if available
          if (request.user) {
            scope.setUser({
              id: request.user.id,
              email: request.user.email,
            });
          }

          // Add tags for filtering
          scope.setTag('handler', context.getHandler().name);
          scope.setTag('controller', context.getClass().name);

          Sentry.captureException(error);
        });

        return throwError(() => error);
      }),
    );
  }
}
