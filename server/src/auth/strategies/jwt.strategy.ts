import { ExtractJwt, Strategy } from 'passport-jwt';
import { PassportStrategy } from '@nestjs/passport';
import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { JwtPayload, RequestUser } from '../../common/types';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  private readonly logger = new Logger(JwtStrategy.name);

  constructor(configService: ConfigService) {
    const jwtSecret = configService.get<string>('JWT_SECRET');

    if (!jwtSecret) {
      throw new Error(
        'FATAL: JWT_SECRET environment variable is not configured. ' +
        'Authentication will not work. Please set JWT_SECRET in your .env file.'
      );
    }

    if (jwtSecret.length < 32) {
      throw new Error(
        'FATAL: JWT_SECRET is too short (minimum 32 characters required). ' +
        'Please use a strong secret for production.'
      );
    }

    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: jwtSecret,
    });

    // Log successful initialization (without exposing secret)
    new Logger(JwtStrategy.name).log('JWT Strategy initialized successfully');
  }

  async validate(payload: JwtPayload): Promise<RequestUser> {
    // This object is injected into request.user
    return { userId: payload.sub, email: payload.email };
  }
}