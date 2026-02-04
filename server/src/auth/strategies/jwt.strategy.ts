import { ExtractJwt, Strategy } from "passport-jwt";
import { PassportStrategy } from "@nestjs/passport";
import { Injectable, Logger, UnauthorizedException } from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import { JwtPayload, RequestUser } from "../../common/types";
import { PrismaService } from "../../prisma/prisma.service";

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  private readonly logger = new Logger(JwtStrategy.name);

  constructor(
    configService: ConfigService,
    private readonly prisma: PrismaService,
  ) {
    const jwtSecret = configService.get<string>("JWT_SECRET");

    if (!jwtSecret) {
      throw new Error(
        "FATAL: JWT_SECRET environment variable is not configured. " +
          "Authentication will not work. Please set JWT_SECRET in your .env file.",
      );
    }

    if (jwtSecret.length < 32) {
      throw new Error(
        "FATAL: JWT_SECRET is too short (minimum 32 characters required). " +
          "Please use a strong secret for production.",
      );
    }

    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: jwtSecret,
    });

    // Log successful initialization (without exposing secret)
    new Logger(JwtStrategy.name).log("JWT Strategy initialized successfully");
  }

  async validate(payload: JwtPayload): Promise<RequestUser> {
    // Check if user is banned
    const user = await this.prisma.user.findUnique({
      where: { id: payload.sub },
      select: { isBanned: true },
    });

    if (user?.isBanned) {
      throw new UnauthorizedException("Your account has been suspended");
    }

    // This object is injected into request.user
    return { userId: payload.sub, email: payload.email };
  }
}
