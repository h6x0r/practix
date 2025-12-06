
import { Injectable, UnauthorizedException, ConflictException } from '@nestjs/common';
import { UsersService } from '../users/users.service';
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcrypt';
import { LoginDto, RegisterDto } from './dto/auth.dto';

@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private jwtService: JwtService,
  ) {}

  async register(registerDto: RegisterDto) {
    // Check if user exists
    const existing = await this.usersService.findOne(registerDto.email);
    if (existing) {
      throw new ConflictException('User with this email already exists');
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(registerDto.password, 10);

    // Create user
    const user = await this.usersService.create({
      email: registerDto.email,
      name: registerDto.name,
      password: hashedPassword,
      isPremium: false,
      plan: null, // Explicitly set no plan
      preferences: {
        editorFontSize: 14,
        editorMinimap: false,
        editorTheme: 'vs-dark',
        editorLineNumbers: true,
        notifications: {
            emailDigest: true,
            newCourses: true
        }
      }, 
    });

    const token = this.generateToken(user);
    
    // Transform to DTO
    return { user: this.transformUser(user), token };
  }

  async login(loginDto: LoginDto) {
    const user = await this.usersService.findOne(loginDto.email);
    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isMatch = await bcrypt.compare(loginDto.password, user.password);
    if (!isMatch) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const token = this.generateToken(user);

    // Transform to DTO
    return { user: this.transformUser(user), token };
  }

  private generateToken(user: any) {
    const payload = { email: user.email, sub: user.id };
    return this.jwtService.sign(payload);
  }

  /**
   * Shapes the DB entity into the Frontend DTO.
   * Now strictly returns what is in the DB, no fake generation.
   */
  private transformUser(user: any) {
    const { password, ...result } = user;
    return result;
  }
}
