import { Injectable } from "@nestjs/common";
import { AuthGuard } from "@nestjs/passport";

@Injectable()
export class OptionalJwtAuthGuard extends AuthGuard("jwt") {
  handleRequest(err: any, user: any, info: any) {
    // No error is thrown if no user is found.
    // We simply return null for the user, allowing the controller to handle guest logic.
    return user || null;
  }
}
