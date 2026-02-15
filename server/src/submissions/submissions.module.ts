import { Module } from "@nestjs/common";
import { SubmissionsController } from "./submissions.controller";
import { SubmissionsService } from "./submissions.service";
import { SubmissionPromptService } from "./submission-prompt.service";
import { SubmissionHistoryService } from "./submission-history.service";
import { TestParserService } from "./test-parser.service";
import { ResultFormatterService } from "./result-formatter.service";
import { QueueModule } from "../queue/queue.module";
import { TasksModule } from "../tasks/tasks.module";
import { SubscriptionsModule } from "../subscriptions/subscriptions.module";
import { GamificationModule } from "../gamification/gamification.module";
import { CacheModule } from "../cache/cache.module";
import { AiModule } from "../ai/ai.module";
import { PlaygroundThrottlerGuard } from "../common/guards/playground-throttler.guard";

@Module({
  imports: [
    QueueModule,
    TasksModule,
    SubscriptionsModule,
    GamificationModule,
    CacheModule,
    AiModule,
  ],
  controllers: [SubmissionsController],
  providers: [
    SubmissionsService,
    SubmissionPromptService,
    SubmissionHistoryService,
    TestParserService,
    ResultFormatterService,
    PlaygroundThrottlerGuard,
  ],
  exports: [TestParserService, ResultFormatterService],
})
export class SubmissionsModule {}
