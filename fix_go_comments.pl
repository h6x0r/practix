#!/usr/bin/env perl
use strict;
use warnings;

my @files = qw(
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/01-request-id-context.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/02-structured-fields.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/03-log-level-filter.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/logging/topics/implementation/tasks/04-async-buffered-logger.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/01-prometheus-endpoint.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/02-thread-safe-counter.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/03-histogram-latency.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/04-gauge-metrics.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/metrics/topics/implementation/tasks/05-metrics-registry.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/01-safe-calls.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/02-safe-goroutines.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/03-panic-to-error.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/panic-recovery/topics/implementation/tasks/04-retry-on-panic.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/01-pointer-operations.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/02-nil-safe-access.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/03-pointer-swap-advanced.ts
  /Users/isyahex/Desktop/prjct/kodla-starter/server/prisma/seeds/shared/modules/go/pointersx/topics/fundamentals/tasks/04-optional-pattern.ts
);

my $count = 0;

foreach my $file (@files) {
    if (-f $file) {
        # Read file
        open(my $fh, '<', $file) or die "Cannot open $file: $!";
        my $content = do { local $/; <$fh> };
        close($fh);

        # Replace 2 or more spaces before // with a single tab
        my $original = $content;
        $content =~ s/ {2,}(\/\/)/{

\t$1/gm;

        if ($content ne $original) {
            # Write back
            open($fh, '>', $file) or die "Cannot write $file: $!";
            print $fh $content;
            close($fh);

            $count++;
            my $basename = $file;
            $basename =~ s/.*\///;
            print "Fixed: $basename\n";
        }
    } else {
        print "File not found: $file\n";
    }
}

print "\nTotal files modified: $count\n";
