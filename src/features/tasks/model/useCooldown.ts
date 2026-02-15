import { useState, useRef, useCallback, useEffect } from 'react';
import { RATE_LIMIT_COOLDOWN_MS } from './taskRunner.utils';

/**
 * Hook for managing rate-limiting cooldown between requests.
 * Provides cooldown state (remaining ms) and methods to start/check cooldown.
 */
export const useCooldown = () => {
  const [cooldownRemaining, setCooldownRemaining] = useState(0);
  const lastRequestTimeRef = useRef<number>(0);
  const cooldownTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (cooldownTimerRef.current) {
        clearInterval(cooldownTimerRef.current);
      }
    };
  }, []);

  const startCooldown = useCallback(() => {
    lastRequestTimeRef.current = Date.now();
    setCooldownRemaining(RATE_LIMIT_COOLDOWN_MS);

    if (cooldownTimerRef.current) {
      clearInterval(cooldownTimerRef.current);
    }

    cooldownTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - lastRequestTimeRef.current;
      const remaining = Math.max(0, RATE_LIMIT_COOLDOWN_MS - elapsed);

      if (isMountedRef.current) {
        setCooldownRemaining(remaining);
      }

      if (remaining === 0 && cooldownTimerRef.current) {
        clearInterval(cooldownTimerRef.current);
        cooldownTimerRef.current = null;
      }
    }, 100);
  }, []);

  const isRateLimited = useCallback((): boolean => {
    const elapsed = Date.now() - lastRequestTimeRef.current;
    return elapsed < RATE_LIMIT_COOLDOWN_MS;
  }, []);

  return { cooldownRemaining, startCooldown, isRateLimited };
};
