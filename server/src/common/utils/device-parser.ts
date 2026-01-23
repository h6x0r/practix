import { DeviceType } from '@prisma/client';

/**
 * Mobile device patterns in User-Agent strings
 * Includes: iPhone, iPad, Android phones/tablets, Windows Phone, etc.
 */
const MOBILE_PATTERNS = [
  /iPhone/i,
  /iPad/i,
  /iPod/i,
  /Android.*Mobile/i, // Android phones (not tablets)
  /Android.*(?!Mobile)/i, // Android tablets - still count as mobile for our purposes
  /webOS/i,
  /BlackBerry/i,
  /Windows Phone/i,
  /Opera Mini/i,
  /IEMobile/i,
  /Mobile/i, // Generic mobile indicator
];

/**
 * Parse User-Agent string to determine device type
 * @param userAgent - The User-Agent header string
 * @returns DeviceType enum value (MOBILE, DESKTOP, or UNKNOWN)
 */
export function parseDeviceType(userAgent: string | undefined | null): DeviceType {
  if (!userAgent) {
    return DeviceType.UNKNOWN;
  }

  // Check if any mobile pattern matches
  const isMobile = MOBILE_PATTERNS.some((pattern) => pattern.test(userAgent));

  if (isMobile) {
    return DeviceType.MOBILE;
  }

  // If we have a User-Agent but it's not mobile, assume desktop
  // Common desktop browsers: Chrome, Firefox, Safari, Edge, Opera on Windows/Mac/Linux
  return DeviceType.DESKTOP;
}

/**
 * Get a human-readable device type name
 * @param deviceType - The DeviceType enum value
 * @returns Human-readable string
 */
export function getDeviceTypeName(deviceType: DeviceType): string {
  switch (deviceType) {
    case DeviceType.MOBILE:
      return 'Mobile';
    case DeviceType.DESKTOP:
      return 'Desktop';
    default:
      return 'Unknown';
  }
}
