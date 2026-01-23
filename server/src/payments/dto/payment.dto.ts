import { IsString, IsNumber, IsOptional, IsEnum, IsBoolean } from 'class-validator';

export enum PaymentProvider {
  PAYME = 'payme',
  CLICK = 'click',
}

export enum OrderType {
  SUBSCRIPTION = 'subscription',
  PURCHASE = 'purchase',
}

export enum PurchaseType {
  ROADMAP_GENERATION = 'roadmap_generation',
  AI_CREDITS = 'ai_credits',
}

/**
 * DTO for creating a checkout session
 */
export class CreateCheckoutDto {
  @IsEnum(OrderType)
  orderType: OrderType;

  @IsString()
  @IsOptional()
  planId?: string; // For subscriptions

  @IsEnum(PurchaseType)
  @IsOptional()
  purchaseType?: PurchaseType; // For one-time purchases

  @IsNumber()
  @IsOptional()
  quantity?: number; // For purchases (e.g., number of roadmap generations)

  @IsEnum(PaymentProvider)
  provider: PaymentProvider;

  @IsString()
  @IsOptional()
  returnUrl?: string;
}

/**
 * Response from checkout creation
 */
export class CheckoutResponse {
  orderId: string;
  paymentUrl: string;
  amount: number;
  currency: string;
  provider: PaymentProvider;
}

/**
 * Payme webhook request (JSON-RPC 2.0)
 */
export class PaymeWebhookDto {
  @IsString()
  method: string;

  @IsOptional()
  params?: Record<string, unknown>;

  @IsOptional()
  id?: number;
}

/**
 * Click webhook request
 */
export class ClickWebhookDto {
  @IsNumber()
  click_trans_id: number;

  @IsNumber()
  service_id: number;

  @IsString()
  merchant_trans_id: string;

  @IsNumber()
  @IsOptional()
  merchant_prepare_id?: number;

  @IsNumber()
  amount: number;

  @IsNumber()
  action: number; // 0 = Prepare, 1 = Complete

  @IsString()
  sign_time: string;

  @IsString()
  sign_string: string;

  @IsNumber()
  @IsOptional()
  error?: number;

  @IsString()
  @IsOptional()
  error_note?: string;
}

/**
 * Payment history response item
 */
export class PaymentHistoryItem {
  id: string;
  type: 'subscription' | 'purchase';
  description: string;
  amount: number;
  currency: string;
  status: string;
  provider?: string;
  createdAt: Date;
}
