import os
import stripe
from datetime import datetime, timezone
from typing import Dict, List, Optional
from mongo_models import User, Subscription, UsageTracking, upgrade_user_subscription, track_usage
import logging
from dotenv import load_dotenv

load_dotenv()

# Stripe configuration
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

# Plan configuration
PLANS = {
    'basic': {
        'name': 'Basic Plan',
        'price': 100,  # $1.00 in cents
        'transaction_limit': 100,
        'description': '100 transaction categorizations per month'
    },
    'premium': {
        'name': 'Premium Plan', 
        'price': 500,  # $5.00 in cents
        'transaction_limit': 500,
        'description': '500 transaction categorizations per month'
    }
}

class StripeService:
    def __init__(self):
        self.stripe = stripe
        if not stripe.api_key:
            logging.warning("Stripe API key not configured")
    
    def create_customer(self, email: str, name: str = None) -> Optional[stripe.Customer]:
        """Create a Stripe customer"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={'source': 'categorease'}
            )
            logging.info(f"Created Stripe customer: {customer.id} for {email}")
            return customer
        except stripe.error.StripeError as e:
            logging.error(f"Error creating Stripe customer: {e}")
            return None
    
    def create_or_get_customer(self, user: User) -> Optional[stripe.Customer]:
        """Create or retrieve existing Stripe customer for user"""
        if user.stripe_customer_id:
            try:
                return stripe.Customer.retrieve(user.stripe_customer_id)
            except stripe.error.StripeError:
                # Customer doesn't exist, create new one
                pass
        
        # Create new customer
        customer = self.create_customer(user.email)
        if customer:
            user.stripe_customer_id = customer.id
            user.save()
        
        return customer
    
    def create_checkout_session(self, user: User, plan_type: str, 
                              success_url: str, cancel_url: str) -> Optional[stripe.checkout.Session]:
        """Create a Stripe Checkout session for subscription"""
        if plan_type not in PLANS:
            logging.error(f"Invalid plan type: {plan_type}")
            return None
        
        customer = self.create_or_get_customer(user)
        if not customer:
            return None
        
        try:
            # Get or create price for the plan
            price = self.get_or_create_price(plan_type)
            if not price:
                return None
            
            session = stripe.checkout.Session.create(
                customer=customer.id,
                payment_method_types=['card'],
                line_items=[{
                    'price': price.id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    'user_id': str(user._id),
                    'plan_type': plan_type
                },
                subscription_data={
                    'metadata': {
                        'user_id': str(user._id),
                        'plan_type': plan_type
                    }
                }
            )
            
            logging.info(f"Created checkout session: {session.id} for user {user.email}")
            return session
            
        except stripe.error.StripeError as e:
            logging.error(f"Error creating checkout session: {e}")
            return None
    
    def get_or_create_price(self, plan_type: str) -> Optional[stripe.Price]:
        """Get or create a Stripe price for the plan"""
        if plan_type not in PLANS:
            return None
        
        plan = PLANS[plan_type]
        
        # First try to find existing price
        try:
            prices = stripe.Price.list(
                active=True,
                lookup_keys=[f"categorease_{plan_type}"]
            )
            if prices.data:
                return prices.data[0]
        except stripe.error.StripeError:
            pass
        
        # Create new product and price
        try:
            product = stripe.Product.create(
                name=plan['name'],
                description=plan['description'],
                metadata={
                    'plan_type': plan_type,
                    'transaction_limit': str(plan['transaction_limit'])
                }
            )
            
            price = stripe.Price.create(
                product=product.id,
                unit_amount=plan['price'],
                currency='usd',
                recurring={'interval': 'month'},
                lookup_key=f"categorease_{plan_type}",
                metadata={'plan_type': plan_type}
            )
            
            logging.info(f"Created product {product.id} and price {price.id} for {plan_type}")
            return price
            
        except stripe.error.StripeError as e:
            logging.error(f"Error creating product/price: {e}")
            return None
    
    def create_customer_portal_session(self, user: User, return_url: str) -> Optional[stripe.billing_portal.Session]:
        """Create a customer portal session for subscription management"""
        customer = self.create_or_get_customer(user)
        if not customer:
            return None
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer.id,
                return_url=return_url
            )
            return session
        except stripe.error.StripeError as e:
            logging.error(f"Error creating portal session: {e}")
            return None
    
    def get_subscription_info(self, subscription_id: str) -> Optional[stripe.Subscription]:
        """Get subscription information from Stripe"""
        try:
            return stripe.Subscription.retrieve(subscription_id)
        except stripe.error.StripeError as e:
            logging.error(f"Error retrieving subscription: {e}")
            return None
    
    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription"""
        try:
            stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
            logging.info(f"Cancelled subscription: {subscription_id}")
            return True
        except stripe.error.StripeError as e:
            logging.error(f"Error cancelling subscription: {e}")
            return False
    
    def create_usage_record(self, subscription_item_id: str, quantity: int, 
                          timestamp: int = None) -> Optional[stripe.UsageRecord]:
        """Create a usage record for metered billing"""
        try:
            usage_record = stripe.UsageRecord.create(
                subscription_item=subscription_item_id,
                quantity=quantity,
                timestamp=timestamp or int(datetime.now().timestamp()),
                action='increment'
            )
            return usage_record
        except stripe.error.StripeError as e:
            logging.error(f"Error creating usage record: {e}")
            return None
    
    def handle_webhook(self, payload: str, sig_header: str) -> Dict:
        """Handle Stripe webhook events"""
        if not STRIPE_WEBHOOK_SECRET:
            logging.error("Stripe webhook secret not configured")
            return {'status': 'error', 'message': 'Webhook secret not configured'}
        
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except ValueError as e:
            logging.error(f"Invalid payload: {e}")
            return {'status': 'error', 'message': 'Invalid payload'}
        except stripe.error.SignatureVerificationError as e:
            logging.error(f"Invalid signature: {e}")
            return {'status': 'error', 'message': 'Invalid signature'}
        
        # Handle the event
        if event['type'] == 'checkout.session.completed':
            return self._handle_checkout_completed(event['data']['object'])
        elif event['type'] == 'invoice.payment_succeeded':
            return self._handle_payment_succeeded(event['data']['object'])
        elif event['type'] == 'invoice.payment_failed':
            return self._handle_payment_failed(event['data']['object'])
        elif event['type'] == 'customer.subscription.updated':
            return self._handle_subscription_updated(event['data']['object'])
        elif event['type'] == 'customer.subscription.deleted':
            return self._handle_subscription_deleted(event['data']['object'])
        else:
            logging.info(f"Unhandled webhook event type: {event['type']}")
            return {'status': 'success', 'message': 'Event received'}
    
    def _handle_checkout_completed(self, session) -> Dict:
        """Handle successful checkout completion"""
        try:
            user_id = session['metadata']['user_id']
            plan_type = session['metadata']['plan_type']
            
            # Get the subscription
            subscription_id = session['subscription']
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Update user subscription
            current_period_end = datetime.fromtimestamp(
                subscription['current_period_end'], 
                tz=timezone.utc
            )
            
            success = upgrade_user_subscription(
                user_id=user_id,
                stripe_subscription_id=subscription_id,
                plan_type=plan_type,
                current_period_end=current_period_end
            )
            
            if success:
                logging.info(f"Successfully upgraded user {user_id} to {plan_type}")
                return {'status': 'success', 'message': 'Subscription created'}
            else:
                logging.error(f"Failed to upgrade user {user_id}")
                return {'status': 'error', 'message': 'Failed to update user'}
                
        except Exception as e:
            logging.error(f"Error handling checkout completion: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _handle_payment_succeeded(self, invoice) -> Dict:
        """Handle successful payment"""
        try:
            subscription_id = invoice['subscription']
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            user_id = subscription['metadata']['user_id']
            plan_type = subscription['metadata']['plan_type']
            
            # Reset user's monthly usage
            user = User.find_by_id(user_id)
            if user:
                tier_limits = {
                    'basic': PLANS['basic']['transaction_limit'],
                    'premium': PLANS['premium']['transaction_limit']
                }
                user.reset_monthly_usage(tier_limits)
                logging.info(f"Reset monthly usage for user {user.email}")
            
            return {'status': 'success', 'message': 'Payment processed'}
            
        except Exception as e:
            logging.error(f"Error handling payment success: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _handle_payment_failed(self, invoice) -> Dict:
        """Handle failed payment"""
        try:
            subscription_id = invoice['subscription']
            subscription = stripe.Subscription.retrieve(subscription_id)
            user_id = subscription['metadata']['user_id']
            
            # You might want to send an email notification here
            logging.warning(f"Payment failed for user {user_id}")
            
            return {'status': 'success', 'message': 'Payment failure processed'}
            
        except Exception as e:
            logging.error(f"Error handling payment failure: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _handle_subscription_updated(self, subscription) -> Dict:
        """Handle subscription updates"""
        try:
            user_id = subscription['metadata']['user_id']
            status = subscription['status']
            
            # Update subscription status in database
            db_subscription = Subscription.find_by_stripe_id(subscription['id'])
            if db_subscription:
                db_subscription.status = status
                db_subscription.save()
                logging.info(f"Updated subscription status to {status} for user {user_id}")
            
            return {'status': 'success', 'message': 'Subscription updated'}
            
        except Exception as e:
            logging.error(f"Error handling subscription update: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _handle_subscription_deleted(self, subscription) -> Dict:
        """Handle subscription cancellation"""
        try:
            user_id = subscription['metadata']['user_id']
            
            user = User.find_by_id(user_id)
            if user:
                user.subscription_tier = 'free_trial'
                user.transactions_remaining = 0
                user.save()
                
                # Update subscription status
                db_subscription = Subscription.find_by_stripe_id(subscription['id'])
                if db_subscription:
                    db_subscription.status = 'canceled'
                    db_subscription.save()
                
                logging.info(f"Downgraded user {user.email} to free tier")
            
            return {'status': 'success', 'message': 'Subscription cancelled'}
            
        except Exception as e:
            logging.error(f"Error handling subscription deletion: {e}")
            return {'status': 'error', 'message': str(e)}

# Initialize the service
stripe_service = StripeService()

def create_subscription_checkout(user: User, plan_type: str, success_url: str, cancel_url: str) -> Optional[str]:
    """Create a checkout session and return the URL"""
    session = stripe_service.create_checkout_session(user, plan_type, success_url, cancel_url)
    return session.url if session else None

def create_customer_portal_url(user: User, return_url: str) -> Optional[str]:
    """Create a customer portal session and return the URL"""
    session = stripe_service.create_customer_portal_session(user, return_url)
    return session.url if session else None

def track_transaction_usage(user: User, transaction_count: int) -> bool:
    """Track transaction usage for billing purposes"""
    try:
        # Record usage in MongoDB
        success = track_usage(
            user_id=str(user._id),
            transactions_used=transaction_count,
            transaction_details=[{
                'timestamp': datetime.now(timezone.utc),
                'transaction_count': transaction_count,
                'cost': 0  # Cost tracking can be added later
            }]
        )
        
        if success:
            logging.info(f"Tracked {transaction_count} transactions for user {user.email}")
        
        return success
        
    except Exception as e:
        logging.error(f"Error tracking transaction usage: {e}")
        return False

def get_plan_info() -> Dict:
    """Get information about available plans"""
    return PLANS