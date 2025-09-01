from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    def __init__(self):
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        db_name = os.getenv('MONGODB_DATABASE', 'categorease')
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        # Collections
        self.users = self.db.users
        self.subscriptions = self.db.subscriptions
        self.usage_tracking = self.db.usage_tracking
        
        # Create indexes for performance
        self.users.create_index("email", unique=True)
        self.users.create_index("cognito_sub", unique=True)
        self.subscriptions.create_index("user_id")
        self.subscriptions.create_index("stripe_subscription_id", unique=True)
        self.usage_tracking.create_index([("user_id", 1), ("date", -1)])

db = MongoDB()

class User:
    def __init__(self, email: str, cognito_sub: str, subscription_tier: str = "free_trial"):
        self.email = email
        self.cognito_sub = cognito_sub
        self.subscription_tier = subscription_tier
        self.transactions_remaining = 100 if subscription_tier == "free_trial" else 0
        self.transactions_used_this_period = 0
        self.current_period_start = datetime.now(timezone.utc)
        self.current_period_end = None
        self.stripe_customer_id = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def save(self):
        """Save user to MongoDB"""
        self.updated_at = datetime.now(timezone.utc)
        user_data = self.__dict__.copy()
        
        if hasattr(self, '_id'):
            # Update existing user
            result = db.users.update_one(
                {'_id': self._id},
                {'$set': user_data}
            )
            return result.modified_count > 0
        else:
            # Insert new user
            result = db.users.insert_one(user_data)
            self._id = result.inserted_id
            return result.acknowledged
    
    @classmethod
    def find_by_email(cls, email: str) -> Optional['User']:
        """Find user by email"""
        user_data = db.users.find_one({'email': email})
        if user_data:
            user = cls.__new__(cls)
            user.__dict__.update(user_data)
            return user
        return None
    
    @classmethod
    def find_by_cognito_sub(cls, cognito_sub: str) -> Optional['User']:
        """Find user by Cognito subject ID"""
        user_data = db.users.find_one({'cognito_sub': cognito_sub})
        if user_data:
            user = cls.__new__(cls)
            user.__dict__.update(user_data)
            return user
        return None
    
    @classmethod
    def find_by_id(cls, user_id: str) -> Optional['User']:
        """Find user by MongoDB ObjectId"""
        user_data = db.users.find_one({'_id': ObjectId(user_id)})
        if user_data:
            user = cls.__new__(cls)
            user.__dict__.update(user_data)
            return user
        return None
    
    def has_transactions_remaining(self) -> bool:
        """Check if user has transactions remaining"""
        return self.transactions_remaining > 0
    
    def decrement_transactions(self, count: int = 1) -> bool:
        """Decrement transaction count if available"""
        if self.transactions_remaining >= count:
            self.transactions_remaining -= count
            self.transactions_used_this_period += count
            self.save()
            return True
        return False
    
    def reset_monthly_usage(self, tier_limits: dict):
        """Reset monthly usage based on subscription tier"""
        limit_map = {
            'basic': tier_limits.get('basic', 100),
            'premium': tier_limits.get('premium', 500),
            'free_trial': tier_limits.get('free_trial', 100)
        }
        
        self.transactions_remaining = limit_map.get(self.subscription_tier, 0)
        self.transactions_used_this_period = 0
        self.current_period_start = datetime.now(timezone.utc)
        self.save()

class Subscription:
    def __init__(self, user_id: ObjectId, stripe_subscription_id: str, 
                 status: str, plan_type: str, current_period_end: datetime):
        self.user_id = user_id
        self.stripe_subscription_id = stripe_subscription_id
        self.status = status  # active, canceled, incomplete, etc.
        self.plan_type = plan_type  # basic, premium
        self.current_period_end = current_period_end
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def save(self):
        """Save subscription to MongoDB"""
        self.updated_at = datetime.now(timezone.utc)
        subscription_data = self.__dict__.copy()
        
        if hasattr(self, '_id'):
            # Update existing subscription
            result = db.subscriptions.update_one(
                {'_id': self._id},
                {'$set': subscription_data}
            )
            return result.modified_count > 0
        else:
            # Insert new subscription
            result = db.subscriptions.insert_one(subscription_data)
            self._id = result.inserted_id
            return result.acknowledged
    
    @classmethod
    def find_by_user_id(cls, user_id: ObjectId) -> Optional['Subscription']:
        """Find active subscription by user ID"""
        subscription_data = db.subscriptions.find_one({
            'user_id': user_id,
            'status': {'$in': ['active', 'trialing']}
        })
        if subscription_data:
            subscription = cls.__new__(cls)
            subscription.__dict__.update(subscription_data)
            return subscription
        return None
    
    @classmethod
    def find_by_stripe_id(cls, stripe_subscription_id: str) -> Optional['Subscription']:
        """Find subscription by Stripe subscription ID"""
        subscription_data = db.subscriptions.find_one({'stripe_subscription_id': stripe_subscription_id})
        if subscription_data:
            subscription = cls.__new__(cls)
            subscription.__dict__.update(subscription_data)
            return subscription
        return None

class UsageTracking:
    def __init__(self, user_id: ObjectId, transactions_used: int, 
                 batch_id: str = None, transaction_details: list = None):
        self.user_id = user_id
        self.date = datetime.now(timezone.utc).date()
        self.transactions_used = transactions_used
        self.batch_id = batch_id
        self.transaction_details = transaction_details or []
        self.created_at = datetime.now(timezone.utc)
    
    def save(self):
        """Save usage tracking to MongoDB"""
        usage_data = self.__dict__.copy()
        result = db.usage_tracking.insert_one(usage_data)
        self._id = result.inserted_id
        return result.acknowledged
    
    @classmethod
    def get_monthly_usage(cls, user_id: ObjectId, year: int, month: int) -> int:
        """Get total transactions used in a specific month"""
        start_date = datetime(year, month, 1).date()
        if month == 12:
            end_date = datetime(year + 1, 1, 1).date()
        else:
            end_date = datetime(year, month + 1, 1).date()
        
        pipeline = [
            {
                '$match': {
                    'user_id': user_id,
                    'date': {'$gte': start_date, '$lt': end_date}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'total_transactions': {'$sum': '$transactions_used'}
                }
            }
        ]
        
        result = list(db.usage_tracking.aggregate(pipeline))
        return result[0]['total_transactions'] if result else 0
    
    @classmethod
    def get_daily_usage(cls, user_id: ObjectId, date: datetime.date) -> list:
        """Get usage details for a specific date"""
        usage_records = db.usage_tracking.find({
            'user_id': user_id,
            'date': date
        })
        
        records = []
        for record in usage_records:
            usage = cls.__new__(cls)
            usage.__dict__.update(record)
            records.append(usage)
        
        return records

# Utility functions for common operations
def create_user_with_subscription(email: str, cognito_sub: str, stripe_customer_id: str = None) -> User:
    """Create a new user with free trial"""
    user = User(email=email, cognito_sub=cognito_sub, subscription_tier="free_trial")
    if stripe_customer_id:
        user.stripe_customer_id = stripe_customer_id
    user.save()
    return user

def upgrade_user_subscription(user_id: str, stripe_subscription_id: str, 
                            plan_type: str, current_period_end: datetime) -> bool:
    """Upgrade user to paid subscription"""
    user = User.find_by_id(user_id)
    if not user:
        return False
    
    # Update user subscription tier
    user.subscription_tier = plan_type
    
    # Reset transaction limits based on plan
    tier_limits = {'basic': 100, 'premium': 500}
    user.transactions_remaining = tier_limits.get(plan_type, 0)
    user.current_period_end = current_period_end
    user.save()
    
    # Create subscription record
    subscription = Subscription(
        user_id=user._id,
        stripe_subscription_id=stripe_subscription_id,
        status='active',
        plan_type=plan_type,
        current_period_end=current_period_end
    )
    subscription.save()
    
    return True

def track_usage(user_id: str, transactions_used: int, batch_id: str = None, 
                transaction_details: list = None) -> bool:
    """Track transaction usage for billing"""
    try:
        user_obj_id = ObjectId(user_id)
        usage = UsageTracking(
            user_id=user_obj_id,
            transactions_used=transactions_used,
            batch_id=batch_id,
            transaction_details=transaction_details
        )
        return usage.save()
    except Exception as e:
        print(f"Error tracking usage: {e}")
        return False