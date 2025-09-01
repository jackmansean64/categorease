import os
import jwt
from functools import wraps
from flask import request, jsonify, g
from flask_cognito import CognitoAuth
from mongo_models import User, create_user_with_subscription
import logging
from dotenv import load_dotenv

load_dotenv()

class CognitoConfig:
    """Configuration for AWS Cognito"""
    COGNITO_REGION = os.getenv('COGNITO_REGION', 'us-east-1')
    COGNITO_USER_POOL_ID = os.getenv('COGNITO_USER_POOL_ID')
    COGNITO_APP_CLIENT_ID = os.getenv('COGNITO_APP_CLIENT_ID')
    
    @classmethod
    def is_configured(cls):
        return all([cls.COGNITO_USER_POOL_ID, cls.COGNITO_APP_CLIENT_ID])

# Initialize Cognito Auth
cognito_auth = None
if CognitoConfig.is_configured():
    cognito_auth = CognitoAuth()

def init_cognito(app):
    """Initialize Cognito authentication with Flask app"""
    if not CognitoConfig.is_configured():
        logging.warning("Cognito not configured - authentication will be disabled")
        return
    
    app.config['COGNITO_REGION'] = CognitoConfig.COGNITO_REGION
    app.config['COGNITO_USER_POOL_ID'] = CognitoConfig.COGNITO_USER_POOL_ID
    app.config['COGNITO_CLIENT_ID'] = CognitoConfig.COGNITO_APP_CLIENT_ID
    
    cognito_auth.init_app(app)

def auth_required(f):
    """Decorator to require authentication for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip auth if Cognito is not configured (development mode)
        if not CognitoConfig.is_configured():
            logging.warning("Authentication skipped - Cognito not configured")
            # Create a mock user for development
            g.current_user = create_mock_user()
            return f(*args, **kwargs)
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header required'}), 401
        
        try:
            # Expected format: "Bearer <token>"
            token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
            
            # Verify JWT token with Cognito
            claims = cognito_auth.verify(token)
            
            if not claims:
                return jsonify({'error': 'Invalid token'}), 401
            
            # Get user info from claims
            email = claims.get('email')
            cognito_sub = claims.get('sub')
            
            if not email or not cognito_sub:
                return jsonify({'error': 'Invalid token claims'}), 401
            
            # Find or create user
            user = User.find_by_cognito_sub(cognito_sub)
            if not user:
                # Create new user with free trial
                user = create_user_with_subscription(email=email, cognito_sub=cognito_sub)
                logging.info(f"Created new user: {email}")
            
            # Store user in Flask g object
            g.current_user = user
            
            return f(*args, **kwargs)
            
        except (jwt.InvalidTokenError, IndexError, Exception) as e:
            logging.error(f"Authentication error: {e}")
            return jsonify({'error': 'Invalid token'}), 401
    
    return decorated_function

def optional_auth(f):
    """Decorator for endpoints that work with or without authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.current_user = None
        
        # Skip auth if Cognito is not configured
        if not CognitoConfig.is_configured():
            g.current_user = create_mock_user()
            return f(*args, **kwargs)
        
        # Try to get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
                claims = cognito_auth.verify(token)
                
                if claims:
                    email = claims.get('email')
                    cognito_sub = claims.get('sub')
                    
                    if email and cognito_sub:
                        user = User.find_by_cognito_sub(cognito_sub)
                        if not user:
                            user = create_user_with_subscription(email=email, cognito_sub=cognito_sub)
                        g.current_user = user
                        
            except Exception as e:
                logging.warning(f"Optional auth failed: {e}")
        
        return f(*args, **kwargs)
    
    return decorated_function

def usage_required(min_transactions=1):
    """Decorator to check if user has enough transaction credits"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user') or not g.current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            user = g.current_user
            
            if not user.has_transactions_remaining() or user.transactions_remaining < min_transactions:
                return jsonify({
                    'error': 'Insufficient transaction credits',
                    'transactions_remaining': user.transactions_remaining,
                    'subscription_tier': user.subscription_tier,
                    'upgrade_required': True
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def get_current_user():
    """Get the current authenticated user"""
    return getattr(g, 'current_user', None)

def create_mock_user():
    """Create a mock user for development when Cognito is not configured"""
    mock_user = User(
        email="dev@example.com",
        cognito_sub="dev-user-123",
        subscription_tier="premium"  # Give full access in dev mode
    )
    mock_user._id = "dev-user-id"
    mock_user.transactions_remaining = 1000
    return mock_user

def extract_user_info_from_token(token: str) -> dict:
    """Extract user information from JWT token without verification (for development)"""
    try:
        # Decode without verification (for development only)
        decoded = jwt.decode(token, options={"verify_signature": False})
        return {
            'email': decoded.get('email'),
            'cognito_sub': decoded.get('sub'),
            'name': decoded.get('name', decoded.get('given_name', ''))
        }
    except Exception as e:
        logging.error(f"Error extracting user info from token: {e}")
        return {}

def get_user_subscription_status(user: User) -> dict:
    """Get user's subscription status and usage information"""
    if not user:
        return {
            'authenticated': False,
            'subscription_tier': 'none',
            'transactions_remaining': 0,
            'transactions_used': 0
        }
    
    return {
        'authenticated': True,
        'email': user.email,
        'subscription_tier': user.subscription_tier,
        'transactions_remaining': user.transactions_remaining,
        'transactions_used_this_period': user.transactions_used_this_period,
        'current_period_start': user.current_period_start.isoformat() if user.current_period_start else None,
        'current_period_end': user.current_period_end.isoformat() if user.current_period_end else None
    }

def consume_user_transactions(user: User, count: int) -> bool:
    """Consume transaction credits and return success status"""
    if user.decrement_transactions(count):
        logging.info(f"User {user.email} consumed {count} transactions. Remaining: {user.transactions_remaining}")
        return True
    else:
        logging.warning(f"User {user.email} attempted to consume {count} transactions but only has {user.transactions_remaining}")
        return False