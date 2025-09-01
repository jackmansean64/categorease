# CategorEase Monetization Setup Guide

This guide walks you through setting up the complete monetization system for CategorEase, including user authentication, subscription billing, and usage tracking.

## Prerequisites

1. **MongoDB** - Set up a MongoDB instance (local or cloud like MongoDB Atlas)
2. **AWS Account** - For Cognito authentication
3. **Stripe Account** - For payment processing
4. **Python Environment** - Ensure you have Python 3.8+ installed

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Fill in your configuration values in `.env` (see sections below)

## Step 3: MongoDB Setup

### Option A: Local MongoDB
1. Install MongoDB locally
2. Start MongoDB service
3. Set `MONGODB_URI=mongodb://localhost:27017/` in `.env`

### Option B: MongoDB Atlas (Recommended for production)
1. Create a free MongoDB Atlas account
2. Create a new cluster
3. Get your connection string and set it in `MONGODB_URI`
4. Example: `MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/categorease?retryWrites=true&w=majority`

## Step 4: AWS Cognito Setup

### Create User Pool
1. Go to AWS Cognito Console
2. Create a new User Pool:
   - **Pool name**: CategorEase Users
   - **Sign-in options**: Email
   - **Password requirements**: Set as desired
   - **MFA**: Optional (recommended for production)

### Configure Google OAuth
1. In your User Pool, go to **Sign-in experience** → **Federated identity providers**
2. Add Google as an identity provider:
   - Get Google OAuth credentials from [Google Cloud Console](https://console.cloud.google.com/)
   - Create OAuth 2.0 Client ID for web application
   - Add your domain to authorized origins
3. Configure attribute mapping between Google and Cognito

### Create App Client
1. In your User Pool, go to **App integration** → **App clients**
2. Create a new app client:
   - **App type**: Public client
   - **Auth flows**: Allow SRP authentication
   - **OAuth flows**: Authorization code grant
   - **OAuth scopes**: openid, email, profile
   - **Callback URLs**: Your application URLs

### Update Environment Variables
```env
COGNITO_REGION=us-east-1
COGNITO_USER_POOL_ID=us-east-1_xxxxxxxxx
COGNITO_APP_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Step 5: Stripe Setup

### Create Stripe Account
1. Sign up at [Stripe.com](https://stripe.com)
2. Get your API keys from the Dashboard

### Configure Products and Prices (Automatic)
The application will automatically create products and prices when first run. You can also do this manually:

1. **Basic Plan**: $1/month, 100 transactions
2. **Premium Plan**: $5/month, 500 transactions

### Set up Webhooks
1. In Stripe Dashboard, go to **Developers** → **Webhooks**
2. Add endpoint: `https://yourdomain.com/api/webhooks/stripe`
3. Select these events:
   - `checkout.session.completed`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy the webhook signing secret

### Update Environment Variables
```env
STRIPE_SECRET_KEY=sk_test_xxxxxxxxxxxxxxxxxxxx
STRIPE_PUBLISHABLE_KEY=pk_test_xxxxxxxxxxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxxxxxxxxx
```

## Step 6: Deploy and Test

### Development Testing
1. Start the Flask application:
```bash
python app/server_flask.py
```

2. Open your Excel add-in and test:
   - Authentication flow (currently shows dev message)
   - Transaction categorization with usage tracking
   - Subscription management

### Production Deployment
1. Set up proper SSL certificates
2. Configure production MongoDB
3. Update Cognito callback URLs for production domain
4. Update Stripe webhook endpoints for production
5. Set `DEBUG=False` in production environment

## Step 7: AWS Cognito Hosted UI (Production)

For production, you'll want to set up the Cognito Hosted UI:

1. In Cognito User Pool, go to **App integration** → **Domain**
2. Create a custom domain or use Cognito domain
3. Update the `signIn()` function in `taskpane.html` to redirect to:
```javascript
const cognitoDomain = 'your-cognito-domain.auth.region.amazoncognito.com';
const clientId = 'your-client-id';
const redirectUri = encodeURIComponent(window.location.origin);
const authUrl = `https://${cognitoDomain}/oauth2/authorize?response_type=code&client_id=${clientId}&redirect_uri=${redirectUri}&scope=openid+email+profile&identity_provider=Google`;
window.location.href = authUrl;
```

## Usage Tracking and Billing

The system automatically:
- Tracks transaction usage per user
- Enforces subscription limits
- Handles subscription renewals via Stripe webhooks
- Resets monthly usage limits on billing cycle

## Testing the Complete Flow

1. **User Registration**: User signs up via Google OAuth
2. **Free Trial**: User gets 100 free transactions to try the service
3. **Usage Tracking**: Each categorization batch decrements user credits
4. **Upgrade Flow**: When credits run low, user sees upgrade options
5. **Payment Processing**: Stripe handles subscription creation
6. **Webhook Processing**: System updates user subscription status
7. **Monthly Reset**: Usage resets automatically on billing cycle

## Monitoring and Maintenance

- Monitor Stripe webhooks for failed payments
- Check MongoDB for usage patterns and scaling needs
- Monitor AWS Cognito for authentication issues
- Set up alerts for system errors

## Security Considerations

1. **Environment Variables**: Never commit `.env` file to version control
2. **API Keys**: Rotate keys regularly
3. **Webhook Security**: Verify webhook signatures
4. **User Data**: Encrypt sensitive user data
5. **Rate Limiting**: Implement rate limiting for API endpoints

## Support and Troubleshooting

### Common Issues

1. **Authentication not working**: Check Cognito configuration and callback URLs
2. **Payments failing**: Verify Stripe webhook endpoints and signatures
3. **Usage tracking incorrect**: Check MongoDB connection and data consistency
4. **Subscription status not updating**: Verify webhook events are being processed

### Logs
Check `flask_app.log` for application logs and debugging information.

## Next Steps

1. Set up monitoring and analytics
2. Implement email notifications for payment issues
3. Add more detailed usage analytics
4. Consider implementing usage-based pricing tiers
5. Add customer support features