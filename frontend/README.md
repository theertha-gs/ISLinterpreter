# ISL Translator with Firebase Authentication

This project is a sign language translator with Firebase authentication integration.

## Firebase Setup

To set up Firebase for authentication:

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Add a web app to your Firebase project
3. Enable Authentication in the Firebase console and set up Email/Password authentication
4. Copy your Firebase configuration from the Firebase console
5. Create a `.env.local` file in the root of your project (use `.env.local.example` as a template)
6. Fill in the Firebase configuration values in your `.env.local` file

## Running the Project

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Authentication Flow

The application uses Firebase Authentication with the following features:

- Email/password login
- User registration
- Secure authentication state management
- Protected routes with middleware
- Logout functionality

## Environment Variables

Make sure to set up the following environment variables in your `.env.local` file:

```
# Firebase Configuration
NEXT_PUBLIC_FIREBASE_API_KEY=your-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-auth-domain
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-storage-bucket
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-messaging-sender-id
NEXT_PUBLIC_FIREBASE_APP_ID=your-app-id
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=your-measurement-id
``` 