"use client"

import { useEffect, useState } from 'react';
import { AuthProvider } from '@/lib/auth-context';

// This component ensures AuthProvider only runs on the client side
function ClientAuthProvider({ children }: { children: React.ReactNode }) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    // Return a placeholder or loading state
    return <>{children}</>;
  }

  return <AuthProvider>{children}</AuthProvider>;
}

export default ClientAuthProvider; 