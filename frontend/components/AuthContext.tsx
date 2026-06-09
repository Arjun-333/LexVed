"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";

interface AuthUser {
  username: string;
  role: "user" | "admin";
  displayName: string;
}

interface AuthContextType {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  authFetch: (url: string, options?: RequestInit) => Promise<Response>;
}

const AuthContext = createContext<AuthContextType | null>(null);

const API_URL = typeof window !== 'undefined' 
  ? `http://${window.location.hostname}:5000` 
  : (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000");

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Restore session from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem("lexved_token");
    if (storedToken) {
      // Validate token with backend
      fetch(`${API_URL}/api/auth/me`, {
        headers: { Authorization: `Bearer ${storedToken}` },
      })
        .then((res) => {
          if (!res.ok) throw new Error("Token invalid");
          return res.json();
        })
        .then((data) => {
          setUser({
            username: data.username,
            role: data.role,
            displayName: data.display_name,
          });
          setToken(storedToken);
        })
        .catch(() => {
          localStorage.removeItem("lexved_token");
          setToken(null);
          setUser(null);
        })
        .finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    try {
      const res = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Connection failed" }));
        return { success: false, error: data.detail || "Authentication failed" };
      }

      const data = await res.json();
      setToken(data.token);
      setUser({
        username: data.username,
        role: data.role,
        displayName: data.display_name,
      });
      localStorage.setItem("lexved_token", data.token);
      return { success: true };
    } catch (err) {
      return { success: false, error: "Cannot reach LexVed server. Is the backend running?" };
    }
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("lexved_token");
  }, []);

  const authFetch = useCallback(
    async (url: string, options: RequestInit = {}) => {
      const headers = new Headers(options.headers || {});
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }
      const res = await fetch(url, { ...options, headers });

      // Auto-logout on 401
      if (res.status === 401) {
        logout();
      }

      return res;
    },
    [token, logout]
  );

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isAdmin: user?.role === "admin",
        isLoading,
        login,
        logout,
        authFetch,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
