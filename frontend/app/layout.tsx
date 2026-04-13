import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LexVed | Advanced Legal Intelligence",
  description: "Private legal RAG system powered by Llama 3. Elite legal research with domain-aware sub-indexing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@600;700&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/icon?family=Material+Icons+Round"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased font-body">
        {children}
      </body>
    </html>
  );
}
