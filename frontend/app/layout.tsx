import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Persi — AI Founder Assistant",
  description: "Your AI-powered startup co-pilot. From idea to launch.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
