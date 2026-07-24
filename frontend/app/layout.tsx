import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "BITS Research Regulations Search",
  description:
    "Search BITS Pilani research regulations, PhD procedures, fellowships, and travel-grant documents with source evidence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
