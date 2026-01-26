import "./globals.css";
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { DM_Sans } from "next/font/google";


const dm = DM_Sans({ subsets: ["latin"] });

export const metadata = {
  title: "BITS Admission IR",
  description: "BITS Admission Chatbot",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={dm.className}>{children}</body>
    </html>
  )
}
