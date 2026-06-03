import { Manrope, Orbitron } from "next/font/google";
import "./globals.css";

const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"]
});

const orbitron = Orbitron({
  variable: "--font-orbitron",
  subsets: ["latin"]
});

export const metadata = {
  title: "Pokémon Identifier",
  description: "Identify Pokémon from images with premium UI"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${manrope.variable} ${orbitron.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
