/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: "#0A0C10",
        panelBg: "#12151E",
        panelBorder: "rgba(255, 255, 255, 0.08)",
        safeGreen: "#10B981",
        warningAmber: "#F59E0B",
        haltRed: "#EF4444",
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', "monospace"],
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
