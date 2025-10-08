import { useEffect, useState } from "react";
import RSNNDesigner from "./RSNNDesigner";
import "./App.css";

export default function App() {
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const isDark = theme === "dark";

  return (
    <div className={`min-h-screen flex flex-col items-center theme-root ${isDark ? "theme-root-dark" : ""}`}>
      <div className="w-full max-w-6xl px-4 pt-6">
        <div className="flex justify-end mb-4">
          <button className="theme-toggle-button" onClick={toggleTheme} type="button">
            {isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
          </button>
        </div>
        <RSNNDesigner />
      </div>
    </div>
  );
}
