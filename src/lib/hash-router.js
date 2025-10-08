import { useEffect } from "react";

function ensureHashPrefix(prefix) {
  if (typeof window === "undefined") {
    return;
  }
  const { location } = window;
  if (!location.hash || !location.hash.startsWith(prefix)) {
    const base = location.href.split("#")[0];
    location.replace(`${base}${prefix}`);
  }
}

export function HashRouter({ children, prefix = "#/" }) {
  useEffect(() => {
    ensureHashPrefix(prefix);
    const handleHashChange = () => ensureHashPrefix(prefix);
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, [prefix]);

  return children;
}

export default HashRouter;
