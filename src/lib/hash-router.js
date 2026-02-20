import { Fragment, createElement, useEffect } from "react";

function ensureHashPrefix(prefix) {
  if (typeof window === "undefined") {
    return;
  }
  const { location } = window;
  if (!location.hash || !location.hash.startsWith(prefix)) {
    window.history.replaceState(null, "", location.pathname + location.search + prefix);
  }
}

// Ensure hash is set before React renders
if (typeof window !== "undefined") {
  ensureHashPrefix("#/");
}

export function HashRouter({ children, prefix = "#/" }) {
  useEffect(() => {
    ensureHashPrefix(prefix);
    const handleHashChange = () => ensureHashPrefix(prefix);
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, [prefix]);

  return createElement(Fragment, null, children);
}

export default HashRouter;
