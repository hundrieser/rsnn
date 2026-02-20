// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import { createRoot } from "react-dom/client";
import { act } from "react";
import React from "react";
import App from "./App.jsx";

describe("App render test", () => {
  it("renders App content", async () => {
    const div = document.createElement("div");
    document.body.appendChild(div);
    
    await act(async () => {
      const root = createRoot(div);
      root.render(React.createElement(React.StrictMode, null, React.createElement(App)));
    });
    
    // Check that key content is visible
    const hasTitle = div.innerHTML.includes("Spiking Neural Network");
    const hasButtons = div.innerHTML.includes("Select");
    console.log("Has title:", hasTitle);
    console.log("Has buttons:", hasButtons);
    console.log("Full innerHTML:", div.innerHTML.slice(0, 500));
    
    expect(hasTitle).toBe(true);
    expect(hasButtons).toBe(true);
  });
});
