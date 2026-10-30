import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Fonts and KaTeX are bundled, so the demo works offline.
import "@fontsource-variable/inter";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";
import "katex/dist/katex.min.css";
import "uplot/dist/uPlot.min.css";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
