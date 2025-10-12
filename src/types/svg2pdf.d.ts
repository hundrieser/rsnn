declare module "svg2pdf.js";

declare module "jspdf" {
  interface jsPDF {
    svg(element: SVGElement, options?: Record<string, unknown>): Promise<void>;
  }
}
