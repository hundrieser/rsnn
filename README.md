# GUI for Simulating Spiking Neural Networks

An interactive **React + TypeScript** canvas to **draw**, **simulate**, and **visualize** spiking neural networks.  
Add neurons, connect them with weighted directed edges, assign exogenous input spike trains, and step through the resulting recurrent spiking cascades with a raster plot and inline dynamics panels. Includes grouping, reusable **modules**, JSON import/export, and PDF canvas export.

Simply open the link: [hundrieser.github.io/rsnn](https://hundrieser.github.io/rsnn) 

---

## Features

- **Canvas editing**
  - Add/move/delete neurons on an SVG canvas (snap-to-grid)
  - Directed connections with arbitrary real weights (inhibitory/excitatory); no self-loops
  - Edge waypoints & routing: double-click a path to add handles, drag to adjust
  - Edit weights via an Edge Inspector; optionally hide labels for “standard” weights (1, 2, −2)
  - Input/Hidden/Output neuron roles and per-neuron label visibility/offsets

- **Input spike trains**
  - Per-input neuron exogenous spike trains (space/comma/newline separated)
  - One-click generators: Poisson(λ=3 Hz) sampler and integer grid (1, 2, …, ⌊T⌋)

- **Event-driven recurrent simulation**
  - Exponential decay memory parameter **h** with exact **h = 0** and **h = ∞** semantics
  - Strict threshold **meta > 1**; synchronized “wave” propagation at each event time
  - **Spike raster** (all neurons) and optional inline **per-neuron dynamics panels**
  - **Animation** of cascades per input time with prev/freeze/resume/next/end controls

- **Workflow & reuse**
  - **Groups**: group/ungroup selections, toggle labels, **compress** to a single glyph and **expand**
  - **Modules**: design motifs in a dedicated module editor and instantiate them on the main board

- **File I/O**
  - **Export JSON** (complete project state) & **Import JSON** (with sanitation)
  - **Export Canvas as PDF** (vector; via browser print dialog)

- **Quality-of-life**
  - Undo (Ctrl/Cmd+Z), marquee select (Shift-drag), keyboard nudging, ESC to cancel connect/dialogs
  - Tweak canvas size and visual styling (colors, headers, output potentials in panels)

