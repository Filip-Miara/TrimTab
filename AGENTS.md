## Optimization-Driven Development

These rules apply to ALL work in this project.

### 1. Optimize Everything
- Before delivery, identify and apply at least one concrete optimization (runtime, memory, architecture, or latency).
- Document each optimization: what changed, why, and the measured impact.
- Prefer algorithmic improvements over micro-optimizations.

### 2. Document Optimizations
- Every optimization gets an entry in `docs/optimizations/` (create a markdown file per component).
- Each entry: problem, approach, implementation, before/after metrics.

### 3. Modular & Reusable Code
- Extract reusable components into `src/lib/` or `src/utils/`.
- Avoid coupling — each module should have one responsibility and a clear interface.
- Before writing new code, search for existing utilities that cover the need.

### 4. No Shortcuts — Get It Right
- "Quick fix" is not an option. Fix root causes, not symptoms.
- Every function must handle edge cases: empty inputs, singletons, type mismatches, resource exhaustion.
- Tests are mandatory for non-trivial logic.
- If a problem is hard, research it rather than guessing.

### 5. Proactive Research via Triadic Synthesis
- For any non-trivial design decision or optimization opportunity, delegate an async `triadic-synthesis-engine` analysis via the `delegate()` tool.
- Use the Triadic Synthesis Engine (TSE) to discover novel approaches before settling on a solution.
- The TSE provides: structural decomposition (Concept-Wise Manipulation), relational analysis (Meta-Synthesis Engine lens cascade), and potential-space generation (Autopoietic Inquiry Engine).
- Always run at minimum the rapid mode (phases 0-5 + 11) for design decisions, full mode for architecture changes.
- Document TSE findings alongside the implementation decisions they informed.
