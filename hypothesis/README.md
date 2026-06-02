# Hypothesis Experiment Pattern

Use this folder to keep short research records for simulation hypotheses. Each record should be small enough to review quickly and precise enough to reproduce.

Name hypothesis records as `H<number>_<experiment_name>_hypothesis.md`. The report scripts copy the matching record into the same output folder as the generated HTML, so each result folder contains the CSVs, report, and hypothesis snapshot used to interpret it.

## Pattern

1. State the hypothesis.
   - Make one falsifiable claim.
   - Name the simulation feature or solver behavior being tested.
   - Avoid combining multiple claims in one hypothesis.

2. Set up the experiment.
   - Use the simplest scene that isolates the claim.
   - Define the modes being compared.
   - Keep unrelated solver/settings changes fixed.
   - Sweep only the variable needed to stress the claim.

3. Choose measured quantities.
   - Prefer solver outputs when the hypothesis is about physical response.
   - Add derived quantities only when they make the solver outputs physically interpretable.
   - Add geometry/state checks when solver outputs alone could hide a bad state.

4. Add validity gates.
   - Identify conditions that make the comparison invalid.
   - Fail or mark inconclusive when those conditions occur.
   - Keep debug data separate from primary physics data.

5. Separate generation from interpretation.
   - Newton experiment scripts generate CSVs only.
   - Report scripts read CSVs and generate summaries/HTML.
   - Do not make the experiment depend on the report.

6. Conclude from evidence.
   - Report pass, fail, or inconclusive.
   - Tie the conclusion directly to measured quantities and validity gates.
   - Keep the record concise.

## Recommended Record Fields

- Hypothesis
- Setup
- Modes
- Sweep variable
- Measured quantities
- Validity gates
- Output files
- Result
- Notes / next hypotheses

## Writing the HTML Report

Report scripts live in `tools/<experiment_name>_report.py` and read the CSVs the
experiment wrote. All reports share one renderer, `tools/_report_common.py`, so
they look identical and cannot drift apart. Do not hand-write `<style>` blocks,
SVG, or page scaffolding in a report script; build data specs and let the shared
module render them.

The shared module emits self-contained inline SVG with no external dependencies,
so a report is a single offline HTML file. It imports only the standard library.

1. Import the shared module as a sibling.
   - Add the script's directory to `sys.path`, then `import _report_common as rc`.
   - This works whether the script is run directly or loaded by its test.

   ```python
   import sys
   from pathlib import Path

   sys.path.insert(0, str(Path(__file__).resolve().parent))

   import _report_common as rc  # noqa: E402
   ```

2. Build figures from data, not markup.
   - A `rc.Series(xs, ys, label, color, draw_line=, draw_marker=, dash=)` is one curve.
   - A `rc.Figure(title, xlabel, ylabel, series, x_range, y_range, ...)` is one chart.
     Optional fields: `x_ticks`, `log_y`, `hlines`, `xbands`, `ybands`.
   - Use `rc.mode_series(rows_by_mode, x_key=, y_key=, scale=)` to get the canonical
     reduce off / reduce on pair in one call.
   - Compute axis bounds with `rc.padded_range(values, include=, floor_span=)`.

   ```python
   series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="solver_fz_N")
   figure = rc.Figure(
       title="Solver vertical support",
       xlabel="time [s]",
       ylabel="Fz [N]",
       series=series,
       x_range=rc.padded_range([t for s in series for t in s.xs], include=(0.0,)),
       y_range=rc.padded_range([y for s in series for y in s.ys], include=(0.0,)),
   )
   ```

3. Lay figures out and assemble the page.
   - `rc.figure_grid(figures, columns=1)` wraps figures in a responsive grid
     (`columns` 1-3; collapses to one column on narrow screens).
   - `rc.figure_tabs([rc.TabPanel(label, content), ...])` builds a tabbed switcher
     with self-contained, scoped JavaScript. Panel content is raw HTML, typically a
     caption plus one or more `figure_grid` blocks.
   - `rc.bullet_list(items)` escapes a list of strings into `<li>` rows; wrap it in
     your own `<ul>`. `rc.data_table(headers, rows)` renders a styled table from
     pre-formatted string cells.
   - `rc.render_page(title=, body=, extra_css=)` wraps the assembled body in the
     canonical document shell. Use `extra_css` only for a report-specific component
     the theme does not cover (e.g. a setup schematic).

4. Keep the contract the tests and result folders rely on.
   - Expose `TIMESERIES_CSV`, `SUMMARY_CSV`, and `HYPOTHESIS_RECORD_NAME` constants.
   - Expose `write_html_report(...)` that writes the HTML and `shutil.copyfile`s the
     matching hypothesis record into the output folder next to the HTML.
   - Title the page `H<n>: <name> Contact Reduction` and number figure tabs
     `Figure 1`, `Figure 2`, ... for consistency across reports.

5. Use the shared style; do not re-theme.
   - Mode colors and labels come from `rc.MODES`, `rc.MODE_LABELS`, `rc.MODE_COLORS`,
     and `rc.REFERENCE_COLOR`. Dashed analytic references use `rc.REFERENCE_COLOR`.
   - Numbers go through `rc.format_number` / `rc.format_percent` so formatting matches.

Add a `tools/test_<experiment_name>_report.py` that writes tiny synthetic CSVs,
calls `write_html_report`, and asserts the key strings appear in the output. Run
`uvx pre-commit run -a` before committing. `tools/cube_on_plate_tipping_report.py`
is the smallest end-to-end example to copy from.
