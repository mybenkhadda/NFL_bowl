# Combine Intelligence Lab — manual browser checklist (Milestone 10)

*Nobody has launched the Milestone 10 version of the app. Run this once after 10A passes. Tick each box; for any failure, send the page name, the step and the traceback shown in the terminal.*

**Launch (your terminal):** `pip install -r apps/combine_intelligence_lab/requirements.txt` → `cd apps/combine_intelligence_lab` → `streamlit run app.py`. Open the printed URL in Chrome or Edge at 1440 px or wider; repeat section 11 at ~800 px.

## 1 · Shell
- [ ] Sidebar shows the title, the tagline, **Presentation / demo mode**, **Position filter** and the data-version box (mart v1 · full mode · build time · Lab engine 10.1).
- [ ] Navigation shows five groups: Start · Prospects · NFL movement · Decide · Evidence (10 pages).
- [ ] No page shows the SMOKE warning (full mart).

## 2 · Overview
- [ ] Three question cards render.
- [ ] Signature comparison shows Anthony Bradford vs Mason McCormick, 5.08 s / 5.08 s, with a speed-by-distance chart.
- [ ] Tiles: 510 prospects (418 tracked) · 793 attempts · 3,266 twin pairs (2,097 with a difference) · 416,507 episodes · 23 case studies.
- [ ] **Open Stopwatch Twins with this pair** lands on that exact pair.

## 3 · Prospect Intelligence
- [ ] Search by typing a name (e.g. "Mason McCormick"); tiles show position, class, forty, split, weight/height, attempts, tier.
- [ ] Relative position context tiles show percentiles or "not available".
- [ ] Tabs: speed vs distance, speed vs time, acceleration vs speed, sprint replay (two attempts) and the measures table with ⓘ tooltips (hover shows repeatability · redundancy · NFL transfer · coverage).
- [ ] Choose a prospect with **no tracked attempt** (set Position filter, pick one with "0" attempts): the "No tracked forty attempt" message, no chart, no number invented.
- [ ] A prospect with one attempt: replay tab shows the single-attempt message.

## 4 · Stopwatch Twins
- [ ] Ranked table of comparable prospects; selecting another B updates every chart.
- [ ] Filters: tolerance slider cannot exceed 0.03; "Require the 10-yard split" reduces the list; tier "high only" reduces it; a feature filter reduces it; an over-tight combination shows the "No stopwatch twin" message.
- [ ] Replay: ▶ plays, ❚❚ pauses, slider scrubs; **time ↔ distance** and **speed ↔ acceleration** switch; attempt selectors change the traces; playback 0.25× is visibly slower than 2×; the caption reports how many cursor positions are interpolated; at the end of a shorter recording the marker says "end of recording".
- [ ] Phase chart lists five phases; a phase without data says "not identified"; the phase-reading card matches.
- [ ] Reliability table shows verdict badges and "Describes an athlete?"; stability cards for both athletes; NFL correspondence card.
- [ ] HTML download opens offline in a browser with charts.

## 5 · Signature Stories
- [ ] Three cards with ≤ 100-word texts; each tab shows hero visual, checklist, narration, technical expander, provenance JSON, Markdown download.
- [ ] Story 1 selection line states whether the McConkey/Bond pair passed the criteria.
- [ ] "All 09D case studies" tab lists 23 cases.

## 6 · NFL Movement Demands
- [ ] Heatmap; demand/dimension/position selectors; exposure table with eligible player-plays, rate, share, duration, distance; overlap note; example replay plays.

## 7 · NFL Movement Gap
- [ ] Matrix with D / P / × / n/a marks and a legend; hovering a cell shows class, counts, denominator and reason.
- [ ] Share tiles per position; largest-gap note.
- [ ] Position view: coloured bars + candidate measurements card (ends with the "value not established" sentence).
- [ ] Two-position comparison: grouped bars, ratio and flags (e.g. differing eligible plays); intensity distribution for the chosen demand.
- [ ] Executive summary download.

## 8 · Movement Match Lab
- [ ] Coverage bars, abstention card, matrix table with mean coverage, interval and RMS robust-z distance, alignment chart.

## 9 · Scouting Decision Simulator
- [ ] All five questions selectable; a two-athlete question with no B asks for a second prospect.
- [ ] B list shows stored twins first ("stopwatch twin…"), then others ("not a stored twin"); choosing a non-twin shows the explanation and **no** measured difference.
- [ ] Action shows six sections with category tags; tiles count the five categories.
- [ ] Export: three HTML downloads open; three **Prepare PDF** buttons produce PDFs (or a clear matplotlib message).
- [ ] From Stopwatch Twins, **Take this pair to the decision simulator** pre-selects B.

## 10 · Report Generator and Research Evidence
- [ ] Prospect report: HTML, one-page PDF card, full PDF report.
- [ ] Case study report with three views.
- [ ] Research Evidence: nine plain-language expanders each list executed claims (C01–C20); registry tables render.

## 11 · Presentation, responsiveness, wording
- [ ] Presentation mode enlarges type and charts; the demo route links work in order.
- [ ] At ~800 px width, columns stack and tables scroll without overlapping text.
- [ ] Search the UI text: "grade", "ranking" and "probability" appear only in statements that the Lab produces none; never "hidden star", "sleeper", "bust", "better player".
- [ ] Selected athlete persists across Prospect Intelligence → Stopwatch Twins → Movement Match Lab → Simulator.
