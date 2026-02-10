# YAML Config Files — Line-by-Line Explanation

The YAML config files are where **all the prompt engineering lives**. They define who each agent is (agents.yaml) and what each agent does (tasks.yaml). This separation means you can tune agent behavior by editing YAML — no Python changes needed.

---

## agents.yaml — Agent Personas

Each agent has three fields: `role`, `goal`, and `backstory`. Together, these form the system prompt that tells the LLM who it is and how to behave.

### trend_scanner

```yaml
trend_scanner:
  role: >
    {industry} Competitive Trend Scanner
```
**Lines 1-3:** The `>` in YAML is a **folded block scalar** — it takes the following indented lines and joins them into a single line, replacing newlines with spaces. So this becomes the string `"{industry} Competitive Trend Scanner"`. The `{industry}` placeholder gets replaced at runtime by Python's `.format()` method (e.g., becomes `"Hydraulics & Mobile Machinery Competitive Trend Scanner"`).

```yaml
  goal: >
    Find the latest competitor news, product launches, product changes, R&D investments,
    patent filings, engineering blog posts, technical announcements, press releases,
    and hiring patterns for {company} and its competitors ({competitors}) in the {industry} industry.
```
**Lines 4-7:** The goal is specific and comprehensive. Notice how it lists many types of intelligence (product launches, R&D investments, patent filings, hiring patterns). This isn't just decoration — LLMs respond to specificity. A vague goal like "find competitor news" produces vague results. Listing specific categories primes the model to look for each one.

```yaml
  backstory: >
    You are an expert competitive intelligence analyst who monitors industry trends
    around the clock. You dig deep into product release notes, engineering blogs,
    R&D spending disclosures, patent databases, and technical conference announcements.
    You understand that product roadmap signals, R&D investment shifts, and engineering
    talent moves are the earliest indicators of competitive strategy changes.
    You always run multiple targeted searches per competitor to get comprehensive coverage.
```
**Lines 8-14:** The backstory is the most powerful prompt engineering technique here. It establishes **expertise and methodology**. When you tell an LLM "you are an expert who digs deep into product release notes," it actually produces more detailed, thorough analysis. This is called **persona prompting** — the LLM adopts the described persona's knowledge and approach.

The last line ("You always run multiple targeted searches") is a behavioral instruction disguised as backstory. It reminds the agent to be thorough.

```yaml
  llm: openai/gpt-4o
```
**Line 15:** A metadata field indicating which LLM to use. This is **informational only** in the current codebase — the actual model is hardcoded in `graph.py` (`_openai("gpt-4o")`). Including it here documents the intended model for each agent, making it easy to see the full agent spec in one place.

### company_analyst

```yaml
company_analyst:
  role: >
    {industry} Company Analyst
  goal: >
    Analyze raw competitive intelligence findings and produce a detailed assessment
    that serves Engineering, Sales, and Strategy teams at {company}. Identify product
    capability gaps, R&D investment patterns, go-to-market shifts, and technical
    differentiation across {competitors}.
  backstory: >
    You are a sharp business and technical analyst with deep experience in the {industry}
    sector. You understand both the business and engineering sides — you can read a
    product changelog and infer strategic intent, spot R&D spending patterns that signal
    future product direction, and identify sales positioning shifts from pricing and
    packaging changes. You categorize findings specifically for Engineering (what to
    build/improve), Sales (competitive positioning and objection handling), and Strategy
    (market positioning and investment priorities).
```
**Lines 17-33:** The analyst agent. Key prompt engineering techniques:
- **Multi-audience framing**: The goal explicitly mentions three audiences (Engineering, Sales, Strategy). This forces the LLM to organize its output for different readers.
- **Concrete examples in backstory**: "read a product changelog and infer strategic intent" gives the LLM a specific example of the kind of thinking it should do. Abstract instructions ("be analytical") are less effective than concrete demonstrations.

### strategy_advisor

```yaml
strategy_advisor:
  role: >
    Strategic Intelligence Advisor
  goal: >
    Synthesize competitive analysis into actionable strategic recommendations for
    {company}'s Engineering, Sales, and Strategy leadership. Recommendations must be
    specific enough that each team can act on them in their next planning cycle.
  backstory: >
    You are a former strategy consultant who now advises companies on competitive
    positioning in the {industry} sector. You combine market analysis with strategic
    frameworks to deliver recommendations segmented by function — telling Engineering
    where to invest R&D, Sales how to reposition against competitors, and Strategy
    leadership where to place long-term bets. You back every recommendation with
    specific competitive evidence.
```
**Lines 35-49:** The strategy advisor. Notice the **escalating specificity** across agents:
- Scanner: "find things" (broad collection)
- Analyst: "categorize and assess" (structured analysis)
- Advisor: "recommend specific actions" (actionable output)

Each agent builds on the previous one's work, getting more specific and action-oriented.

### report_writer

```yaml
report_writer:
  role: >
    Intelligence Briefing Writer
  goal: >
    Produce a comprehensive, executive-ready weekly competitive intelligence briefing
    for {company} that Engineering VPs, Sales leadership, and C-suite can each find
    actionable in their domain.
  backstory: >
    You are an experienced business writer who specializes in detailed, high-impact
    intelligence reports. You structure information with clear sections for different
    audiences — an executive summary for the C-suite, a product and R&D section for
    Engineering, a competitive positioning section for Sales, and strategic recommendations
    for leadership. You include specific details, data points, and evidence rather than
    vague summaries.
```
**Lines 51-65:** The writer agent. The backstory emphasizes **structure and specificity**. The instruction "Include specific details, data points, and evidence rather than vague summaries" is fighting against the LLM's natural tendency to generalize. Without this instruction, you'd get vague statements like "competitors are investing heavily in R&D." With it, you get "Parker Hannifin increased R&D spending by 12% in Q3, focusing on electrification."

### quality_evaluator

```yaml
quality_evaluator:
  role: >
    Competitive Intelligence Quality Evaluator
  goal: >
    Evaluate the quality of competitive analysis and strategic recommendations
    produced for {company} in the {industry} industry. Ensure deliverables meet
    the required structure, specificity, and evidence standards before they are
    compiled into the final briefing.
  backstory: >
    You are a rigorous quality assurance specialist for competitive intelligence
    deliverables. You have reviewed hundreds of competitive briefings and know
    the difference between a report that drives executive action and one that
    gets ignored. You check for structural completeness, specificity, evidence
    backing, and actionability. You are strict but fair.
```
The evaluator agent. Two important design choices here:

1. **"Strict but fair"**: This calibration instruction prevents the evaluator from being either too lenient (passing everything) or too harsh (failing everything and causing infinite retry loops). Without it, you'd need to tune the rubric much more carefully.

2. **Same model tier as the nodes it judges**: The evaluator uses Claude Sonnet — the same caliber as the analyze and recommend nodes. Using a weaker model to judge a stronger model's output would be unreliable. Using a stronger model would work but costs more. Same-tier evaluation is the pragmatic middle ground.

This agent is reused by both the main pipeline evaluator (`evaluate_quality` task) and the annual report evaluator (`evaluate_annual_report` task). One persona, two rubrics.

### annual_report_analyst

```yaml
annual_report_analyst:
  role: >
    {industry} Financial & Strategic Intelligence Analyst
  goal: >
    Extract deep competitive intelligence from annual reports, SEC filings,
    investor materials, LinkedIn profiles, and customer review sites for
    {company}'s competitors ({competitors}) in the {industry} industry.
  backstory: >
    You are an expert financial and strategic intelligence analyst who
    specializes in extracting actionable insights from annual reports,
    10-K filings, investor presentations, and other public disclosures.
    When annual reports or SEC filings are not available (e.g. private
    companies), you are resourceful — you mine the competitor's own
    website for product details, company information, press releases,
    and case studies. You never pad a report with generic observations —
    every point you make is backed by a specific fact, name, or number.
```
The annual report analyst. The backstory handles a practical reality: **not all competitors are public companies**. If the LLM can't find SEC filings, it needs to know that mining the competitor's website, LinkedIn, and trade press is an acceptable alternative — not a failure. The instruction to "never pad with generic observations" directly fights the LLM's tendency to fill in gaps with vague industry commentary when real data is scarce.

---

## tasks.yaml — Task Definitions

Each task has `description` (what to do), `expected_output` (what the result should look like), and metadata (`agent`, `context`, `output_file`).

### scan_competitor

```yaml
scan_competitor:
  description: >
    Search the web thoroughly for recent competitive activity about {competitor}
    in the {industry} industry as of {current_date}. Run multiple targeted searches
    covering different angles. Focus on:
    - New product releases, feature updates, version launches, and product roadmap announcements
    - R&D investments, research papers, patent filings, and technical blog posts
    - Engineering team growth, key technical hires, and talent acquisitions
    - Pricing changes, new packaging/tiers, or business model shifts
    - Partnerships, integrations, ecosystem plays, and channel strategy changes
    - Acquisitions, mergers, funding rounds, and financial performance indicators
    - Conference talks, demos, and technical previews of upcoming capabilities
    - Customer wins, case studies, and go-to-market messaging changes
    - Analyst reports, market share data, and industry rankings
```
**Lines 1-15:** The task description is exhaustively specific. Each bullet point is a **category of intelligence** to look for. This serves as a checklist for the LLM — it systematically addresses each category rather than focusing on whatever it finds first. The `{current_date}` placeholder ensures the agent knows what "recent" means.

```yaml
    Gather as much relevant, recent intelligence as possible about {competitor}
    relative to {company}. Be thorough — run at least 2-3 searches with different
    search terms to cover product, business, and technical angles.
```
**Lines 16-18:** Explicit instruction to be thorough and use multiple searches. Even though the search queries are hardcoded in `graph.py`, this instruction still helps the LLM understand the breadth of coverage expected when it summarizes the results.

```yaml
  expected_output: >
    A comprehensive structured list of competitive intelligence findings for
    {competitor}. Include:
    - Product & Technology: releases, features, technical capabilities, R&D signals
    - Business & Strategy: pricing, partnerships, M&A, funding, go-to-market changes
    - Talent & Organization: key hires, team growth, leadership changes
    Each finding should include the source URL, date, and a 2-3 sentence summary
    of why it matters. Include at least 5-8 findings where available.
```
**Lines 19-26:** The expected output definition. This is like giving the LLM a **template** to fill in. By specifying the structure (three categories, source URL, date, summary), you get consistently formatted output that the downstream analyst node can easily parse. "At least 5-8 findings" sets a minimum bar so the LLM doesn't give you two bullet points and call it done.

```yaml
  agent: trend_scanner
```
**Line 27:** Metadata linking this task to its agent. In the current code, this mapping is done in `graph.py` by which agent prompt the node function loads. This field documents the intended pairing.

### analyze_findings

```yaml
analyze_findings:
  description: >
    Analyze the raw competitive intelligence gathered about {company}'s competitors
    ({competitors}) in the {industry} industry. Produce analysis segmented for three
    audiences:

    FOR ENGINEERING LEADERSHIP:
    - Product capability gaps: where competitors have shipped features {company} lacks
    - Technology bets: what technical approaches competitors are investing in
    ...

    FOR SALES LEADERSHIP:
    - Competitive positioning shifts: how competitors are changing their messaging
    - Pricing and packaging changes that affect deal competitiveness
    ...

    FOR STRATEGY LEADERSHIP:
    - Market trend patterns across all competitors
    - Potential threats ranked by likelihood and impact
    ...

    Categorize all findings by urgency: immediate (act this week), short-term (this quarter),
    and long-term (6-12 months).
```
**Lines 28-56:** The analysis task. Two important prompt patterns here:

1. **Audience segmentation**: By explicitly defining three audiences with specific sub-categories, the LLM produces structured output that's actually useful to different teams. Without this, you'd get a wall of text that nobody can quickly scan.

2. **Urgency classification**: The instruction to categorize by urgency (immediate/short-term/long-term) adds a **decision-making dimension**. It's not enough to know what competitors are doing — leaders need to know what requires action *now* vs what to monitor.

### strategic_recommendations

```yaml
strategic_recommendations:
  description: >
    Based on the competitive analysis for {company} in the {industry} industry,
    develop strategic recommendations organized by business function:

    ENGINEERING RECOMMENDATIONS:
    - R&D investment priorities: where to increase/decrease engineering investment
    ...

    SALES RECOMMENDATIONS:
    - Competitive battle card updates: key talking points against each competitor
    ...

    STRATEGY RECOMMENDATIONS:
    - Market positioning adjustments
    ...

    Each recommendation must cite specific competitive evidence and include:
    expected impact (high/medium/low), implementation difficulty, and suggested
    timeline. Prioritize by impact and feasibility.
```
**Lines 64-95:** The recommendations task. The key instruction is "Each recommendation must cite specific competitive evidence and include: expected impact, implementation difficulty, and suggested timeline." This forces the LLM to make each recommendation **actionable and justified** rather than generic advice like "invest in R&D."

```yaml
  context:
    - analyze_findings
```
**Lines 96-97:** Metadata indicating this task depends on the analysis task's output. In the current code, this dependency is enforced by the graph edges (analyze -> recommend). This field documents the data flow.

### write_briefing

```yaml
write_briefing:
  description: >
    Compile all competitive intelligence, analysis, and strategic recommendations
    into a comprehensive weekly briefing document for {company} leadership.
    The briefing must include these sections:

    1. EXECUTIVE SUMMARY
    2. PRODUCT & TECHNOLOGY LANDSCAPE
    3. MARKET & BUSINESS INTELLIGENCE
    4. COMPETITOR DEEP DIVES
    5. THREAT ASSESSMENT
    6. STRATEGIC RECOMMENDATIONS
    7. WATCH LIST
    ...

    Format as clean, detailed markdown. Date the report as {current_date}.
    Be specific and detailed — include company names, product names, dates, and
    data points rather than vague generalizations.
```
**Lines 99-142:** The briefing task defines a 7-section report structure. This is essentially a **document template** in prose form. The LLM follows this structure almost exactly, producing a professional-looking report every time.

The instruction "Be specific and detailed — include company names, product names, dates, and data points rather than vague generalizations" is repeated from the agent backstory because it's that important. LLMs tend toward vagueness; repetition in prompts reinforces specificity.

```yaml
  expected_output: >
    A professional, comprehensive competitive intelligence briefing (2000+ words)
    in markdown format with all seven sections above. Must be specific, evidence-based,
    and actionable for Engineering, Sales, and Strategy teams.
    Formatted as markdown without '```'.
```
**Lines 143-147:** Sets expectations: 2000+ words (prevents too-short reports), markdown format, and the crucial "without '```'" instruction — without this, the LLM might wrap the entire output in a code block, which would break the markdown rendering in Gradio.

```yaml
  output_file: output/briefing.md
```
**Line 152:** Metadata indicating where the output is saved. In the current code, this is handled by `graph.py` (line 213). This field documents the intended behavior.

### evaluate_quality

```yaml
evaluate_quality:
  description: >
    You are evaluating two competitive intelligence deliverables produced for
    {company} in the {industry} industry (competitors: {competitors}).

    === ANALYSIS RUBRIC ===
    1. Three clearly labeled audience sections: Engineering, Sales, and Strategy
    2. Urgency ratings (immediate / short-term / long-term) on findings
    3. Specific evidence from the scan (competitor names, product names, dates)
    4. A cross-cutting "Key Patterns" section with 3-5 significant trends

    === RECOMMENDATIONS RUBRIC ===
    1. 12-20 recommendations organized by function
    2. Each recommendation cites specific competitive evidence
    3. Each recommendation has impact rating, difficulty rating, and timeline
    4. A "Top 5 Priorities" summary at the top

    === DELIVERABLES TO EVALUATE ===
    ANALYSIS: {analysis}
    RECOMMENDATIONS: {recommendations}
  expected_output: >
    A JSON object with "evaluation_result" and "evaluation_feedback" keys.
```

The evaluator task for the main pipeline. Notice how the rubric criteria **directly mirror** the requirements in `analyze_findings` and `strategic_recommendations`. This is intentional — the evaluator checks whether the upstream nodes produced what their task descriptions asked for. If the analysis task says "include urgency ratings" and the evaluator rubric checks for urgency ratings, the loop is closed.

The `{analysis}` and `{recommendations}` placeholders inject the actual deliverables into the evaluation prompt. The evaluator sees both at once so it can check cross-cutting quality (e.g., do the recommendations reference evidence from the analysis?).

The expected output is JSON with a 4-way verdict: `"pass"`, `"fail_analysis"`, `"fail_recommendations"`, or `"fail_both"`. This granularity lets the retry routing logic fix only what's broken.

### scan_annual_report

```yaml
scan_annual_report:
  description: >
    Conduct a deep-dive investigation into {competitor} in the {industry} industry.

    SOURCE PRIORITY: annual reports, SEC filings, company website, LinkedIn,
    trade press, customer reviews, patent databases.

    DISAMBIGUATION: {competitor} is a {industry} company competing with {company}.
    Ignore search results about unrelated companies with similar names.

    15 sections: Company Overview, Product Portfolio, Pricing, Customers,
    Go-to-Market, R&D, Financials, Team, Customer Sentiment, Market Position,
    M&A, Geographic Presence, Patents, Regulatory Risks, Strategic Assessment.

    QUALITY RULES:
    - Every claim must cite a source URL inline.
    - Do not repeat the same fact in multiple sections.
    - If a section has no data, write one sentence and move on.
    - Do not use the same source more than 3 times.
```

The annual report task is the longest in the codebase. Key prompt engineering patterns:

1. **Source priority ordering**: Tells the LLM which sources to prefer (annual reports > SEC filings > company website > ...). Without this, the LLM might over-rely on generic news articles instead of primary sources.

2. **Disambiguation instruction**: Company names can be ambiguous across industries. The instruction to ignore unrelated companies prevents the LLM from confusing, say, "Parker" the pen company with "Parker Hannifin" the hydraulics company.

3. **Anti-padding rules**: "If a section has no data, write one sentence and move on" prevents the LLM from generating paragraphs of generic industry commentary to fill empty sections. It's better to have an honest "No data available" than a padded section that looks comprehensive but says nothing.

4. **Source diversity cap**: "Do not use the same source more than 3 times" forces the LLM to draw from multiple sources rather than just summarizing one article repeatedly.

### evaluate_annual_report

```yaml
evaluate_annual_report:
  description: >
    You are evaluating a deep-dive annual report produced for {competitor}.

    === RUBRIC ===
    1. All 15 sections present
    2. Specific facts, data points, and named examples — not generic observations
    3. Source URLs cited inline
    4. No repetition across sections
    5. Thin sections stated briefly, not padded
    6. Diverse sources (not the same source repeated 3+ times)
    7. 2000+ words with depth concentrated where real data exists
  expected_output: >
    A JSON object with "evaluation_result" ("pass" or "fail") and
    "evaluation_feedback" with specific gaps to fix.
```

The annual report evaluator task. Compared to `evaluate_quality`, this has a simpler verdict: just `"pass"` or `"fail"` (not the 4-way split). This makes sense because there's only one deliverable to evaluate (one competitor's report), not two separate deliverables.

The rubric criteria are derived directly from the `scan_annual_report` quality rules — again, the evaluator checks what the upstream task asked for. This creates a closed feedback loop: if the report violates a quality rule (e.g., "same source used 5 times"), the evaluator catches it and the retry includes specific feedback ("need more diverse sourcing").

This task reuses the `quality_evaluator` agent persona — same evaluator personality, different rubric. This demonstrates the value of separating agents (who) from tasks (what): one agent can perform multiple evaluation tasks with different criteria.

---

## Why YAML for Prompts?

1. **Separation of concerns**: Prompt engineering and Python engineering are different skills. YAML lets prompt engineers work without touching code.
2. **Readability**: Multi-line prompts in YAML are cleaner than Python triple-quoted strings with `\n` and `f"..."` everywhere.
3. **Versioning**: You can track prompt changes in git diffs more clearly when they're in dedicated files.
4. **Reusability**: The same YAML structure could be loaded by different graph implementations or even different frameworks.
