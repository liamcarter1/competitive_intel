# YAML Config Files — Line-by-Line Explanation

The YAML config files are where **all the prompt engineering lives**. They define who each agent is (agents.yaml) and what each agent does (tasks.yaml). This separation means you can tune agent behavior by editing YAML — no Python changes needed.

The configs cover two pipelines: the **news monitor** (news_digest_curator agent + compile_news_digest task) and the **annual report deep-dive** (annual_report_analyst agent + scan_annual_report/evaluate_annual_report tasks). A shared quality_evaluator agent handles evaluation for the annual report pipeline.

---

## agents.yaml — Agent Personas

Each agent has three fields: `role`, `goal`, and `backstory`. Together, these form the system prompt that tells the LLM who it is and how to behave.

### news_digest_curator

```yaml
news_digest_curator:
  role: >
    {industry} Competitive News Digest Curator
```
The `>` in YAML is a **folded block scalar** — it takes the following indented lines and joins them into a single line, replacing newlines with spaces. So this becomes the string `"{industry} Competitive News Digest Curator"`. The `{industry}` placeholder gets replaced at runtime by Python's `.format()` method (e.g., becomes `"Hydraulics & Mobile Machinery Competitive News Digest Curator"`).

```yaml
  goal: >
    Deduplicate, categorize, and format raw competitive news search results into a
    clean, scannable news digest organized by competitor for {company}'s leadership.
```
The goal is concise and action-oriented: three verbs (deduplicate, categorize, format) that describe exactly what the curator does. Notice this is a **curation** goal, not an analysis goal. The old briefing pipeline had four agents (scanner, analyst, advisor, writer) that each built on the previous one's work with increasingly analytical output. The news digest curator deliberately stops short of analysis — it organizes raw search results without adding interpretation. This is a design choice: curation is cheaper (one LLM call vs four) and more reliable (no hallucinated insights), at the cost of leaving the "so what?" to the human reader.

```yaml
  backstory: >
    You are a news editor who specializes in competitive intelligence digests. You
    receive raw search results from multiple queries per competitor and your job is
    to eliminate duplicates, assign each item to the right category, discard irrelevant
    or off-topic results, and produce a clean markdown digest that an executive can
    scan in 5 minutes. You never add information beyond what the search results contain —
    you are a curator, not an analyst. You preserve original dates, URLs, and source
    attributions exactly as provided.
```
The backstory establishes a **news editor persona** — someone who curates and organizes rather than creates. This is crucial prompt engineering for a curation task. Two key behavioral guardrails are embedded in the backstory:

1. **"You never add information beyond what the search results contain"**: This is the anti-hallucination instruction. Without it, GPT-4o-mini would happily embellish search result summaries with its own knowledge, making the digest unreliable. By explicitly stating the boundary, we get a faithful pass-through of the raw data.

2. **"You preserve original dates, URLs, and source attributions exactly as provided"**: URLs and dates are the most hallucination-prone data types for LLMs. This instruction tells the model to treat them as sacred — copy them verbatim, don't reconstruct them from memory.

The "5 minutes" detail is a subtle but effective framing device. It tells the LLM the output should be scannable — short summaries, clear structure, no walls of text.

```yaml
  llm: openai/gpt-4o-mini
```
A metadata field indicating which LLM to use. This is **informational only** in the current codebase — the actual model is hardcoded in `graph.py`. Including it here documents the intended model for each agent, making it easy to see the full agent spec in one place. GPT-4o-mini is a good fit for this agent because the task is structuring and deduplicating, not deep reasoning — it's fast and cheap for what is essentially an editorial formatting job.

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

This agent is used by the annual report evaluator (`evaluate_annual_report` task). The persona is generic enough to evaluate different types of deliverables — if future pipelines need quality gates, the same agent can be paired with new task rubrics.

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

### compile_news_digest

```yaml
compile_news_digest:
  description: >
    You are a competitive news curator for {company} in the {industry} industry.
    Below are raw search results (tagged [NEWS (date)] or [WEB]) for each
    competitor: {competitors}.

    Today is {current_date}. Time window: {time_window_label}.
```
The opening establishes context: who we are (a news curator for a specific company), what we're working with (raw tagged search results), and the time frame. The `{time_window_label}` placeholder lets the pipeline pass in a human-readable description of the search window (e.g., "past 30 days"), which the LLM can use when deciding whether results are relevant.

```yaml
    Your job:
    1. DEDUPLICATE: Many queries return the same article. Keep only one entry per
       unique news story (match on URL or headline). Prefer the version with the
       most detail.
    2. DISCARD OFF-TOPIC: Remove results about unrelated companies that share a
       similar name but operate in a different industry. Use the disambiguation
       context provided for each competitor.
    3. CATEGORIZE: Assign each item to exactly one category per competitor:
       - Product & Technology
       - Business & Financial
       - People & Organization
       - Market & Customers
       - Regulatory & Compliance
       - Other
    4. SORT: Within each category, sort by date (newest first). For [WEB] items
       without a date, place them after dated items.
    5. FORMAT: Use this exact format for each item:
       - **[Date]** | Summary sentence (1-2 sentences max). [Source](URL)
       For [WEB] items without a date, omit the date:
       - Summary sentence. [Source](URL)
    6. If a competitor has no results in a category, write "(none this period)".
    7. NEVER add information beyond what the search results contain. Do not
       analyze, editorialize, or speculate. You are a curator, not an analyst.
```
This is the heart of the task — a **numbered step-by-step procedure**. Several prompt engineering patterns are at work:

1. **Explicit deduplication logic**: Raw Serper results from 15+ queries per competitor inevitably overlap. The instruction to "match on URL or headline" gives the LLM a concrete deduplication rule rather than a vague "remove duplicates." The preference for "the version with the most detail" handles the case where the same article appears in multiple queries with different snippet lengths.

2. **Disambiguation by reference**: Instead of repeating the full disambiguation rules, the task says "use the disambiguation context provided for each competitor." This works because `graph.py` injects the LLM-generated context sentence from `_disambiguate_competitor()` directly into the raw results block. The prompt trusts the LLM to cross-reference.

3. **Fixed category taxonomy**: Six categories (Product & Technology, Business & Financial, People & Organization, Market & Customers, Regulatory & Compliance, Other) provide just enough granularity to be useful without overwhelming. The "Other" bucket is a catch-all that prevents the LLM from forcing results into ill-fitting categories. Each item goes to **exactly one** category — no duplication across sections.

4. **Exact formatting template**: The `**[Date]** | Summary. [Source](URL)` format is precise enough that the LLM reproduces it nearly verbatim. Giving an exact template produces far more consistent formatting than describing the desired format in prose.

5. **The curator guardrail**: Repeating "You are a curator, not an analyst" from the backstory reinforces the boundary. Without this, GPT-4o-mini tends to editorialize — adding phrases like "This could signal a strategic shift toward..." when it should simply report what the article says.

```yaml
    RAW SEARCH RESULTS:
    {raw_results}
```
The `{raw_results}` placeholder is where the actual search data gets injected. At runtime, `graph.py` formats the raw Serper results (with their `[NEWS (date)]` or `[WEB]` tags, URLs, and snippets) into a single text block and inserts it here. This is a **data injection pattern** — the task description is a template, and the variable data fills the placeholder.

```yaml
  expected_output: >
    A clean markdown digest with this structure:

    # Latest News Monitor — Competitive Intelligence
    **Company:** {company} | **Industry:** {industry} | **Date:** {current_date}
    **Time window:** {time_window_label} | **Competitors:** {competitors}

    ---

    ## [Competitor Name]

    ### Product & Technology
    - **[Date]** | Summary. [Source](URL)

    ### Business & Financial
    - **[Date]** | Summary. [Source](URL)

    ### People & Organization
    - (none this period)

    ### Market & Customers
    - **[Date]** | Summary. [Source](URL)

    ### Regulatory & Compliance
    - (none this period)

    ### Other
    - (none this period)

    ---

    (Repeat for each competitor)

    Rules:
    - No duplicate stories (same URL or same headline = one entry)
    - Every item has a clickable [Source](URL) — never invent URLs
    - Dates from [NEWS] tags must be preserved exactly
    - No analysis, commentary, or recommendations — just curated news
    - Formatted as clean markdown without code fences
```
The expected output is essentially a **complete structural template**. This is more prescriptive than the old briefing's expected output — because the news digest has a rigid structure (same 6 categories for every competitor), providing the exact layout as a template produces highly consistent output. The LLM fills in the data but follows the skeleton exactly.

The trailing "Rules" section restates the key constraints. This redundancy is intentional: LLMs process long prompts, and critical rules stated only once at the top may lose influence by the time the model generates the end of its output. Restating them at the bottom of the expected output section keeps them fresh in the model's attention window.

The "without code fences" instruction prevents the LLM from wrapping the markdown in triple backticks, which would break rendering in Gradio.

```yaml
  agent: news_digest_curator
```
Metadata linking this task to its agent. In the current code, this mapping is done in `graph.py` by which agent prompt the node function loads. This field documents the intended pairing.

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

2. **Disambiguation instruction**: Company names can be ambiguous across industries. The instruction to ignore unrelated companies prevents the LLM from confusing, say, "Parker" the pen company with "Parker Hannifin" the hydraulics company. As with the briefing scan, `graph.py` supplements this with an LLM-generated context sentence from `_disambiguate_competitor()` that replaces the generic text with a precise company identity.

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
