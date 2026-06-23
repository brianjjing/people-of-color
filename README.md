# Architect site-plan POC

One Claude call turns a zoning PDF + a numeric plot + program constraints into a
drawn, compliance-checked apartment footprint.

```bash
pip install -r requirements.txt
python site_plan.py --make-sample   # writes sample_zoning.pdf, *.json
python site_plan.py --check         # offline geometry self-check (no API)

# Anthropic route (default):
export ANTHROPIC_API_KEY=sk-ant-...
python site_plan.py                 # -> site_plan.png + JSON to stdout

# Gemini route (separate):
export GEMINI_API_KEY=...
python site_plan.py --provider gemini
```

Real inputs:

```bash
python site_plan.py --zoning code.pdf --plot lot.json --constraints program.json
```

- `lot.json`: lot polygon as `[[x,y], ...]` vertices in feet, origin front-left.
- `program.json`: `{"units", "stories", "unit_size_sqft", "shape"}`.

## How it works

`zoning_block()` attaches the PDF natively (base64 document block) — that *is* the
RAG step for a single doc. Claude reads the setback/height/coverage rules and
returns a footprint via a `json_schema` structured output; matplotlib draws the
lot, setback line, and building rectangles and flags setback violations.

Skipped: vector store / chunking / embeddings (one PDF fits in context — add them
only for a whole code library), true polygon setback-offsetting (bbox inset assumes
a rectangular lot), multi-building / parking / unit-level layout.
# people-of-color
