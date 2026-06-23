"""Site-plan POC: zoning PDF + plot + program constraints -> drawn footprint.

ponytail: the "RAG pipeline" for a single zoning doc is just attaching the PDF
as an API document block. One PDF fits in context, so no chunking / embeddings /
vector store. Swap to real retrieval only if you ever feed a whole municipal code
library instead of one district's rules.
"""
import argparse
import base64
import json
import os
import sys

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPolygon, Rectangle

MODEL = "claude-opus-4-8"
GEMINI_MODEL = "gemini-2.5-pro"

# Claude returns the footprint in this shape; json_schema makes it non-optional.
SCHEMA = {
    "type": "object",
    "properties": {
        "setbacks_ft": {
            "type": "object",
            "properties": {
                "front": {"type": "number"}, "rear": {"type": "number"},
                "left": {"type": "number"}, "right": {"type": "number"},
            },
            "required": ["front", "rear", "left", "right"],
            "additionalProperties": False,
        },
        "max_height_ft": {"type": "number"},
        "max_lot_coverage": {"type": "number"},  # fraction 0..1
        "buildings": {  # footprint rectangles, feet, in the lot coordinate frame
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"}, "y": {"type": "number"},
                    "width": {"type": "number"}, "height": {"type": "number"},
                    "label": {"type": "string"},
                },
                "required": ["x", "y", "width", "height", "label"],
                "additionalProperties": False,
            },
        },
        "stories": {"type": "integer"},
        "building_height_ft": {"type": "number"},
        "compliance_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["setbacks_ft", "max_height_ft", "max_lot_coverage", "buildings",
                 "stories", "building_height_ft", "compliance_notes"],
    "additionalProperties": False,
}

PROMPT = """You are a site planner producing a compliant footprint for one apartment building.

Coordinate system: feet. Origin (0,0) at the FRONT-LEFT corner of the lot. +x runs
along the street frontage to the right; +y runs from the street toward the rear.
The front setback is measured from y=0, rear from the max-y edge, left from x=0,
right from the max-x edge.

Lot polygon vertices (feet): {plot}

Program constraints: {constraints}

Using the attached zoning code:
1. Read the front/rear/side setbacks, max building height, and max lot coverage.
2. Place building footprint rectangle(s) inside the buildable area (lot minus setbacks).
3. Size the footprint so each floor holds units*unit_size_sqft/stories of area, then
   arrange the rectangle(s) to match the requested shape description.
4. Verify: footprint within setbacks, total footprint/lot area <= max coverage,
   stories*~11ft <= max height. List any issues in compliance_notes (say "compliant"
   if all pass).

Return ONLY the structured object."""


def zoning_block(path):
    """PDF -> base64 document block; .txt -> text block. Both go to the model as-is."""
    if path.lower().endswith(".pdf"):
        data = base64.standard_b64encode(open(path, "rb").read()).decode()
        return {"type": "document", "title": "Zoning Code",
                "source": {"type": "base64", "media_type": "application/pdf", "data": data}}
    return {"type": "document", "title": "Zoning Code",
            "source": {"type": "text", "media_type": "text/plain", "data": open(path).read()}}


def plan(zoning_path, plot, constraints):
    client = anthropic.Anthropic()
    text = PROMPT.format(plot=json.dumps(plot), constraints=json.dumps(constraints))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [zoning_block(zoning_path),
                                               {"type": "text", "text": text}]}],
    )
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def plan_gemini(zoning_path, plot, constraints):
    """Separate Gemini route. Uses GEMINI_API_KEY (or GOOGLE_API_KEY)."""
    from google import genai
    from google.genai import types
    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env
    if zoning_path.lower().endswith(".pdf"):
        zoning = types.Part.from_bytes(data=open(zoning_path, "rb").read(),
                                       mime_type="application/pdf")
    else:
        zoning = open(zoning_path).read()
    # ponytail: response_mime_type JSON gives valid JSON without translating SCHEMA
    # into Gemini's schema dialect; the prompt carries the shape. Add response_schema
    # only if the model drifts from it.
    text = (PROMPT.format(plot=json.dumps(plot), constraints=json.dumps(constraints))
            + "\n\nReturn JSON matching this schema:\n" + json.dumps(SCHEMA))
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[zoning, text],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(resp.text)


def bbox(plot):
    xs = [p[0] for p in plot]; ys = [p[1] for p in plot]
    return min(xs), min(ys), max(xs), max(ys)


def within_setbacks(buildings, plot, s):
    """True if every footprint rect sits inside the lot's setback inset (rectangular lot)."""
    x0, y0, x1, y1 = bbox(plot)
    ix0, iy0, ix1, iy1 = x0 + s["left"], y0 + s["front"], x1 - s["right"], y1 - s["rear"]
    return all(b["x"] >= ix0 - 1e-6 and b["y"] >= iy0 - 1e-6
               and b["x"] + b["width"] <= ix1 + 1e-6
               and b["y"] + b["height"] <= iy1 + 1e-6 for b in buildings)


def draw(plot, r, out="site_plan.png"):
    x0, y0, x1, y1 = bbox(plot)
    s = r["setbacks_ft"]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.add_patch(MplPolygon(plot, closed=True, fill=False, edgecolor="black", lw=2))
    # ponytail: setback inset drawn from the lot bbox — fine for the rectangular
    # lots this POC handles; an irregular polygon would need true offsetting.
    ax.add_patch(Rectangle((x0 + s["left"], y0 + s["front"]),
                           (x1 - s["right"]) - (x0 + s["left"]),
                           (y1 - s["rear"]) - (y0 + s["front"]),
                           fill=False, edgecolor="gray", ls="--", lw=1, label="setback line"))
    for b in r["buildings"]:
        ax.add_patch(Rectangle((b["x"], b["y"]), b["width"], b["height"],
                               facecolor="#9ecae1", edgecolor="#08519c", lw=1.5, alpha=0.8))
        ax.text(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, b["label"],
                ha="center", va="center", fontsize=9)
    ax.set_xlim(x0 - 15, x1 + 15); ax.set_ylim(y0 - 15, y1 + 15)
    ax.set_aspect("equal"); ax.set_xlabel("feet (frontage)"); ax.set_ylabel("feet (depth)")
    ok = within_setbacks(r["buildings"], plot, s)
    ax.set_title(f"Site Plan — {r['stories']} stories, {r['building_height_ft']:.0f} ft  "
                 f"[{'within setbacks' if ok else 'SETBACK VIOLATION'}]")
    fig.text(0.02, 0.02, "Compliance: " + "; ".join(r["compliance_notes"]),
             fontsize=8, wrap=True)
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


SAMPLE_ZONING = """RESIDENTIAL DISTRICT R-3 ZONING CODE

Sec. 3.1 Setbacks. Front: 25 ft. Rear: 20 ft. Each side: 10 ft.
Sec. 3.2 Height. Maximum building height: 45 ft.
Sec. 3.3 Lot coverage. Buildings shall not cover more than 40% of lot area.
Sec. 3.4 Use. Multi-family apartment buildings are permitted.
"""
SAMPLE_PLOT = [[0, 0], [120, 0], [120, 150], [0, 150]]  # 120ft x 150ft = 18,000 sqft
SAMPLE_CONSTRAINTS = {"units": 12, "stories": 3, "unit_size_sqft": 900,
                      "shape": "single rectangular bar parallel to the street frontage"}


def make_sample():
    with PdfPages("sample_zoning.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.92, SAMPLE_ZONING, va="top", fontsize=12, family="monospace")
        pdf.savefig(fig); plt.close(fig)
    json.dump(SAMPLE_PLOT, open("sample_plot.json", "w"))
    json.dump(SAMPLE_CONSTRAINTS, open("sample_constraints.json", "w"), indent=2)
    print("wrote sample_zoning.pdf, sample_plot.json, sample_constraints.json")


def _check():
    """Offline geometry self-check — no API call."""
    s = {"front": 25, "rear": 20, "left": 10, "right": 10}
    good = [{"x": 10, "y": 25, "width": 100, "height": 60, "label": "ok"}]
    bad = [{"x": 5, "y": 25, "width": 100, "height": 60, "label": "x<left"}]
    assert within_setbacks(good, SAMPLE_PLOT, s)
    assert not within_setbacks(bad, SAMPLE_PLOT, s)
    assert bbox(SAMPLE_PLOT) == (0, 0, 120, 150)
    print("checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoning", default="sample_zoning.pdf")
    ap.add_argument("--plot", default="sample_plot.json")
    ap.add_argument("--constraints", default="sample_constraints.json")
    ap.add_argument("--out", default="site_plan.png")
    ap.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic")
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return _check()
    if a.make_sample:
        return make_sample()
    plot = json.load(open(a.plot))
    constraints = json.load(open(a.constraints))
    result = {"anthropic": plan, "gemini": plan_gemini}[a.provider](a.zoning, plot, constraints)
    print(json.dumps(result, indent=2))
    draw(plot, result, a.out)


if __name__ == "__main__":
    sys.exit(main())