#!/usr/bin/env python3
"""
Positive EXFOLIATE variant prompt pack.

Creates a CSV of 64 scene definitions (voice + two image prompts) in:
  assets_exfoliate_positive/prompts_positive.csv

No generation is run; this only materializes the prompts.
"""

import csv
from pathlib import Path

OUTPUT_DIR = Path("assets_exfoliate_positive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "prompts_positive.csv"

# 64 body parts / regions
AREAS = [
    "left shoulder", "right shoulder", "upper back", "lower back", "chest", "abdomen", "left thigh", "right thigh",
    "left shin", "right shin", "left calf", "right calf", "neck", "nape", "scalp", "left forearm", "right forearm",
    "left upper arm", "right upper arm", "left hand", "right hand", "left palm", "right palm", "left fingers", "right fingers",
    "hip", "waist", "lower leg", "ankle", "foot", "toes", "inner thigh", "outer thigh", "buttock", "lower back center",
    "upper chest", "sternum area", "left ribcage", "right ribcage", "elbow", "knee", "shoulder blade left", "shoulder blade right",
    "calf left", "calf right", "heel", "arch of foot", "groin area", "upper back center", "lower abdomen", "side torso left",
    "side torso right", "wrist", "thumb area", "index finger", "ring finger", "back of hand", "collarbone", "behind knee",
    "back of neck", "jawline", "temple", "solar plexus", "hipbone"
]

# 64 obscure or uncommon afflictions (plain English, no Latin)
AFFLICTIONS = [
    "phantom itching", "chill bumps rash", "nettle sting scars", "wire-brush dermatitis",
    "coal dust staining", "patchy scaling", "old rope burn marks", "windburn striations",
    "mild trench foot", "chalky dry flare", "pinprick bruising", "sun-lamp mottling",
    "barber's rash", "cold snap fissures", "salt-crust scalp", "ink-stain spotting",
    "ropey muscle knots", "granular dryness", "grain dust rash", "wood splinter freckles",
    "tool-handle callus", "lime dust veil", "paper cut lattice", "dye blotches",
    "belt-line chafe", "soot shadowing", "shear-wool scratches", "ankle harness rub",
    "boot welt impressions", "silt-caked toes", "linen-rub redness", "outer seam abrasions",
    "saddle sore patch", "grindstone sheen", "radiator heat blush", "soot-smudged ribs",
    "binder strap lines", "old fall bruises", "ridge-plank scars", "hook-lift marks",
    "chalk pit dusting", "knotty bursitis look", "heel fissures", "arch strain crease",
    "mild rope burn", "soot glaze", "brick dust specks", "cinder freckles",
    "cold-room pallor", "steam-room flush", "sawtooth scrape", "old tourniquet band",
    "glove line dryness", "thumb pad polish", "index nick grid", "ring groove redness",
    "back-of-hand scuff", "collar rub line", "popliteal damp rash", "tensed neck cords",
    "jaw clench tightness", "temple pulse hue", "sunken plexus pallor", "hip-edge bruises"
]

# 16 positivity variants to rotate through
LEAD_INS = [
    "Brief note on", "Clinical check of", "Observation for", "Quick assessment of",
    "Status review for", "Focused report on", "Concise note about", "Targeted observation for",
    "Short update on", "Immediate finding for", "Rapid chart entry on", "Snapshot assessment of",
    "Current reading for", "Terse statement on", "Field note for", "On-site summary of"
]

POSITIVE_CLAUSES = [
    "On the positive side, response appears contained",
    "Brightly viewed, the condition stays localized",
    "In better light, overall resilience is evident",
    "Optimistically, progression seems halted",
    "Encouragingly, no spread beyond the noted area",
    "Constructively, tone and circulation look steady",
    "Favorable view: symptoms remain bounded",
    "Seen kindly, texture is improving",
    "Upside noted: comfort reports are neutral",
    "Silver lining: presentation is narrow in scope",
    "From a helpful angle, color remains consistent",
    "Advantageously, no new findings nearby",
    "Positively speaking, tolerance is intact",
    "Looking at the upside, skin integrity is holding",
    "Counting the upside, sensitivity is unchanged",
    "Reading the plus column, margins are clean",
    "In a hopeful frame, irritation is minimal",
    "Through a kinder lens, surface remains stable",
    "With an optimistic note, no escalation observed",
    "Noting the advantage, patient comfort unchanged",
    "Framed well, the impact stays minor",
    "Favorably, boundary remains clear",
    "With a bright view, dryness is controlled",
    "Gently noted, the site is calm",
    "Constructively, localized warmth is reducing",
    "Encouraging view: pattern is consistent",
    "Helpful perspective: texture is smoothing",
    "Advantage seen: hue is steady",
    "Positive angle: no secondary signs",
    "Upliftingly, area is manageable",
    "In better spirit, condition holds steady",
    "Kindly put, irritation is modest",
    "Supportively, only focal change present",
    "Reassuringly, scope is limited",
    "Sunnier take: recovery trajectory looks even",
    "Confidence point: discomfort not worsening",
    "Uplook: presentation remains tight",
    "Lightly stated, symptoms are contained",
    "Benefit noted: swelling is slight",
    "Promising note: boundary unchanged",
    "Comforting view: tone remains even",
    "Soft outlook: only one locus involved",
    "Positive reflection: resilience present",
    "Constructive lens: status is orderly",
    "Reassuring tone: pattern is stable",
    "Bright remark: area stays defined",
    "Hopeful remark: no upward trend",
    "Optimistic remark: margins intact",
    "Encouraging remark: condition is quiet",
    "Helpful remark: findings are limited",
    "Favorable remark: impact is narrow",
    "Uplifting remark: no spread detected",
    "Good sign: region remains focused",
    "Positive sign: tissue appears steady",
    "Helpful sign: comfort reports neutral",
    "Warm note: mild and contained",
    "Optimism: surface looks orderly",
    "Sanguine view: state is constrained",
    "Reassurance: change is slight",
    "Comfort: situation is contained",
    "Heartening note: no added complications",
    "Cheering note: boundaries hold",
    "Upbeat point: stability seen",
    "Bright point: only focal irritation",
    "Kind note: manageable presentation",
    "Encouraging point: quiet progression"
]

ACTION_CLAUSES = [
    "Keep the briefing concise and clinical.",
    "Document succinctly and proceed.",
    "Maintain a terse, professional tone.",
    "Note briefly and move to the next check.",
    "Record with precision and minimal wording.",
    "File a short observation only.",
    "Log the finding plainly.",
    "State the condition and advance.",
    "Keep remarks minimal and exact.",
    "Summarize crisply before moving on.",
    "Capture in brief, professional terms.",
    "Offer a short, measured note.",
    "State it simply and proceed.",
    "Maintain clipped, clinical phrasing.",
    "Log in sharp, economical language.",
    "Express in lean, direct wording.",
    "Keep it spare and disciplined.",
    "Use clear, minimal phrasing.",
    "Deliver only essential detail.",
    "Limit to key clinical points.",
    "Confine to one crisp statement.",
    "Stay terse and professional.",
    "Keep verbiage tightly controlled.",
    "Report with economy of words.",
    "Leave out embellishment.",
    "File a compact line item.",
    "Stick to the essentials.",
    "Mark it briefly and continue.",
    "Note it cleanly in one line.",
    "Maintain disciplined brevity.",
    "Keep diction sharp and short.",
    "Apply a fast, factual entry.",
    "Use streamlined wording.",
    "Register the detail in brief.",
    "Close with a compact remark.",
    "Retain a minimal phrasing.",
    "Keep syntax plain and short.",
    "Enter the fact without delay.",
    "State and move on.",
    "Keep the chart lean.",
    "Conclude with one concise line.",
    "Keep focus and brevity.",
    "Apply clipped wording.",
    "Stay factual and lean.",
    "Keep narration compressed.",
    "Restrict to critical detail.",
    "Note it efficiently.",
    "Stay economical with language.",
    "Finalize with terse wording.",
    "Hold to precise brevity.",
    "Maintain exact, short phrasing.",
    "Finish with minimal prose.",
    "Keep commentary lean.",
    "Use a single concise remark.",
    "Adhere to strict brevity.",
    "Retain only essential phrasing.",
    "End with a crisp note.",
    "Keep the entry tight.",
    "Close with sparse language.",
    "Conserve words and proceed.",
    "Limit to a focused statement.",
    "Deliver a trimmed observation.",
    "Summarize in one lean sentence.",
    "Finalize succinctly."
]


def make_voice(idx: int, area: str, affliction: str) -> str:
    lead = LEAD_INS[idx % len(LEAD_INS)]
    pos = POSITIVE_CLAUSES[idx % len(POSITIVE_CLAUSES)]
    act = ACTION_CLAUSES[idx % len(ACTION_CLAUSES)]
    return f"{lead} {affliction} at the {area}. {pos}. {act}"


def make_body_prompt(area: str, affliction: str) -> str:
    return (
        f"Full body photo of a strange man in homemade clothes, clearly showing {affliction} at the {area}, "
        "clinical lighting, gritty 35mm, neutral background, high detail."
    )


def make_close_prompt(area: str, affliction: str) -> str:
    return (
        f"Clinical close-up of the {area} displaying {affliction}, harsh light, macro focus, medical documentation style."
    )


def main():
    rows = []
    for i, (area, aff) in enumerate(zip(AREAS, AFFLICTIONS)):
        rows.append({
            "id": i,
            "area": area,
            "affliction": aff,
            "voice": make_voice(i, area, aff),
            "image_full_body": make_body_prompt(area, aff),
            "image_closeup": make_close_prompt(area, aff),
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} prompt rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
