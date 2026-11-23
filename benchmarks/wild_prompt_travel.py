"""RTNVE overnight prompt journey: radical ideas, zero LoRA tags, smooth changes.

This module only makes a deterministic frame-indexed conditioning PLAN.
The video remains one global AnimateDiff latent video; CLIP embeddings are
interpolated across neighbouring prompts, not rendered images crossfaded.

Prompts deliberately mix unrelated subjects, places, materials, scales and
actions. 60-140 frame dwell, randomized 20-60-frame easing. The dwell simply
means the new subject gets a chance to exist before the next metamorphosis.
"""
from __future__ import annotations
import random

ARTIST_SCENES = [
    "cloned taxis and pedestrians jammed in a New York intersection, people pushing cars, a person sitting on a bench",
    "twenty strange faces squeezed into a subway carriage, overlapping coats, limbs and blurred expressions",
    "a crowded airport hall with duplicate travellers and luggage caught inside broken glass",
    "people merging into a hundred bodies in the aisle of a bus, faces appearing out of each other's clothes",
    "four people holding each other in a cluttered flat, heads blurred into misplaced furniture",
    "a high-angle surveillance view of a busy city street, duplicated vehicles and torn newspaper buildings",
    "a packed underground train platform with faces squeezed against windows, ripped paper surfaces",
    "twenty strangers with different clothes and hair trapped inside an enormous television screen",
    "overcrowded office workers carrying fragments of damaged cars through a railway station",
    "a crowded city street with pedestrians emerging from cracked computer monitors",
]

WORLDS = [
    "an impossible supermarket built inside the ribs of a whale",
    "an abandoned swimming pool filled with tiny crowded train stations",
    "a long hotel corridor folded inside a wrecked car",
    "an airport baggage hall opening into a damp cathedral",
    "a city built from hundreds of stacked wardrobes",
    "a flooded cinema filled with railway tracks",
    "a staircase packed with miniature furnished apartments",
    "a derelict museum made from enormous pieces of paper",
    "a botanical garden growing inside a demolished office tower",
    "an old theatre overtaken by a working motorway",
    "a dense suburban estate growing out of an open book",
    "a railway tunnel lined with thousands of human faces",
    "a ruined department store suspended over a reservoir",
    "an underground car park transformed into a greenhouse",
    "a crowded kitchen taking up the interior of a passenger plane",
    "a giant elevator opening into hundreds of tiny rooms",
    "a forest of television aerials growing out of an old hospital",
    "an industrial shipyard collapsed into a small living room",
    "a busy street market on top of gigantic library shelves",
    "a ruined amusement park inside a transparent suitcase",
    "an empty football stadium made from old household furniture",
    "a concrete city wrapped around a broken mechanical clock",
    "a stone plaza filled with hundreds of overturned office chairs",
    "an underground cinema beneath an overgrown botanical station",
    "an endless tunnel of graffiti-covered doors, each hiding a crowded room",
    "a crowded classroom whose windows open into the ocean",
    "a luxury hotel built from thousands of battered washing machines",
    "a city made from cut-up old photographs on a kitchen table",
    "a demolished skyscraper inside a small glass greenhouse",
    "a crowded seaside funfair beneath a city overpass",
    "a suburban home with hundreds of impossible overlapping floors",
    "a train station whose roof is an enormous living human face",
]

INTRUDERS = [
    "a giant face assembled from traffic and apartment windows",
    "miniature commuters crawling through cracked televisions",
    "a procession of cars carried by unfamiliar animals",
    "hundreds of blurred people squeezed into a single doorway",
    "a chandelier made from moving underground carriages",
    "a crowd of strangers with photocopiers instead of heads",
    "a huge office chair crowded with miniature people",
    "houses with human limbs protruding through their walls",
    "people pushing an entire house through traffic",
    "an enormous hand pulling tiny cars from the pavement",
    "strangers with their faces replaced by shifting architecture",
    "a crowd emerging from open drawers in an antique desk",
    "automobiles carrying other smaller automobiles as passengers",
    "a wall of duplicated people with coats growing into rooms",
    "a small figure pulling an entire city along a carpet",
    "a bus containing dozens of mismatched bedrooms",
    "human figures made from paper, screens and broken glass",
    "a staircase with people turning into living household objects",
    "a mass of root-covered vehicles moving as one organism",
    "a painting of a crowd that keeps spilling into the room",
    "hundreds of paper hands pushing through concrete walls",
    "a huge transparent fruit filled with busy apartment windows",
    "an enormous eye staring through overlapping suburban houses",
    "a wave of clothes and bodies flowing over empty train seats",
]

CHANGES = [
    "shattering into glass fragments that grow into new buildings",
    "ripping apart in paper layers to reveal other crowded worlds",
    "heads and windows melting together into impossible architecture",
    "vehicles splitting into plants and reforming as people",
    "interiors unfolding through impossible new rooms",
    "the whole environment gradually turning inside out",
    "walls fracturing into dozens of connected moving figures",
    "faces and staircases swapping places in broken reflections",
    "the architecture folding like a huge torn photograph",
    "crowds compressing into objects then blooming into new rooms",
    "broken mirrors producing different unfamiliar environments",
    "everything morphing through wet paper, foliage and glass",
    "people dissolving into handwritten text and forming again",
    "streets breaking into countless small interconnected structures",
    "furniture stretching into a maze of living passages",
    "glass and concrete shattering while human bodies take their place",
    "doors opening into their own interiors, exposing tiny people",
    "buildings peeling like printed wallpaper into moving traffic",
    "human bodies flowing into metallic structures and back",
    "architecture growing through faces and reassembling differently",
    "paper and torn posters forming huge moving organisms",
    "a continuous cascading transformation of scale and identity",
]

MATERIALS = [
    "wet newsprint and dirty glass",
    "shattered mirrors and peeling paint",
    "old plywood and rusted metal",
    "muddy reflections and broken ceramic",
    "torn posters and green climbing roots",
    "transparent liquid and damp paper",
    "faded carpets and fractured concrete",
    "cracked porcelain and tangled cables",
    "smudged ink and shattered television screens",
    "broken glass and growing plants",
    "rainwater and powdered brick",
    "collapsing book pages and weathered stone",
]

VIEWS = [
    "wide CCTV-style high-angle view",
    "disturbing medium crowd angle",
    "distant architectural documentary viewpoint",
    "overhead view filled with densely layered events",
    "frontal wide shot with intricate foreground and background",
    "fixed security-camera composition",
    "compressed telephoto shot with many overlapping subjects",
    "strange low angle looking through crowded objects",
]

def make_prompt(rng: random.Random, previous: str = "", previous_location: str = ""):
    """Select substantially different settings, not small variants of a street."""
    for _ in range(80):
        if rng.random() < 0.28:
            place = rng.choice(ARTIST_SCENES)
        else:
            place = rng.choice(WORLDS)
        if place == previous_location:
            continue
        prompt = (
            f"{rng.choice(VIEWS)}, {place}, {rng.choice(INTRUDERS)}, "
            f"{rng.choice(CHANGES)}, {rng.choice(MATERIALS)}, "
            "dense detailed foreground and background, strange documentary realism"
        )
        if prompt != previous:
            return prompt, place
    raise RuntimeError("Couldn't make a distinct prompt")


def make_plan(
    frames: int = 1000,
    seed: int = 9191,
    min_interval: int = 60,
    max_interval: int = 140,
    transition_min: int = 20,
    transition_max: int = 60,
) -> dict:
    """Build a continuous schedule with a DIFFERENT morph length each time.

    Each semantic state lasts a random 60-140 frames. In its final 20-60
    frames, positive CLIP embeddings morph gradually into the next state.
    60-frame changes are only picked for segments long enough to leave
    at least one frame fully under the old prompt. No hard scene cuts.
    """
    if frames < 1 or frames > 10000:
        raise ValueError("frames must be 1..10000")
    if min_interval < 2 or max_interval < min_interval:
        raise ValueError("invalid duration bounds")
    if not 1 <= transition_min <= transition_max:
        raise ValueError("invalid transition bounds")
    if transition_min >= min_interval:
        raise ValueError("shortest transition must fit within shortest interval")

    rng = random.Random(seed)
    prompts = []
    segments = []
    timeline = []
    previous = ""
    previous_location = ""
    start = 0
    while start < frames:
        prompt, place = make_prompt(rng, previous, previous_location)
        current = len(prompts)
        prompts.append(prompt)
        previous, previous_location = prompt, place
        interval = rng.randint(min_interval, max_interval)
        end = min(start + interval, frames)
        segments.append({
            "start": start, "end_exclusive": end,
            "prompt_id": current,
            "next_prompt_id": None,
            "transition_start": end,
            "transition_frames": 0,
        })
        start = end

    for i, segment in enumerate(segments):
        next_id = i + 1 if i + 1 < len(prompts) else None
        segment["next_prompt_id"] = next_id
        if next_id is not None:
            # The first segment might be only 60 frames; keep >=1 frame fully
            # conditioned to its original subject before the new one arrives.
            duration = segment["end_exclusive"] - segment["start"]
            max_allowed = min(transition_max, duration - 1)
            transition_frames = rng.randint(transition_min, max_allowed)
            segment["transition_frames"] = transition_frames
            segment["transition_start"] = segment["end_exclusive"] - transition_frames

        for frame in range(segment["start"], segment["end_exclusive"]):
            fade = segment["transition_frames"]
            if next_id is None or frame < segment["transition_start"]:
                alpha = 0.0
            else:
                x = (frame - segment["transition_start"] + 1) / fade
                alpha = x * x * (3 - 2 * x)  # gradual CLIP conditioning, not an RGB fade
            timeline.append({
                "frame": frame,
                "from_prompt": i,
                "to_prompt": i if next_id is None else next_id,
                "alpha": round(alpha, 7)
            })

    assert len(timeline) == frames
    return {
        "version": 2,
        "generator": "RTNVE wild procedural conceptual prompts; no LoRAs",
        "seed": seed, "frames": frames,
        "min_interval": min_interval,
        "max_interval": max_interval,
        "transition_min": transition_min,
        "transition_max": transition_max,
        "prompts": prompts,
        "segments": segments,
        "frame_schedule": timeline,
        "warning": "Gradually mixed neural TEXT conditioning, not image crossfades."
    }
