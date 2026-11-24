"""Curated, token-budgeted urban prompt travel. No GPU dependencies."""
import random

BACKBONE = "documentary photograph, crisp fine detail, deep focus, dense foreground and background, natural light"
NEGATIVE = ("nsfw, blurry, out of focus, low resolution, low detail, oversmoothed, "
            "fog, flat lighting, washed out, empty background, indistinct shapes, "
            "cartoon, CGI, watermark, logo, vignette, shallow depth of field")
FAMILIES = (
    ("street", (
        "high angle busy city junction, crowded pedestrians pushing duplicated taxis, torn posters, shattered glass",
        "elevated street surveillance, crowds crossing between cars, people emerging from newspaper walls, fractured pavement",
        "high angle crowded street, people carrying miniature cars, windows breaking into paper layers, weeds",
    )),
    ("transport", (
        "wide underground platform, packed commuters, overlapping coats and merging faces, fractured advertising screens",
        "wide crowded airport terminal, duplicate travellers, suitcases splitting into people, broken glass, torn posters",
        "elevated railway concourse, intertwined bodies, passengers spilling from train windows, torn paper and tiles",
    )),
    ("architecture", (
        "wide damaged city street, abandoned taxis, pedestrians among rubble, concrete walls peeling into paper",
        "elevated urban demolition, crowded workers and wrecked vehicles, brick buildings splitting into furnished rooms",
        "wide fractured city block, people carrying broken windows, apartment walls unfolding into streets, debris",
    )),
    ("flooded", (
        "wide flooded railway hall, crowds crossing shallow water, reflections containing unfamiliar faces, shattered tiles",
        "elevated flooded shopping arcade, people and overturned chairs, reflective floors peeling into paper",
        "wide waterlogged public concourse, pedestrians between columns, broken glass, reflections turning into buildings",
    )),
    ("overgrowth", (
        "high angle busy street, cars and crowds entwined with climbing plants, pavement opening into roots",
        "wide ruined city junction, pedestrians emerging from foliage, shattered windows, vines growing through taxis",
        "elevated urban market, people carrying trees, apartment windows becoming leaves, torn posters and concrete",
    )),
)
ADJACENCY = ((0,1,4),(1,0,2),(2,1,3),(3,2,4),(4,3,0))

def make_plan(frames=360, seed=9191, min_interval=80, max_interval=140,
              transition_min=24, transition_max=48):
    """Dwell is fully conditioned time, with morphs additional to dwell.
    Constrain all lengths to fit exactly; no truncated final phase."""
    if not (1 <= frames <= 10000 and 1 <= min_interval <= max_interval
            and 1 <= transition_min <= transition_max):
        raise ValueError("Invalid frame or timing range")
    r = random.Random(seed)
    feasible = [n for n in range(1, frames//min_interval+1)
                if n*min_interval+(n-1)*transition_min <= frames
                <= n*max_interval+(n-1)*transition_max]
    if not feasible:
        raise ValueError("Frame count cannot fit requested complete dwells and morphs")
    n = r.choice(feasible)
    bounds = []
    for i in range(n):
        bounds.append((min_interval,max_interval))
        if i+1<n:
            bounds.append((transition_min,transition_max))
    remaining=frames
    lengths=[]
    for i,(lo,hi) in enumerate(bounds):
        tail=bounds[i+1:]
        value=r.randint(max(lo,remaining-sum(b for a,b in tail)),
                        min(hi,remaining-sum(a for a,b in tail)))
        lengths.append(value)
        remaining-=value
    assert remaining==0
    prompts, families, segments, schedule = [], [], [], []
    family=r.randrange(len(FAMILIES))
    for i in range(n):
        if i:
            pool=ADJACENCY[family] if r.random()<.7 else range(len(FAMILIES))
            family=r.choice(pool)
        label, variants=FAMILIES[family]
        options=[v+", "+BACKBONE for v in variants if v+", "+BACKBONE not in prompts]
        if not options:
            options=[v+", "+BACKBONE for v in variants if v+", "+BACKBONE!=prompts[-1]]
        prompts.append(r.choice(options))
        families.append(label)
    start=0
    for i in range(n):
        dwell=lengths[i*2]
        fade=lengths[i*2+1] if i+1<n else 0
        end=start+dwell+fade
        nxt=i+1 if fade else None
        segments.append(dict(start=start,end_exclusive=end,prompt_id=i,
                             next_prompt_id=nxt,transition_start=start+dwell,
                             transition_frames=fade,dwell_frames=dwell,family=families[i]))
        for f in range(start,end):
            x=(f-start-dwell+1)/fade if fade and f>=start+dwell else 0.0
            schedule.append(dict(frame=f,from_prompt=i,to_prompt=i if nxt is None else nxt,
                                 alpha=round(x*x*(3-2*x),7)))
        start=end
    assert len(schedule)==frames and start==frames
    return dict(version=3,generator="curated urban continuity",seed=seed,frames=frames,
                prompts=prompts,families=families,segments=segments,frame_schedule=schedule,
                min_interval=min_interval,max_interval=max_interval,
                transition_min=transition_min,transition_max=transition_max,
                negative=NEGATIVE,dwell_definition="fully conditioned frames, excluding morph")
