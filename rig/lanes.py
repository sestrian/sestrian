"""Training lanes (Sharding Road, Phase 2) — the throughput wall.

Before lanes, a block accepts one delta per miner up to a global cap, so a
thousand miners contend for a handful of seats. Lanes partition the EXPERT
pages into `n_lanes` groups; the beacon deterministically assigns each miner a
lane per epoch, and a v5 block only accepts a miner's delta if its claimed
pages lie in (backbone ∪ that miner's lane). Throughput then scales with lanes
instead of a single queue, and two miners in different lanes never collide.

Everything here is a pure function of (epoch, miner pubkey, active page table)
— identical on every node, golden-vectored, no per-block state.
"""

import hashlib


def n_lanes(active_expert_pages: int, lane_width: int) -> int:
    """How many lanes the current model supports: expert pages / lane_width,
    at least 1. Grows automatically as the model grows."""
    if active_expert_pages <= 0:
        return 1
    return max(1, active_expert_pages // max(1, lane_width))


def lane_of_miner(epoch: int, miner_pub: str, n: int) -> int:
    """Deterministic lane for a miner in an epoch — a hash so assignment can't
    be gamed by picking a key, and rotates every epoch so no miner owns a lane.
    """
    if n <= 1:
        return 0
    h = hashlib.sha256(
        f"sestrian-lane|v5|{epoch}|{miner_pub}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n


def lane_pages(lane: int, n: int, expert_page_ids: list[int]) -> set[int]:
    """The EXPERT page ids assigned to `lane`: a round-robin stripe over the
    active expert pages (stripe, not block, so growth appends spread evenly and
    every lane keeps getting new experts). Backbone (page 0) is never here —
    it is claimable by everyone, always."""
    if n <= 1:
        return set(expert_page_ids)
    return {p for k, p in enumerate(expert_page_ids) if k % n == lane}


def claimable_pages(epoch: int, miner_pub: str, model, lane_width: int) -> set[int]:
    """The full set of pages a miner may claim this epoch: backbone + its lane's
    experts. `model` is a ModelState; only ACTIVE pages participate."""
    expert_ids = [i for i in range(len(model.pages))
                  if model.pages[i][2] != "backbone" and model.is_active(i)]
    n = n_lanes(len(expert_ids), lane_width)
    lane = lane_of_miner(epoch, miner_pub, n)
    pages = lane_pages(lane, n, expert_ids)
    # backbone pages (kind == backbone), always claimable
    for i in range(len(model.pages)):
        if model.pages[i][2] == "backbone" and model.is_active(i):
            pages.add(i)
    return pages
