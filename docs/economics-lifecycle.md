# Economics: the lifecycle plan (protocol rev 6)

Why the token parameters are what they are. This is the design record of the
lifecycle analysis behind rev 6; the constants live in `rig/token.py` (the spec)
and `node/core/src/token.rs` (the mirror). Companion:
[data-provenance-and-payouts.md](data-provenance-and-payouts.md).

## The grounding fact

**Inference payers are the only place outside value enters the system.** Every
other actor (miner, data owner, proposer, server) is paid in tokens, and those
tokens have value only because someone eventually needs them to buy inference.
So the token's fundamental value is discounted future inference revenue, and the
design question for every parameter is: *does this spend the emission budget in
a way that maximizes the model's future usefulness per token minted?* Emission
is the bootstrap budget of the network. It gets spent exactly once.

## The mission constraint

Sestrian's miners do **useful work that must continue forever**: perpetual
public training is the product. Bitcoin can end its subsidy because ledger
security is cheap relative to fees; a model that must keep learning cannot. This
is the single place we deliberately *adapt* Bitcoin rather than copy it.

## Lifecycle phases and who earns what

| Phase | Model state | Miner income | Data income | Server income |
|---|---|---|---|---|
| **0 Bootstrap** | weak, no paying users | emission (speculative value) | emission data share (provenance-routed) | ~none |
| **1 First usefulness** | good at something narrow | emission + first fee-pool drips | emission share + first royalties | first fees |
| **2 Maturity** | broadly useful, fees ≳ emission | emission + fee training pool | mostly usage royalties | fee majority |
| **3 Perpetual** | fee-funded steady state | tail emission + fee training pool | usage royalties | fees |

The handoffs are the design: data moves from *subsidy-for-teaching* (block data
share, scored) to *revenue-for-being-used* (sketch-attributed royalties); miners
move from *subsidy* to *a share of usage revenue plus a guaranteed tail wage*.
Phase 3's equilibrium is honest: the network trains exactly as much as
inference demand justifies, and the capacity retarget regulates model growth.

## The three rev-6 decisions

**1. Tail emission, not a hard sunset.** Halvings proceed on schedule, but after
`TAIL_EPOCH = 9` halvings the reward floors at `TAIL_REWARD ≈ 0.0977` token/block
**forever** (Monero-proven). `emission()` never returns zero. This is the
guaranteed perpetual training wage; inflation asymptotes to ~0.05%/yr and keeps
falling. A hard sunset would have bet the mission on fees maturing by a fixed
date, and if that bet failed, training stops and the death spiral begins.

**2. Every inference fee splits 60/20/20** (`FEE_SHARE_SERVER/DATA/TRAIN`):
the server is paid instantly (it bore the serving compute; it absorbs division
dust so the split is supply-exact); 20% accumulates in `fee_data_pool` and 20%
in `fee_train_pool`: on-chain consensus balances, drained every block to that
block's provenance-named data owners and delta miners (blocks without recipients
carry them forward; distribution dust burns, same doctrine as emission dust).
Before rev 6 the fee went 100% to the server and **post-subsidy training had no
revenue source at all**: the mission was unfunded at exactly the phase it
mattered. When sketch-based usage attribution (§8) lands, only the data pool's
distribution rule changes; the flows are already live.

**3. 1M-block halving epochs** (was 100k). At the live 60s cadence, 100k-block
epochs meant a halving every ~69 days and the whole subsidy era burning out in
under two years, long before any plausible fee maturity. 1M-block epochs give
~2 years per epoch, Bitcoin-like pacing, and a ~19-year subsidized runway into
the tail.

## Supply

`BASE_REWARD = 50` tokens/block, halving each 1M-block epoch, floored at the
tail: the 10-epoch schedule sums to **≈ 99.9M tokens over ~19 years**, then the
tail adds ~51k tokens/yr (declining forever as a fraction of supply). There is
no other mint: no premine, no founder allocation, dust burns. Per-user limits do
not exist: earning is competitive (scored deltas, named data, VRF proposing),
and the recoverable stakes/bonds (delta bond, data stake, challenge stake) are
working-capital friction, not caps.

## Failure modes this design guards

- **Subsidy exhausted before product exists** → 1M epochs + tail: the runway is
  long and never fully ends.
- **Training unfunded at maturity** → the fee training pool: trainer income
  transitions smoothly from emission to revenue, no cliff.
- **Paying for data nobody uses** → provenance + scoring at training, sketch
  attribution at recall: data is paid for measured teaching and measured use,
  with no recency clock (a corpus that becomes useful years later earns then).
- **Deflationary seize-up / lost-key attrition** → the tail replenishes; 10⁻⁹
  grain divisibility absorbs appreciation.

## What still gates on other work

The *routing* of every flow above is enforced and golden-tested now. The
*weights* feeding two of them ride on off-chain model execution:
delta loss-scores (replacing interim registry-weights in the data share) gate on
delta scoring (task 108), and sketch alignment (replacing pro-rata drain of the
data pool) gates on the §8 attribution milestone. Until then the challenge
market is the backstop, and the interim rules are deterministic and auditable.
