---
layout: post
title: "An RL project I'd been deferring for years"
date: 2026-04-25
math: true
---

Games have been a hobby of mine for as long as I can remember, board
games especially. Reinforcement learning has shown up across a few of
my jobs (never as the main thing, always around the edges), but the
combination of policy search, multi-agent dynamics, and game theory has
been the part of ML I've been most personally drawn to. A
[paper on generalization in RL](https://arxiv.org/abs/1907.02050) is
where I last got to scratch the itch of sharing my own experiments and
what I find interesting about this corner of ML; this post is, in
spirit, the next round of that. The two RL projects I was most excited
to be a part of: off-policy training applied to the John Deere
harvester, and an earlier line of work using PPO on top of RNN forward
models, where simulation was fused with uncertainty estimates to produce
high-quality policies on the Quanser Qube via human-in-the-loop
exploration with minimal human interaction.
I followed AlphaGo, AlphaStar, OpenAI Five and the rest of the
DeepMind / early-OpenAI run when those papers landed, and kept a quiet
running list of experiments I'd want to try if I ever had the time to
wire up the substrate myself.

The substrate was always the wall. A clean game engine, a serializable
state, an action vocabulary, a server that can drive AI play, a UI to
actually see what the policy is doing. That's months of plumbing before
a single training step. With Claude available as a co-engineer to
execute the tedious half of that work alongside me, the wall got low
enough to climb. The pretext became "build a framework where I can play
the board games I like." The actual goal was to put my hands on every
interesting decision in modern RL (reward shaping, exploration,
opponent sampling, equilibrium solvers), using games I actually care
about as the testbed.

This is the story of how that played out, with diagrams.

---

## A short primer (for everyone)

If you've never trained a neural network, read a game-theory textbook, or
worked through [Sutton & Barto's *Reinforcement Learning: An
Introduction*](http://incompleteideas.net/book/the-book-2nd.html), the
rest of this post will still mostly make sense if you read this one
section.

**Reinforcement learning (RL).** A neural network plays a game over and
over. After every move it receives a tiny reward (positive if it scored
points, negative if it didn't) and at the end of the game a big reward
(positive for winning, negative for losing). Over many games, the network
adjusts its weights so it picks the moves that earned it the most reward.
The loop is: see state, pick action, receive reward, learn.

**Policy.** The "brain" of the agent. Mathematically, a function
$\pi_\theta(a \mid s)$: given a state $s$, return a probability over each
legal action $a$. The network's job is to learn $\theta$ (its weights) so
this distribution puts high probability on good moves.

**Value function.** A second head on the same network that estimates "how
much total reward do I expect to get from this state on?" It's not used to
pick actions directly; it's a sidekick that tells the policy gradient how
much better-than-average a particular move was. Written $\hat V(s)$.

**PPO.** A specific recipe for updating the policy. The recipe's slogan is:
*don't take steps that are too big.* In RL, gigantic policy updates can
collapse the policy entirely (it forgets how to play). PPO clips the size
of each update. The objective is

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$
is how much more likely the new policy makes the action vs the old one,
and $A_t$ is the advantage (how much better the action looked than what
the value head expected). The clip stops $r_t$ from drifting outside
$[1-\epsilon, 1+\epsilon]$.

**Behavior cloning (BC, also called supervised fine-tuning, SFT).** Skip
the trial-and-error. Show the network a stack of human (or expert)
demonstrations and train it the same way you'd train an image classifier
or an LLM: "given this input, output a probability distribution that puts
maximum weight on the right answer." It's the same cross-entropy loss
in all three cases. An image classifier sees pixels and predicts the
label "cat"; an LLM sees prior tokens and predicts the next token; BC
sees the game state and predicts the action the human took. The result
is a policy that imitates the demonstrator. Useful as a starting point,
but capped at the demonstrator's skill.

**Nash equilibrium and PSRO.** In a competitive 2-player game, a "Nash
equilibrium" is a (possibly mixed) strategy that you can't do better than
*if your opponent plays optimally too*. PSRO (Policy-Space Response
Oracles) builds Nash mixtures iteratively: keep a *pool* of policies, find
the best counter-policy ("best response") to the current Nash mixture of
the pool, add it to the pool, recompute the Nash mixture over the
expanded pool. Repeat. The hope is that the pool converges to something
no opponent can exploit.

**Payoff matrix.** A square table where entry $(i, j)$ is the win-rate of
policy $i$ playing as p1 against policy $j$ as p2. The Nash mixture is
computed from this table. We'll see ours later.

OK, primer done.

---

## Starting from the substrate

The first thing I asked Claude to help me build was infrastructure: a
TypeScript engine encoding the rules of UNO (plus a few other games
I'll come back to in future posts), a Node server exposing it over
REST + WebSocket, and a React client to drive it all from a browser.
The rest of *this* post is going to stick to UNO; the more interesting
games are a story for another time.

This part is where Claude earned its keep early. I'd written one or two
game engines from scratch before; the marginal cost of writing several
more by hand would have killed the project. Instead I gave Claude
rulebooks and asked it to produce engines in a consistent shape: pure
deterministic game definitions, a session manager, a primitive action
API, a registry. We iterated a lot. Claude would hand me a first pass,
I'd play through a session in the UI, find the place where some edge
case broke or a deck didn't drain, and we'd fix it. The end result was
a substrate where humans, scripts, or learned agents could all be just
another caller of the same API.

## The cold-start problem

To play any of these games against an AI, you need an AI. The first one we
wrote was the one we already knew how to write: greedy bots, sometimes
wrapped in MCTS, hand-coded by Claude per game from the rule definitions.
They could play through to terminal states. They were beatable but
reasonable. They served as the cold-start opponent for everything that
came later.

They were also intellectually unsatisfying. Greedy plays a one-step
optimization on hand-rolled features; it never *learns* anything. It
doesn't model the opponent. It doesn't model future state. It doesn't
discover, for instance, that in 2-player UNO a Reverse acts as a Skip and
should be saved for when the opponent is at one card. I wanted to see what
a policy that actually adapts to the game's structure would look like:
partly because I have an RL background and was curious to apply it on
something non-toy, and partly because the games themselves only become
*interesting* when the agent has to think more than one step ahead.

## Pivoting to RL

The pivot to learned policies opened up a different kind of partnership
with Claude. The infrastructure was Claude doing a lot of code that I knew
how to spec; the RL pipeline was Claude executing on RL choices that I
wanted to make but didn't want to keystroke. I'd say "we need a transformer
policy with per-game schemas and a value head." Claude would put up a first
draft. I'd point out that the value-loss gradient was going to corrupt the
BC-warm-started policy on iter 1 and ask for a `value_warmup_steps`
hyperparameter. Claude would add it. We'd run fit-tests, watch the loss go
to zero on a single trajectory to confirm the model was capable of
representing the policy at all. Then PPO, then training from scratch
against greedy.

### The architecture in more detail

**State.** Every game has its own `GameState` shape (UNO has hands,
discard pile, current color; Catan has a hex map, settlements, dev
cards). For the policy, we summarize the state into a fixed *schema*: a
list of "modalities" with declared shapes. UNO's schema, for instance, has:

- a 21-dim vector for the top discard card (5-color one-hot + 15-symbol
  one-hot + value),
- a 9-dim vector for shared state (current color, direction, pendingDraw,
  drawn-this-turn flag, deck size),
- an entity-list of up to 20 cards for the current player's hand,
- an entity-list of up to 3 opponent summaries,
- a scalar for the current player's score,
- a time scalar for the turn count,
- a categorical for "which player am I."

**Observation.** The policy network sees only what the *acting player*
would see. Hidden information (other players' hands) is replaced by
opponent *summaries* (hand sizes, scores, said-uno flags). This is what
keeps the network learning a general policy rather than a clairvoyant
one.

**Action space.** Per game, every legal move from the engine. UNO has
`play`, `draw`, `draw-stack`, `say-uno`, `challenge-wdf`, `pass`. Each
move object also carries parameters (`cardId`, `chosenColor` for wilds).
At decision time, the policy is given the list of currently-legal moves
and asked to score each one. It outputs a distribution over those legal
indices, never an illegal action. Concretely: each move is encoded into
a vector $\mathbf{m}_i$, and the policy logit is

$$
\text{logit}_i = \text{MLP}\big(\,[\,h_{\text{state}}\,;\, h_{\text{move}_i}\,]\,\big)
$$

where $h_{\text{state}}$ is the pooled transformer encoding of the state
and $h_{\text{move}_i}$ is a learned linear projection of the move
features.

> *Intuition:* the network reads the current state, then for each legal
> move it answers a single question: "how good does this move look from
> where we are?" The score for each move becomes a probability, and only
> legal moves are even on the ballot. Illegal moves get zero mass
> automatically; the network never has to learn "don't do that."

**Network.** A small bidirectional transformer (3 layers, 128 dims,
4 heads). Each modality is encoded into tokens, prepended with a CLS
token, and run through the transformer. The CLS output is the state
embedding. Two MLP heads on top: a policy head (scoring legal moves) and
a value head (scalar).

**Reward shaping.** Per step, the engine returns `deltaReward` (change in
the acting player's score) and `terminalBonus` ($+1$ win, $-1$ loss, $0$
draw or ongoing). We shape it as

$$
r_t = \frac{\Delta_t}{\text{scale}_g} + 2 \cdot b_t \cdot \mathbb{1}[\text{done}_t] - 0.001
$$

The per-game scale ($\text{scale}_g$) is chosen so the per-step magnitude
sits in $[-1, 1]$ on average (UNO uses 200, Splendor uses 15). The
$-0.001$ is a tiny step penalty discouraging stalling.

**Advantage estimation (GAE).** We don't credit a move with the raw
discounted return; we use Generalized Advantage Estimation, which trades
bias against variance:

$$
\hat A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad
\delta_t = r_t + \gamma \hat V(s_{t+1}) - \hat V(s_t)
$$

with $\gamma = 0.99$ and $\lambda = 0.95$. This is what plugs into the
PPO clipped objective above.

> *Intuition:* don't credit a move just by "did I eventually win."
> Credit it by how much *better the next state turned out than what the
> value head expected*. Stack a string of small "pleasant surprises" or
> "small disappointments" with a decay factor (so distant surprises
> matter less than nearby ones), and you get a far smoother signal to
> learn from than the raw end-of-game outcome alone. That smoother
> signal is what GAE is: a way of asking "was this move better or worse
> than predicted, on a sliding window?"

The first time a from-scratch PPO policy beat greedy 70-30 on UNO was
exciting in the way a benchmark number rarely is, because *I had just
played it through the UI ten minutes before* and could compare the
moves it made to what greedy would have done. The verification loop
collapsed: write code with Claude → train → click "AI Turn" in the
browser → think about whether the move made sense → tweak.

## PSRO and the plateau

Beating greedy isn't enough. A policy that beats greedy can still be wildly
exploitable by a different policy the meta has never seen. PSRO is the
loop that addresses this:

```
pool = [greedy]
meta = [1.0]
for k in range(K):
    new_BR = train_PPO(opponent ~ meta)
    pool.append(new_BR)
    payoff = evaluate_all_pairs(pool)
    meta = nash_equilibrium(payoff)
```

The Nash mix tells you which policies in your pool are *necessary*. A
policy with weight $0$ in the final Nash mix is one the equilibrium
doesn't need; either it's redundant or it's strictly worse than other
pool members.

> *Intuition:* picture a roster of fighters. Each round you train the
> *best counter* to the current "team" your roster fields, then add
> them to the roster. After enough rounds, ideally, the team converges
> to a balanced lineup that nobody can exploit by introducing yet
> another fighter. The Nash mix is the lineup; weight zero means
> "doesn't make the team."

This is where things got harder, and also where the human-in-the-loop
verification got more important. PSRO has many moving parts. The inner
PPO loop is short. The BR doesn't always converge. The Nash mix can
collapse. And the only real way to know "did this learn?" is to play the
policies. So I did. Multiple games against `iter_4_br`, `iter_5_br`,
`iter_7_br`, reading their move-by-move behavior in the UI, noticing
when a policy started using `challenge-wdf` (something none of the
earlier iterations did), noticing when one used the symbol-match trick
to switch colors, noticing when one mis-timed its WD4. Each game told me
more than the payoff matrix did.

After about twelve PSRO iterations, the Nash mix locked. Iters 13 through
20 added zero new weight to the meta. Same four policies, all eight more
iterations. I called this "the plateau."

![Nash mix size]({{ '/assets/nash_mix_size.png' | relative_url }})

> **The plateau** (grey line, baseline run): from iteration 12 onward, the
> Nash mixture is exactly the same four policies. Eight iterations of
> additional best-response training added nothing. The amber line is the
> next experiment, getting ahead of the story; that one keeps adding
> policies all the way to iteration 20.

## SFT then RFT

Behavior cloning came in as the bridge. We built a recording layer into
the engine, a dataset manager in the UI, a per-player action filter so you
could selectively clone "winning seat" trajectories or "human-only" moves.
The BC checkpoint became the warm-start init for PSRO's inner PPO, which
I kept calling the SFT→RFT analogue because it really is the same move
LLM trainers make: pretrain on demonstrations to put the policy in a
sensible region of action space, then use RL to push past human ceiling.

And exactly like LLMs, we hit the failure modes. The first attempt
catastrophically degraded the BC-init policy because PPO's value-loss
gradient was driving updates from a fresh-noisy critic. We added a
value-only warmup phase, lower learning rate, smaller clip ratio, lower
entropy coefficient, all of it grounded in the symptom: BC entropy was
0.1 and we were trying to push it back up.

## Asking Claude for a literature review

This is the moment I was most struck by what Claude could be useful for.
Instead of guessing the next intervention, I asked Claude to do a
literature review. *Specifically:* find what the PSRO + deep RL community
recommends for BR initialization, opponent sampling within batches, and
warm-start strategies, with citations and concrete actionable advice for
my setup. What came back was Pipeline PSRO, Fusion-PSRO, NeuPL, Mixed
Oracles, AlphaStar's PFSP. Papers I'd encountered in passing but hadn't
traced through to the level of "and here is the three-line state_dict
fusion that drops BR convergence time in half." Claude synthesized
fifteen-ish papers into three changes I could implement in an afternoon:

**1. Fusion init**: at iteration $k+1$, initialize the new BR's weights as
the Nash-weighted average of the existing pool:

$$
\theta_{\text{new}} = \sum_i \pi_i \cdot \theta_i
$$

where $\pi_i$ is the current Nash weight on pool member $i$ and $\theta_i$
is its parameter vector. This averages the *behaviors* of the pool, giving
the new BR a strong starting point that already plays competently against
the meta.

> *Intuition:* instead of starting the new policy from scratch (or from
> a single hand-picked checkpoint), start it from a weighted blend of
> the policies the equilibrium already cares about. You're handing the
> BR a "consensus opening" instead of asking it to learn to walk before
> it can run. It saves a lot of the inner-loop training budget that
> would have been spent re-discovering the obvious.

**2. Prioritized fictitious self-play (PFSP)**: sample opponents during BR
training proportional to

$$
w_i \propto \pi_i \cdot (1 - \text{wr}_i)^p
$$

where $\text{wr}_i$ is the BR's rolling win-rate against opponent $i$.
This focuses training on the opponents the BR is currently *losing* to,
exactly where new exploitability lives.

> *Intuition:* think of sparring partners. The ones you can already
> beat 95% of the time aren't making you better; the ones still beating
> you are. PFSP biases the training schedule toward the opponents the
> BR is actually struggling against, so every gradient step is fighting
> someone who still has something to teach. Without it, training
> rollouts get filled up with easy wins that don't move the needle.

**3. Value-head reuse**: carry forward the value head and shared backbone
from the previous BR iteration. Re-initialize only the policy head. The
value function generalizes across iterations because the underlying
state-value structure of the game doesn't change just because the
opponent mixture did.

We implemented all three behind CLI flags, re-launched the same 20-iter
PSRO run, and watched the plateau dissolve.

## The plot we cared about most

Here's what changed:

![vs greedy]({{ '/assets/vs_greedy.png' | relative_url }})

> **Each iteration's new policy played against the handcrafted greedy
> baseline.** The grey line (baseline run) bounces around 0.5–1.0 with
> high variance; most BRs only marginally beat greedy. The amber line
> (Fusion+PFSP+VH) is consistently above 0.5, peaks at 1.0 multiple
> times, and shows a clear pattern: every new BR is a meaningful
> improvement over the cold-start oracle. The y-axis is win rate from a
> single seat (the underlying game has a first-player advantage in UNO
> so 0.5 is a neutral baseline).

And the matrices the Nash solver eats, baseline on the left, Fusion on
the right:

![payoff matrices, baseline vs Fusion]({{ '/assets/payoff_matrix_compare.png' | relative_url }})

> **Two 22×22 payoff matrices.** Each cell is the empirical win-rate of
> policy *i* (row) playing as p1 against policy *j* (col) as p2,
> averaged over 6 games. Greens are wins for the row player, reds are
> losses. Amber crosshairs mark the policies the Nash solver chose to
> weight in the final equilibrium. **Left (baseline):** the equilibrium
> collapses onto just four early-iteration policies (`i5`, `i7`, `i9`,
> `i12`); every iteration after iter 12 was wasted. The matrix has
> noisier mid-band cells and no consistent dominance pattern.
> **Right (Fusion+PFSP+VH):** the equilibrium spreads across *six*
> policies, weighted toward later iterations (`i10`, `i14`, `i15`,
> `i17`, `i18`, `i20`). The matrix is visibly cleaner; late-iter rows
> show consistent green against early-iter columns, exactly the
> "newer policies dominate older ones in expectation" pattern you want.
> The right side is what the literature said Fusion-PSRO would deliver,
> and the left is what truncated-BR PSRO does without it.

## Looking inside one PSRO iteration

Six summary signals during a single best-response training run (iter 10
of the Fusion run):

![training diagnostics]({{ '/assets/training_diagnostics.png' | relative_url }})

> **Left**: the value head's mean prediction $\hat V(s)$ tracks the
> mean Monte-Carlo return reasonably well; they should be close if
> the critic is fit. **Middle**: explained variance, $1 - \text{Var}(R - \hat V) / \text{Var}(R)$,
> is a heartbeat for the critic. Above 0 means it's doing better than
> just predicting the mean. Ours is around 0.1–0.2, which is "weak but
> alive". Improving it more would help PPO advantages be cleaner.
> **Right**: entropy (orange) shows the policy gradually sharpening as it
> becomes more confident; gradient norm (green) is the size of each PPO
> update, which stays bounded, no blow-ups, no early collapse. Together
> these are the three "is the inner loop healthy?" plots.

And the actual reward signal during that same iter:

![eval reward curve]({{ '/assets/eval_reward_curve.png' | relative_url }})

> **Two flavors of reward telemetry.** The grey line is the noisy
> per-step training reward; every PPO update reports the average reward
> across the 16 parallel game rollouts that produced it. It's high
> variance because rollouts are short and incomplete. The amber line is
> the *eval* reward: every 20 BR steps, we play 16 *full* games against
> the meta opponent and report the mean. Cleaner, less noisy. You can
> see the policy improving across the iter even though the raw signal
> looks like a chaotic mess.

## Playing iteration 20

Once the run finished, I did the only verification that actually
satisfies me: I played games against the policy through the UI. Same
interface I've been using to play the game from day one: click moves,
opponent moves, see the score.

![UNO gameplay on mobile]({{ '/assets/gameplay_phone.png' | relative_url }})

> **The actual play interface.** Same React UI that drove every other
> experiment in the post: discard pile, draw pile, my hand visible,
> opponent's count + score visible, click a card to play it or "AI Turn"
> to let the trained policy move. The whole thing renders fine on a
> phone, which turns out to matter (more on that below).

In the first game I played, the policy beat me 79–0, mostly off mid-game
DrawTwo / Wild Draw Four combos and one late move where it drew on its
turn, looked at the new card, and immediately played a Red DrawTwo to
deny my UNO.

That last beat is the move that convinced me the network is no longer
just executing pattern-matched openings. It played opportunistically:
"draw, see what I got, attack with it." None of the earlier iterations
I'd played did that.

Across the earlier iters I played, the tactical sophistication rose
visibly with iteration: cleanly-played but thin at the start, the first
(often misused) WD4 challenges appearing mid-pool, and chains, DrawTwo /
WD4 combos, and opportunistic UNO denials showing up in the late
iterations.

### It's not strictly dominant

The policy doesn't always make the right call. In a separate game, it
had a Wild Draw Four with a Yellow
follow-up in hand; the right choice was to pick Yellow as the new color,
because that would have set up its own next play. It picked Green
instead, ended up unable to play after my response, and lost the game it
otherwise could have closed. Picking the wrong color after a wild is
exactly the kind of one-step look-ahead failure that PPO's value head
should be able to catch but evidently doesn't reliably.

So the honest summary is: against my level of play, iter_20 wins more
often than not, but not by overwhelming margin and not on every game. It
makes mistakes I can read and exploit when I'm paying attention. The
gain over iter_1 isn't "AI is now superhuman at UNO". It's "AI is now
playing well enough that I have to actually pay attention." That's the
bar I cared about, and I'm willing to call it cleared, but it's not a
domination story.

## A note on how this got made

I want to flag something about the workflow, because it's not the
typical "deep learning research project" picture.

![Driving from the phone]({{ '/assets/claude_phone.png' | relative_url }})

> **The actual driver's seat.** A Claude.ai conversation on my phone,
> mid-experiment, reading the literature-review reply that turned into
> the Fusion+PFSP+VH-reuse fix. The text-input bar at the bottom is the
> instrument; the long-running Python process on my desktop GPU is the
> thing being instrumented.

Almost none of this was done at my computer. The first few sessions
(where the engine got built and the framework took shape) were at a
desk. Everything after that was from my phone, while doing other things.
Workout sets at the gym, evenings on the couch, weekend mornings between
errands. Claude Max subscription so the assistant never times out
mid-execution, my GPU desktop at home running the long jobs, a local
server I could SSH into from anywhere.

The pattern that emerged: I'd send a message describing the next
experiment ("kick off a 20-iter PSRO run with these knobs, save it
under uno-quick-ppo, and report back when iter 12 lands"), Claude would
spin up the background job, schedule a wakeup for when the next milestone
should hit, and notify me. Between rest periods I'd read the latest
diagnostics screenshot, decide what to do next, fire off the next
instruction. The whole training pipeline above (Fusion init, PFSP, the
literature review, the BC pipeline) was driven from a phone in
fragments measured in single-digit minutes.

The deeper reason I leaned into this workflow: AI is fast becoming
central to what most of us do for a living, and I wanted to push
myself to use it in modes I hadn't before. Agentic background jobs.
Long-running collaborations from a phone. Lit reviews that translate
directly into code changes. A hobby project I actually cared about
turned out to be the right place to practice.

## A note on the playtest workflow

Two kinds of "human" played against the trained policies in this
project, and both ended up mattering:

- **Claude played.** During agentic sessions, Claude sat opposite the
  trained policy, read each state, reasoned out a move, and submitted
  it through the same `POST /api/games/:id/move` endpoint a UI click
  would. LLM on one side, PPO policy on the other, same engine
  refereeing. The challenges Claude raised mid-game ("that WD4 wasn't
  legal, the previous color was Red and you held a Red") were better
  than any unit test at surfacing engine bugs.
- **I played.** Same UI, but with me clicking cards on a phone during
  spare moments. Slower, more emotional, with the actual human stakes
  of "you're at one card, what do you do?" that's hard to simulate.

The modes are complementary. Claude's sessions catch anything
mechanically wrong fast: wrong move list, illegal action accepted,
score off by one, because the LLM is patient and methodical. My
sessions catch whether the policy is *interesting to play against*:
whether it bluffs, sets up, mistimes its WD4, denies an opponent's
UNO. Bugs vs. vibes. Both required, neither sufficient on its own.

## Where this leaves me

What I find genuinely interesting about this collaboration model (and
this is the part that doesn't fit into any single AI-assistant talking
point) is how much of the work was *me directing the experiment* and
*Claude executing the things I'd usually defer*. The big choices have
been mine: what to build, what to investigate, what failure modes to
chase, when to stop chasing them. The keystrokes have largely been
Claude's. We co-debugged engine bugs (the WD4 challenge handler not
clearing `pendingDraw` was one Claude found by reading my session
recording). We co-designed the dataset manager. We co-played a recorded
self-play game where I made decisions turn-by-turn and Claude transcribed
them through the API.

The bar I set at the start was: can these trained policies match or beat
the careful play that I bring to the table? After playing several games
against iter_20 (winning some, losing more), the answer is something
like "yes, in expectation, on the quick UNO variant, against my play
this particular weekend." That's neither a final answer nor a benchmark
number, because the bar is also moving; playing another fifty games
would teach me new patterns, which would change what a "human-equivalent"
policy needs to handle. But it's the answer I trust most, because it's
the one I got the only way I trust to know things: by playing the
result.

## References

- **Sutton & Barto.** *Reinforcement Learning: An Introduction* (2nd ed., 2018). [incompleteideas.net/book](http://incompleteideas.net/book/the-book-2nd.html)
- **Wenke et al.** "Reasoning About Generalization via Conditional Mutual Information." 2019. [arXiv:1907.02050](https://arxiv.org/abs/1907.02050)
- **Silver et al.** "Mastering the game of Go with deep neural networks and tree search." *Nature* 529 (2016). [DOI:10.1038/nature16961](https://www.nature.com/articles/nature16961). *AlphaGo*.
- **Vinyals et al.** "Grandmaster level in StarCraft II using multi-agent reinforcement learning." *Nature* 575 (2019). [DOI:10.1038/s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z). *AlphaStar; introduces PFSP*.
- **Berner et al.** "Dota 2 with Large Scale Deep Reinforcement Learning." 2019. [arXiv:1912.06680](https://arxiv.org/abs/1912.06680). *OpenAI Five*.
- **Vaswani et al.** "Attention Is All You Need." 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). *Transformer*.
- **Schulman et al.** "High-Dimensional Continuous Control Using Generalized Advantage Estimation." 2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438). *GAE*.
- **Schulman et al.** "Proximal Policy Optimization Algorithms." 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347). *PPO*.
- **Lanctot et al.** "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning." NeurIPS 2017. [arXiv:1711.00832](https://arxiv.org/abs/1711.00832). *PSRO*.
- **Muller et al.** "A Generalized Training Approach for Multiagent Learning." ICLR 2020. [arXiv:1909.12823](https://arxiv.org/abs/1909.12823). *α-PSRO / mixed oracles*.
- **McAleer et al.** "Pipeline PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in Large Games." NeurIPS 2020. [arXiv:2006.08555](https://arxiv.org/abs/2006.08555).
- **Liu et al.** "NeuPL: Neural Population Learning." ICLR 2022. [arXiv:2202.07415](https://arxiv.org/abs/2202.07415).
- **Yao et al.** "Fusion-PSRO: Nash Policy Fusion for Policy Space Response Oracles." 2024. [arXiv:2405.21027](https://arxiv.org/abs/2405.21027).
