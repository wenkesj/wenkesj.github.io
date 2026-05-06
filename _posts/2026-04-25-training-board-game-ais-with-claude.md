---
layout: post
title: "Games and RL"
date: 2026-04-25
math: true
---

Games and reinforcement learning have been a hobby of mine for as long as I can remember, 
board games especially. Early on in my career, I had the opportunity to apply
reinforcement learning in settings where research and engineering were pushing
Deep Learning to it's practical limits: on hardware and real world problems.
There has been many highs in my career outside of reinforcement learning since then, 
but it's always stuck around _as a hobby and interest_!

This blog post series is really about board games and agents.
It all started because I was so interested in learning how general agents could help me 
learn and try things that I never would have been able to try in a such a short period of time.

---

## A short primer

I never took classes or was taught RL during school, so I had to look to the internet for resources.
The two books I could say that I learned the most from and where I found my niche and passion for these things, were
[Sutton & Barto's *Reinforcement Learning: An
Introduction*](http://incompleteideas.net/book/the-book-2nd.html) and 
[Giacomo Bonanno's *Game Theory*](https://faculty.econ.ucdavis.edu/faculty/bonanno/PDF/GT_book.pdf).

You don't have to read these books to understand what this post is about, so here are some really high level,
defined terms (as they are applicable here to **Games**).

**Reinforcement learning (RL).** a policy plays a game over and
over. After every move it receives a tiny reward (positive or negative) and 
at the end of the game a big reward (positive for winning, negative for losing). 
Over many games, the RL algorithm we use adjusts the policy so it picks the moves 
that earned it the most reward. The loop is: see game _state_, pick _action_, receive _reward_, _learn_.

**Policy.** The "brain" of the agent. Mathematically, a function
$\pi_\theta(a \mid s)$: given a state $s$, return a probability over each
legal action $a$. The functions job is to learn so
this distribution puts high probability on good moves.

**Value function.** A sidekick that tells the RL algorithm how
much better a particular move was. Mathemetically, a function $\hat V(s)$.

**Proximal Policy Optimization (PPO).** A specific RL algorithm for updating the policy and value. 
The recipe's slogan is: *don't make changes to the policy that are too big.* 
In RL, gigantic policy updates can actually make it so the policy _forgets_ how to play. 
PPO essentially just _clips_ the size of each update.

**Behavior cloning.** The same term you keep hearing with LLMs: Supervised Fine Tuning (SFT). 
Train a policy to reproduce some ouput given an input. The same way you'd train a cat OR dog image classifier
or an LLM: "given this input, output this probability." An image classifier sees pixels and predicts the
label "cat"; an LLM sees prior tokens (sometimes _pixels_, as well!) and predicts the next token; BC
sees the game state and predicts the action the demonstrator provides. The result
is a policy that ~imitates the demonstrator. Useful as a starting point,
but capped at the demonstrator's skill.

**Nash equilibrium.** In a competitive 2-player game, a "Nash
equilibrium" is a policy (or a *roster* of policies) that you just can't do better than
*if your opponent plays optimally, too*. The best teams playing against each other.

**Policy-Space Response Oracles (PSRO)**, builds these policies for the Nash equilibrium iteratively: 
keep a *roster* of policies, find the best counter-policy ("best response") to the current roster, add it, then recompute the next roster. Repeat. The hope is that the roster converges to something no opponent can repeatedly win against.

**Payoff matrix.** A matrix where each row and column ($(i, j)$) is the win-rate of
policy $i$ playing as Player 1 against policy $j$ as Player 2.

---

## Vibes

I have a desktop with a GPU at home that I use only for my personal hobbies 
and a personal Claude Max subscription. I have a Claude Code 
session open on a terminal on my desktop with `/remote-control` enabled, 
and Claude on my phone which allows me to chat with the session on my desktop.
There wasn't much else involved in setting this up, other than the free tier `ngrok` 
with `OAuth` for accessing the local desktop server through my phone.
This project was entirely vibe-coded on some nights and weekends by Claude, 
guided by me talking to it.

The first thing I asked Claude to build was infrastructure: a
TypeScript engine encoding the rules of games I know how to play, 
(enjoy, and find interesting), a Node server exposing it over REST + WebSocket, 
and a React client to drive it all from a browser.

I gave Claude rulebooks and asked it to produce engines in a consistent shape: pure
deterministic game definitions, a session manager, a primitive action
API, a registry. We iterated a lot to ensure that it got the rules correct and the UI made sense. 
Claude would hand me a first pass, I'd play through a session in the UI, 
find the place where some edge case broke and we'd fix it. 
The end result was a substrate where humans, scripts, or learned agents could all be just
another caller of the same API.

## The cold-start problem

To play any of these games against an AI, you need an AI. Learning games 
from _scratch_ (pure entropy and chaos) can be incredibly hard and not really interesting to me. 
The easiest thing to do, is give Claude the rules, and have it come up with a set of anchors: 
greedy bots, sometimes wrapped in an intelligent search such as Monte Carlo Tree Search (MCTS), 
or hand-coded by Claude per game from the rule definitions.
They could play through to terminal states which was the most important. 
They were sometimes beatable but reasonable (dependending on the complexity of the game). 
They served as the cold-start opponent for everything.

They were also intellectually unsatisfying to begin with. In this post, I purposefully don't experiment 
to see if _Claude_ can iteratively come up with the best hard-coded AI's (or _even recreate this setup_!). 
Although, that is interesting to the research community and I: the concept that the RL algorithm can 
itself be a bag of coding LLMs: (e.g., [Code Space Response Oracles](https://arxiv.org/abs/2603.10098)). 
That is for another blog post soon!

The fixed policies don't model future state. They don't learn, for instance, 
that in 2-player UNO a Reverse acts as a Skip and should be saved for when the opponent is at one card. 
I wanted to see what a policy that actually adapts to the game's structure would look like:
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

### Modeling

This post focuses on four games: **UNO**, **Splendor**, **Backgammon**,
and **Catan**. Each game gets its own trained agent 
but they all go through the same modeling recipe: turn the raw
game state into something a network can consume, ask the network to
score the legal moves, and let RL update it from the engine's reward.

**State.** Every game has its own state shape: UNO has hands and a
discard pile, Splendor has gem piles and a card market, Backgammon has
24 points and a bar, Catan has a hex map and settlements. For the
network, we summarize each state into a fixed per-game *schema* (a
list of structured "modalities": scalars, entity lists, categoricals,
with declared shapes). The schema is what the network actually sees:
scalars for things like turn count and hand size, entity lists for the
cards or pieces on the board, categoricals for phase. Hidden information
(other players' hands, the unflipped Frodo identity, etc.) is replaced
by opponent *summaries* like hand sizes and scores, which keeps the
agent learning a general policy rather than a clairvoyant one.

**Actions.** At decision time, the engine hands the network the list of
currently-legal moves and asks it to score each one. The output is a
probability distribution over only the legal indices (illegal moves
never get any mass, so the network never has to learn "don't do that").
Each move is featurized with its action type plus any parameters
(which card, which vertex, which target color), and those features are
combined with the state encoding to produce a per-move logit.

**Reward.** The engine returns a small per-step delta whenever the
acting player's score changes (a settlement built, a card bought, an
opponent forced to draw) plus a terminal bonus on win or loss. We scale
the deltas per game so the magnitudes are comparable across UNO and
Catan, and that's what PPO consumes.

The first time a from-scratch PPO policy beat greedy 70-30 on UNO was
exciting in the way a benchmark number rarely is, because *I had just
played it through the UI ten minutes before* and could compare the
moves it made to what greedy would have done. The verification loop
collapsed: write code with Claude → train → click "AI Turn" in the
browser → think about whether the move made sense → tweak.

## PSRO

Beating greedy isn't enough. A policy that beats greedy can still be
wildly exploitable by a different policy the meta has never seen. PSRO
addresses this: keep a roster of policies, train the best counter-policy
("best response") to the current Nash mixture over the roster, add it
back, recompute the mixture, repeat. The roster is meant to converge to
a balanced lineup nobody can exploit by introducing yet another fighter.

The first 20-iter run plateaued. The Nash mix locked at iter 12 and the
next eight iterations added zero new weight. That collapse was the
trigger for everything that came next.

## Anchor pools, PFSP, and BC warm-start

I asked Claude for a focused literature review on what the PSRO + deep-RL
community does when the inner BR loop stops contributing. We boiled 
down the research to three concrete changes I thought were most practical.

**Anchor pools.** A fixed set of hand-coded heuristic opponents (per
game) that always live in the pool alongside the trained best
responses. Each anchor encodes a coherent style: in Catan, agents like
`city_rush`, `settlement_sprawl`, `road_builder`, `port_specialist`,
`hoarder`. PSRO's opponent sampler is forced to spend a minimum
fraction of training time against every anchor via a sampling **floor**.
The floor stops the BR from tunneling on a single style and forgetting
how to beat the rest (AlphaStar's "main exploiters" trick). The
practical impact: the difference between a Nash mix that locks at four
policies after iter 12 and one that keeps adding policies through iter
20.

**PFSP (prioritized fictitious self-play).** Within the anchor floor,
each opponent *i* is sampled with weight proportional to its meta weight
times $(1 - \text{wr}_i)^p$, where $\text{wr}_i$ is the BR's recent
win-rate against that opponent. Every gradient step is biased toward
the opponent the BR is currently *losing* to (the sparring partners
that still have something to teach). Without PFSP, training rollouts
fill up with easy wins that don't move the needle.

**Per-agent BC + TIES merge.** Behavior cloning is the bridge to PSRO's
inner PPO. The basic move is the same one LLM trainers make: pretrain
on demonstrations to put the policy in a sensible region of action
space (SFT), then use RL to push past it (RFT). The wrinkle that
worked best was *per-agent*: train one BC model per anchor archetype
on its own trajectories, then merge the per-agent state-dicts via
TIES. This preserves each archetype's distinctive behavior in the
merged checkpoint instead of producing a bland average. That checkpoint
becomes iter 0 of PSRO. It cuts the time the BR spends re-discovering
the basics, and it gives the early-iter best responses a coherent
starting style to refine instead of random noise to escape.

## What the policies learned

The single thing that turned Claude's vibe into something I trusted was
the **AI preview overlay** in the UI: when "Step" mode is on, every
move the AI is about to make is annotated with the policy's full
probability distribution and its value-head estimate. Watching those
preferences change across iterations told me more than any payoff
matrix did.

### UNO

![UNO AI preview]({{ '/assets/uploads/2026-05-06T14-17-11-888Z__IMG_0088.jpeg' | relative_url }})

> Trained UNO policy at iter 20, mid-game. Discard is a Wild that was
> set to Red; the network is holding `R↻` (Red Reverse), `R+2`, and
> several other cards. The preview reads 100% on `Play R↻`, with
> `Play R2` and `draw` at 0%. $\hat V$ −0.97, meaning the value head
> expects a near-loss from this state. The policy is right that the
> Reverse is the move (in 2-player UNO, Reverse acts as Skip).

UNO has a lot of variance: one bad shuffle and the game ends before
strategy matters. But iter_20 wins the majority of games against me
on the quick variant. The first time I noticed it had stopped playing
randomly was when it drew on its turn, looked at the new card, and
immediately played a Red `+2` to deny my UNO. That kind of
opportunistic "draw, see, attack" play wasn't there in iter_5.

### Splendor

![Splendor AI preview]({{ '/assets/uploads/2026-05-06T14-17-11-562Z__IMG_0089.jpeg' | relative_url }})

> Trained Splendor policy (iter 28, anchor-pool PSRO). 77% on
> `Reserve L1 K`, 6% on `Reserve L2 K`, 5% on `Reserve L2 G`, plus a
> long tail. $\hat V$ slightly negative. Note the small probability
> annotations on individual market cards (`R 5%`, `R 6%`, `R 77%`)
> coming from the rendered preview overlay.

Splendor is the game where foresight is most readable in the policy's
behavior. Strong play targets nobles to close out games and chains
higher-tier card buys; weak play stockpiles gems without a plan. The
trained policy reliably acquires nobles to win games, and across
iterations its market preferences shift from indiscriminate Level-1
grabbing to selective Level-2 / Level-3 reservations. It beats me
consistently.

### Backgammon

![Backgammon AI preview]({{ '/assets/uploads/2026-05-06T14-15-19-002Z__IMG_0096.jpeg' | relative_url }})

> Trained backgammon policy on a die-3 sub-move. Top suggestion is
> `Commit 19 → 22 (die 3)` at $\hat V$ +0.67. The board overlay shows
> per-destination probabilities: 38% on point 22 and 19, 31% on a
> back-bar drop, 12% mid-board.

Backgammon is interesting because there's a strong textbook baseline:
**pubeval**, the position-evaluation bot from the TD-Gammon era.
Against me the trained policy wins consistently. Against pubeval, it
plateaus around 50% win rate from a competent but unspecialized
starting point, despite an extended fine-tune run; pubeval is just a
strong heuristic and bridging the gap will take a different attack
(better reward shaping, more training, or self-play against pubeval
directly with PFSP). It's a solid game where I have a
clear "harder external baseline still ahead of us" signal.

### Catan

![Catan AI preview]({{ '/assets/uploads/2026-05-06T14-15-18-653Z__IMG_0108.jpeg' | relative_url }})

> Trained Catan policy at setup phase. Top vertex pick: 26.6%
> probability; 51 legal placements total. Each hex shows the policy's
> marginal preference for nearby vertices, and yellow halos mark the
> top picks. $\hat V$ +0.04.

Catan is the hardest game in the set, and the only one where the
trained policy hasn't beaten me yet. Two things make it harder. It's
**3-player**, which means the 2-player Nash solver doesn't apply; we
use **α-rank** instead (a Markov-chain stationary distribution over
the policy space) to compute the meta. And the hand-coded heuristics
are good: the 10-anchor pool includes agents that chain "missing 1
resource → maritime trade → build" and that successfully reach 10 VP
in 3-player games where pure-greedy bots stall at 2 VP.

After 20 PSRO iterations, the trained best responses are still losing
to several of those anchors outright. The α-rank weights end up almost
entirely on the heuristic anchors, with the trained BRs sitting at the
per-anchor floor. The negative result is informative: smart
trade-to-build heuristics cover most of Catan's strategic surface
already, and PPO from a BC start hasn't found a play style that
strictly beats them yet.

## Diagnostics from a single PSRO iter

![training diagnostics]({{ '/assets/training_diagnostics.png' | relative_url }})

> Inner-loop training signals during one BR run. Left: value-head
> prediction tracking the Monte-Carlo return. Middle: explained
> variance ($1 - \text{Var}(R - \hat V) / \text{Var}(R)$), the
> critic's heartbeat (positive means it's beating "always predict the
> mean"). Right: entropy gradually sharpening as the policy gets more
> confident, and gradient norm staying bounded.

![eval reward curve]({{ '/assets/eval_reward_curve.png' | relative_url }})

> Two flavors of reward telemetry. Grey line: noisy per-step training
> reward across 16 parallel rollouts. Amber: cleaner eval reward from
> 16 full games against the meta opponent every 20 BR steps. The
> policy improving across the iter is visible in the eval line even
> though the raw signal looks chaotic.

## Payoff matrices, per game

Each cell is the empirical win-rate of policy *i* (row) playing as p1
against policy *j* (col). Greens are wins for the row, reds are losses.
Amber crosshairs mark the policies the meta solver weighted at >2% in
the final equilibrium. Anchors are listed first along each axis;
trained best responses (`i1`, `i2`, ...) follow.

### UNO

![UNO payoff matrix]({{ '/assets/payoff_uno.png' | relative_url }})

> 26×26 matrix from the UNO quick-variant anchored PSRO run. Mass on
> the meta is spread between the heuristic anchors and several late
> trained iters; you can see the "newer trained iters dominate older
> ones in expectation" pattern in the bottom-right block.

### Splendor

![Splendor payoff matrix]({{ '/assets/payoff_splendor.png' | relative_url }})

> 36×36 matrix from the Splendor anchored PSRO run. The largest pool
> in the post (more anchors, more iterations); the equilibrium spreads
> across both anchors and trained BRs, and the trained policies
> consistently beat earlier anchors.

### Backgammon

![Backgammon payoff matrix]({{ '/assets/payoff_backgammon.png' | relative_url }})

> 28×28 matrix from the backgammon anchor-pool run. Fewer policies in
> the equilibrium because backgammon's variance per game is higher and
> several anchors dominate late-iter trained BRs at this training
> budget.

### Catan

![Catan payoff matrix]({{ '/assets/payoff_catan.png' | relative_url }})

> 19×19 matrix from the 3-player Catan run with α-rank as the meta
> solver. The cell at $(i,j)$ is row *i*'s average per-game credit
> when seated against column *j* (rotated through all three seats per
> cell). Almost the entire α-rank weight ends up on the heuristic
> anchors; trained best responses sit at the per-anchor floor. This
> is the matrix that visually says "the heuristics are still the
> right answer here."

## A note on how this got made

I want to flag something about the workflow, because it's not the
typical "deep learning research project" picture.

![Driving from the phone]({{ '/assets/uploads/2026-05-06T14-44-44-841Z__IMG_0091.jpeg' | relative_url }})

> **The actual driver's seat.** A Claude session on my phone, mid-build,
> scoping the work to add backgammon to the engine. My one-line ask
> ("implement Backgammon, run a 20-iter PSRO, research a set of
> challenging anchors") gets a proper scope estimate back before any
> hours of GPU time get burned. The text-input bar at the bottom is
> the instrument; the long-running Python process on my desktop GPU
> is the thing being instrumented.

Almost none of this was done at my computer. The first few sessions
(where the engine got built and the framework took shape) were at a
desk. Everything after that was from my phone, while doing other things.
Workout sets at the gym, evenings on the couch, weekend mornings between
errands. Claude Max subscription so the assistant never times out
mid-execution, my GPU desktop at home running the long jobs, a local
server I could SSH into from anywhere.

AI is fast becoming central to what most of us do for a living, and I wanted to push
myself to use it in modes I hadn't before. Agentic background jobs.
Long-running collaborations from a phone. Lit reviews that translate
directly into code changes. A hobby project I actually cared about
turned out to be the right place to practice.

## Where this leaves me

What I find genuinely interesting about this collaboration model (and
this is the part that doesn't fit into any single AI-assistant talking
point) is how much of the work was *me directing the experiment* and
*Claude executing the things I'd usually defer*. The big choices have
been mine: what to build, what to investigate, what failure modes to
chase, when to stop chasing them. The keystrokes have entirely been
Claude's. We co-debugged engine bugs, co-designed the dataset manager,
co-played recorded self-play games where I made decisions turn-by-turn
and Claude transcribed them through the API.

The bar I set at the start was: can these trained policies match or
beat the careful play I bring to the table? After playing several games
against each policy, the answer is "yes" for UNO, Splendor, and
Backgammon, and "not yet" for Catan. The two real frontiers right now
are pubeval in backgammon (a strong textbook baseline that the trained
policy plateaus against) and the heuristic anchor pool in 3-player
Catan (where smart trade-to-build heuristics still beat the trained
best responses).

## Future work

A few directions I want to explore from here:

- **Pretrained reasoning models as the policy.**
- **Algorithms beyond PPO for the inner-loop fine-tune.**.

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
