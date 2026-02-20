Project Overview

This project implements the finalized rules of the board game Satellites and develops a very strong AI player.

The rules are finalized and must not be modified.

Primary goals:

Extract a deterministic, headless engine.

Guarantee rule correctness with tests.

Implement a strong PUCT-based MCTS AI.

Optionally extend to AlphaZero-style self-play with a neural network.

Authoritative Rule Clarifications

These define expected engine behavior and must be enforced by tests.

Unit Supply

Each player has 20 total double-sided units.

A player may never have more than 20 total units on the board.

Units may be any mix of bots and tanks.

Satellite Ring

The six satellites are:

2 × Move Tank

2 × Move Bot

1 × Add Tank

1 × Add Bot

At game start, satellites are placed in a random circular order.

Circular order defines adjacency for charge spreading.

Satellite order is part of the game state.

Action Semantics (Critical)

Each satellite charge spent is one independent atomic action.

Charges may be:

Spent on a single unit multiple times, or

Divided across multiple eligible units.

This applies to all satellite types.

There is no compound “multi-step” action. Multi-step behavior emerges from spending multiple charges.

Movement

A single Move action moves any number of units from selected eligible stack exactly one hex to an adjacent hex. This means that units may be left behind. The units that move may move on to a hex contain units of the same type belonging to the same player.

Multi-hex movement is achieved by spending multiple charges.

Movement legality is evaluated per step.

Blocking is enforced per step.

Artefacts block tanks, but bots may move on to them to capture them.

Tank Placement

Tanks may be dropped onto:

An empty hex

A hex containing the player’s own tank stack

Tanks may NOT be dropped:

On artefacts

On opponent starting hexes

Stacks combine normally.

Tank Shooting

Shooting is a distinct atomic action.

A tank stack may shoot into an adjacent hex.

The tank stack does not move.

One charge is consumed per shot.

A tank stack may shoot multiple times in a turn if charges allow.

Combat resolution:

Tank vs Tank:

If attacker stack size ≥ defender stack size → defender destroyed.

Otherwise → no effect.

Tank vs Bot:

Destroy all bots in target hex.

Engine Requirements

The engine must:

Contain no UI or rendering logic.

Be deterministic.

Support:

clone()

legal_actions()

apply_action(action)

is_terminal()

returns()

observation_tensor() (for future NN use)

All rules must be implemented in the engine, not the UI.

Action Encoding

Actions must be encoded as integers.

Game flow is phase-based:

SELECT_SATELLITE with one or more charges

SELECT_DIRECTION

SPEND_CHARGE (repeat until charges exhausted)

AUTOMATIC_CHARGE_SPREAD

Each SPEND_CHARGE action consumes exactly one charge and is one of:

MOVE_ONE_STEP

ADD_BOT

ADD_TANK

TANK_SHOOT

Illegal actions must be masked.

AI Requirements
Phase 1 — MCTS Baseline

Implement PUCT-style MCTS with:

Heuristic Policy (For MCTS)

Heuristics are permitted for:

Leaf evaluation in MCTS

Action priors for PUCT

Move ordering and progressive widening

Heuristics must never:

Alter legal move generation

Override rule correctness

Introduce nondeterminism

Heuristic categories to consider:

Score differential and near-win pressure

Immediate capture threats

Distance-to-artefact race positioning

Tank dominance and adjacency

Bot safety relative to enemy tanks

Lane blocking between bots and artefacts

Satellite charge tempo and denial (avoid gifting opponent power turns)

Heuristic weights are experimental and may be tuned.

No heuristic should hard-code a specific strategic doctrine; the system must allow multiple viable strategies to emerge through search and/or learning.

Random playouts are acceptable only for debugging.

Phase 2 — Neural Network (Optional)

If implemented:

Use PyTorch.

Network must output:

Policy head (masked)

Value head

Self-play must store:

(state, improved_policy, outcome)

Periodically evaluate new networks against previous versions.

Testing Requirements

Use pytest.

Add tests for:

Win condition at 9 points

Unit cap enforcement

Tank placement legality

Shooting behavior and resolution

Satellite random ring order

Charge spreading

Movement blocking

Atomic charge semantics

No AI development proceeds unless rule tests pass.

Non-Goals

No rule changes.

No UI logic inside the engine.

No neural network work until MCTS baseline is stable.