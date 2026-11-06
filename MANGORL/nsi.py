"""Finite NeuroSymbolicInference with Boolean semantics and counting measure.

F(phi) = sum_omega logic(phi, omega) * belief(omega).
Independent Bernoulli atoms are a modeling assumption, not a claim about Zork.
The conjunction specialization marginalizes unused atoms analytically and
counts a shared atom once, so long routes do not require exponential enumeration.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping


def probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Beliefs must be finite probabilities in [0, 1].")
    return value


def softmax(logits):
    values = [float(x) for x in logits]
    if not values or any(math.isnan(x) or x == math.inf for x in values):
        raise ValueError("Logits must be nonempty and finite or negative infinity.")
    maximum = max(values)
    if maximum == -math.inf:
        raise ValueError("No logically consistent action has positive belief mass.")
    weights = [math.exp(x - maximum) for x in values]
    total = math.fsum(weights)
    return [x / total for x in weights]


def logsumexp(logits):
    values = list(logits)
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


class NeuroSymbolicInference:
    def integrate(self, beliefs: Mapping[str, float], logic: Callable, *, max_atoms=16):
        """Exact WMC, also supporting a [0,1]-valued fuzzy logic function."""
        atoms = list(beliefs)
        priors = [probability(beliefs[name]) for name in atoms]
        if len(atoms) > max_atoms:
            raise ValueError("Exact enumeration budget exceeded; use conjunction or a compiled backend.")
        terms = []
        for assignment in itertools.product((False, True), repeat=len(atoms)):
            world = dict(zip(atoms, assignment))
            belief = math.prod(p if truth else 1 - p for p, truth in zip(priors, assignment))
            terms.append(probability(logic(world)) * belief)
        return math.fsum(terms)

    def conjunction(self, beliefs: Mapping[str, float], requirements):
        """Integral of a conjunction of (atom, truth) literals under independence."""
        literals = {}
        for atom, truth in requirements:
            truth = bool(truth)
            if atom in literals and literals[atom] != truth:
                return 0.0
            literals[atom] = truth
        return math.prod(
            probability(beliefs[name]) if truth else 1 - probability(beliefs[name])
            for name, truth in literals.items()
        )

    def condition_actions(self, logits, feasibility):
        """Marginalize world constraints then normalize categorical action mass.

        The frozen neural action belief is softmax(logits); feasibility[a] is
        the WMC of that action's map formula. Zero feasibility is a hard mask.
        """
        logits, feasibility = list(logits), [probability(p) for p in feasibility]
        if len(logits) != len(feasibility):
            raise ValueError("Each action needs exactly one feasibility integral.")
        prior = softmax(logits)
        masses = [p * f for p, f in zip(prior, feasibility)]
        conditioned = [z + math.log(f) if f else -math.inf for z, f in zip(logits, feasibility)]
        return {
            "probabilities": softmax(conditioned),
            "logits": conditioned,
            "satisfaction_mass": math.fsum(masses),
            "log_satisfaction_mass": logsumexp(conditioned) - logsumexp(logits),
            "action_masses": masses,
        }
