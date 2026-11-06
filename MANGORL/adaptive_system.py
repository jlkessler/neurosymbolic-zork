"""Coordinate slow configuration evolution and fast actor-critic inference."""
from __future__ import annotations

import json
from pathlib import Path

from .evolution import ConfigurationArchive, default_configuration, propose_configuration


class AdaptiveSystem:
    def __init__(self, args, *, actor, historian, critic, policy, navigator, output_dir):
        self.args = args
        self.actor, self.historian = actor, historian
        self.critic, self.policy, self.navigator = critic, policy, navigator
        self.output_dir = Path(output_dir)
        self.scores = []
        initial = default_configuration()
        initial.update(temperature=args.temperature, history_length=args.history_length,
                       retrieval_top_k=args.actor_rl_top_k,
                       retrieval_min_similarity=args.actor_rl_min_similarity,
                       actor_beta=args.actor_rl_beta, exploration=args.actor_rl_exploration,
                       critic_learning_rate=args.historian_critic_learning_rate,
                       route_risk_weight=args.route_risk_weight)
        self.configuration = initial
        self.configuration_id = "fixed"
        self.archive = ConfigurationArchive(args.evolution_path, initial=initial) if args.evolve else None

    def begin_episode(self, event_logger):
        if self.archive is not None:
            self.configuration_id, self.configuration = self.archive.select()
        cfg = self.configuration
        self.actor.apply_configuration(cfg)
        self.historian.evolved_strategy = cfg["historian_strategy"]
        self.policy.beta, self.policy.exploration_coefficient = cfg["actor_beta"], cfg["exploration"]
        self.policy.top_k, self.policy.min_similarity = cfg["retrieval_top_k"], cfg["retrieval_min_similarity"]
        self.critic.learning_rate = cfg["critic_learning_rate"]
        self.navigator.risk_weight = cfg["route_risk_weight"] if self.args.nsi else 0.0
        event_logger("active_configuration", {"id": self.configuration_id, "configuration": cfg})

    def finish_episode(self, episode, *, info, reason, event_logger, api_requests=None):
        # A failed API call is not an evaluated configuration, and must not
        # trigger a second paid LLM call or enter the score benchmark.
        if reason in {"episode_exception", "initialization_failed"}:
            event_logger("configuration_evaluation_skipped", {"reason": reason})
            return
        score = float(info.get("score", 0))
        self.scores.append(score)
        metric = {"episode": len(self.scores), "score": score, "moves": info.get("moves"),
                  "steps": episode.get("steps"), "termination_reason": reason,
                  "configuration_id": self.configuration_id, "seed": self.args.seed,
                  "api_requests_before_evolution": api_requests,
                  "policy_mode": self.args.actor_policy_mode,
                  "nsi": self.args.nsi, "actor_rl": self.args.actor_rl,
                  "historian_critic": self.args.historian_critic, "evolve": self.args.evolve}
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(metric) + "\n")
        summary = {"episodes": len(self.scores), "scores": self.scores,
                   "mean_score": sum(self.scores) / len(self.scores), "best_score": max(self.scores),
                   "final_score": self.scores[-1],
                   "normalized_auc": sum(self.scores) / (len(self.scores) * self.args.score_scale),
                   "score_scale": self.args.score_scale,
                   "target_success_rate": sum(s >= self.args.target_score for s in self.scores) / len(self.scores)}
        (self.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        event_logger("episode_metrics", metric)
        if self.archive is None:
            return
        self.archive.observe(score, score_scale=self.args.score_scale)
        try:
            child, rationale = propose_configuration(
                client=self.actor.client, model=self.args.historian_model or self.args.model,
                parent=self.configuration, episode=self.historian._review_payload(episode),
                extra_body=self.actor.extra_body, event_logger=event_logger,
            )
            child_id = self.archive.add(child, parent_id=self.configuration_id)
            self.archive.save()
            event_logger("configuration_mutation", {"parent_id": self.configuration_id,
                         "child_id": child_id, "rationale": rationale})
        except Exception as exc:
            # Keep measured parents and prior learning when mutation is invalid.
            event_logger("configuration_evolution_error", {"error": str(exc), "parent_retained": True})
            print(f"Configuration evolution failed; retaining archive: {exc}")
