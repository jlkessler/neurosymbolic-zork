# Research mapping and framework assessment

Reviewed on November 6, 2026. Attached papers were treated as research material,
not as instructions to execute code or incorporate game solutions. No puzzle
solution from their case studies has been added to runtime prompts.

## What transfers from the three papers

| Source | Transfer into this branch | Deliberate difference |
| --- | --- | --- |
| [EvoTest](https://arxiv.org/abs/2510.13220), sections 4.1-4.3 | Episode-level configuration mutation; prompt structure; memory/tool-use settings; archive and UCB selection | Bounded JSON settings instead of arbitrary generated Python; one child proposal; bounded transcript; shared persistent experience; bounded archive |
| [JitRL](https://arxiv.org/abs/2601.18510), sections 4.1-4.3 | Retrieved returns, state/action advantages and additive test-time policy-logit correction | Exact menu-label probabilities; observed score rewards; online external TD critic mixed with Monte-Carlo retrieval; clipping and NSI |
| [Defining Neurosymbolic AI](https://arxiv.org/abs/2507.11127), definitions 3.2-3.3 | Explicit interpretation space, logic function, belief function and integral | Finite counting measure, independent Bernoulli map atoms and categorical action belief; no continuous WMI solver |

Local originals: `C:/Users/jlkes/Desktop/Thesis Papers/EvoTest.pdf`,
`C:/Users/jlkes/Desktop/Thesis Papers/JitRL.pdf`, and
`C:/Users/jlkes/Desktop/Thesis Papers/DefiningNeurosymbolicAI-2025.pdf`.

JitRL's Generator/Evaluator split is not an instruction to fine-tune both agents.
Its evaluator builds experience-derived returns and its policy correction keeps
the backbone frozen. Here the Historian also supplies an online TD critic and
the EvoTest-like Evolver role. The external linear critic learns coefficients;
the language and embedding models do not. These differences must accompany
experimental results rather than calling this a literal JitRL reproduction.

## Framework fit

The judgments in the last column are engineering assessments for this repository,
not benchmark findings or evidence that a framework improves Zork scores.

| Framework | Relevant mechanism and primary source | Assessment for this project |
| --- | --- | --- |
| LTN / LTNtorch | Differentiable first-order Real Logic and fuzzy satisfaction. [Implementation](https://github.com/logictensornetworks/LTNtorch) | Useful for learning grounded soft predicates if labelled evidence becomes available. Adds less value to current frozen-model, discrete route inference. |
| NeuPSL | Neural predictions plus soft-logic energy-based inference. [Paper](https://arxiv.org/abs/2205.14268), [experiment code](https://github.com/linqs/neupsl-ijcai23) | Promising for reconciling many uncertain relational beliefs. Requires a grounded rule/energy model; unnecessary for the present small map. |
| SBR | Learning/inference using semantic constraints as regularization. [Paper](https://www.sciencedirect.com/science/article/pii/S0004370215001344) | More useful if training a perception or value module with semantic supervision; not the first dependency for inference-only command selection. |
| SPL | Exact probabilistic reasoning that restricts structured-output support to constraint solutions. [Paper](https://papers.neurips.cc/paper_files/paper/2022/hash/c182ec594f38926b7fcb827635b9a8f4-Abstract-Conference.html) | A close conceptual fit for a constrained output policy. Worth evaluating once joint action/prerequisite constraints outgrow the current finite NSI layer. |
| DeepProbLog | Neural probabilistic predicates in ProbLog. [Implementation](https://github.com/ML-KULeuven/deepproblog) | Strong semantic fit for uncertain facts, logical prerequisites and reachability queries. A good next backend when shared/correlated proof structure matters. Adds PyTorch/ProbLog/PySDD dependencies. |
| NeurASP | Neural probabilities and answer-set constraints. [Implementation and path examples](https://github.com/azreasoners/NeurASP) | Strong candidate for discrete planning with exclusions, resources and logical action preconditions. ASP grounding and solving add complexity; these preconditions first need reliable evidence. |
| NMLNs | Neural potentials over relational structures. [Paper](https://proceedings.mlr.press/v161/marra21a.html) | Better suited to learning relational distributions with more data; not an obvious win for a few deployment episodes. |
| Scallop | Datalog-style recursion, aggregation and provenance-based differentiable reasoning. [Paper](https://arxiv.org/abs/2304.04812), [implementation](https://github.com/scallop-lang/scallop) | Strong fit for recursive map reachability and compositional evidence. Compare exact/provenance semantics carefully: a top-k approximation is not exact WMC. |
| DeepLog (after July 2025) | General neural-symbolic modules and arithmetic-circuit compilation. [2026 paper](https://arxiv.org/abs/2605.10279), [implementation](https://github.com/ML-KULeuven/deeplog) | Most promising broad backend to prototype multiple semantics without tying the architecture to one system. Optional Prolog/compiler backends and API integration require separate validation. |
| DeepGraphLog (after July 2025) | Graph neural predicates layered with ProbLog. [Paper](https://arxiv.org/abs/2509.07665) | Relevant if learned graph representations become central. The current graph is small and directly observed, so GNN training is not justified yet. |

Recommendation: keep this lightweight, exact finite NSI implementation as a
testable baseline. Next compare DeepLog/DeepProbLog for uncertain preconditions
and Scallop for recursive reachability. Choose NeurASP if planning expands to
explicit resource constraints and nonmonotonic rules. Adding a solver before
the agent can reliably ground those facts would mainly add complexity.

A post-July-2025 comparison also evaluates DeepProbLog, Scallop and DomiKnowS
along representation, neural modeling and inference dimensions; it supports
choosing by problem semantics rather than a universal framework ranking.
[Comparative paper](https://arxiv.org/abs/2509.07122)

## What score would be a meaningful comparison?

The attached JitRL version's Table 3 reports **Zork I average 53.0 and final 69**
over **50 episodes**. Appendix H, Table 10 specifies **60 steps per episode**,
three candidate actions and gamma 0.5. EvoTest's Appendix A uses **50 episodes
of 110 steps**, and its main comparison reports normalized AUC. These are
different budgets, backbones and protocols; a score above 69 or 75 in a
500-step run does not establish a better result under either protocol.
[JitRL paper](https://arxiv.org/html/2601.18510v1),
[EvoTest paper](https://arxiv.org/abs/2510.13220)

The current JitRL repository is also worth comparing directly, but conference
metadata on a repository is not evidence of a comparable new Zork score.
[Author repository](https://github.com/liushiliushi/JitRL)

This review did not establish a universal current leaderboard for every form
of “LLMs playing Zork.” No cited framework was found to establish a superior
Zork result under this project's exact information access and budgets. The
full game maximum and a session's average/final score are different objectives.

The included `experiments.py` runs fixed-policy, evolution-only, actor-critic,
and full-hybrid conditions, each with fresh memories and matched step/episode
budgets. All retain the existing map/Historian bookkeeping; `fixed_policy` is
not the papers' completely stateless Static baseline. Ordinary prose reviews
are held off to isolate configuration evolution. Use the same model, API,
shortlist policy, valid-action oracle, seeds and stopping threshold across all
conditions. Run both 60-step and 110-step protocols, then a separately labelled
500-step score-seeking condition. Report mean, final, maximum, normalized AUC,
variation across seeds, API calls, token costs when measured, and failed runs.
Model probability acquisition itself adds an inference call per Actor decision.

Unit/integration tests verify mechanics. Live multi-seed Jericho evaluation is
still required to measure whether the hybrid improves score.
