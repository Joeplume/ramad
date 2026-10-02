# RAMAD framework-transfer benchmark protocol

This protocol separates two questions. The archived Figure 2c comparison tests
the complete RAMAD system against general-purpose models used directly. The
framework-transfer experiment tests what happens when the same RAMAD retrieval
context, structured prompt and task constraints are supplied to external
foundation models. The two score matrices are reported separately.

## Source materials

The framework-transfer experiment uses the original overall SERS
experimental-design question. For each framework-enabled condition, the same
five frozen passage texts are inserted in the same order using the same RAMAD
structured prompt. Bare controls receive the same original question without
the passages or RAMAD prompt structure.

The companion archive stores the exact passages, prompt text, model requests,
returned model identifiers and complete responses. Passage and prompt hashes
are retained with each answer so that the common inputs can be checked without
re-running retrieval.

## Generation

The primary within-model controls are Kimi K3 and GPT-5.5, each evaluated in a
bare condition and a RAMAD-framework condition. The archived RAMAD and baseline
answers are included as named anchors in the joint blinded scoring request.
Calls returning an empty or truncated visible answer are treated as failures
and are not scored. The provider, returned model ID, request parameters,
messages, response and usage are saved for completed calls.

No Claude score is reported because the available endpoint did not return a
complete, reproducible response. GPT-5.5 is used as the additional executable
frontier-model control. Historical Figure 2c totals and the new joint scoring
matrix are not pooled.

## Scoring and analysis

Available reviewer families score the blinded joint answer matrix on the
original six dimensions: SR, CSC, DQ, CS, QR and IS, each on a 1–5 scale. Three
rounds are recorded for every reviewer family. The released score table is
calculated from the six dimension scores; reviewer-provided total columns are
retained only for auditing and are not substituted for the dimension sum.

The Qwen3-4B reviewer was unavailable through the configured endpoint for this
run. The completed control therefore contains three reviewer families
and is reported as such rather than being merged with the archived four-reviewer
Figure 2c table.

Human expert scores, if collected, are recorded separately and are never
inferred from model-review scores or copied from the historical table. Repeated
reviewer calls are technical replicates, not independent experimental tasks;
claims from this control are therefore limited to the evaluated question and
model versions.

## Reproduction record

The companion `benchmark/framework_transfer/` directory contains the
bare-versus-framework panel and `benchmark/equal_harness/` contains the
RAMAD/Kimi K3/GPT-5.5 common-framework panel. Both directories include the
question, frozen passages, exact prompts, complete answers, raw review calls,
parsed dimension scores and aggregation outputs. The archived Figure 2c
materials remain under `benchmark/historical/`.

## Reported panels

In the ten-candidate framework-transfer matrix, Kimi K3 increased from 20.44
to 25.67/30 and GPT-5.5 increased from 20.89 to 26.33/30. In the separate
three-candidate equal-harness matrix, RAMAD, Kimi K3 and GPT-5.5 scored 22.89,
26.33 and 23.33/30, respectively. Each value is calculated from the six
dimension scores across three rounds from each of three reviewer families.
