# Release decisions requiring owner confirmation

Original review recorded 2026-10-24. On 2026-10-26, the account owner requested publication of four selected repositories; this repository was selected. The earlier blanket instruction to keep all repositories private is superseded for the selected four. Merge authorization is being resolved separately.

The questions below remain a record of unverified licensing, data-access and scientific details. Publication authorization from the account owner must not be represented as independent proof of institutional permission, a dataset license or full scientific reproduction.
Passing execution tests does not answer the following questions. Potential contributors named below are parties to verify, not a legal determination of ownership.

## GE-R — Code rights and license

Confirm permission from the authors/institutions and contributors to the battery feature pipeline. Identify any upstream code reused in the feature generation or derivative losses before choosing a project license.

Required response: identify the rights holder and approver, the permitted code/legacy-file scope, the written approval or applicable license, and the exact project-wide license authorized. State whether publication of embedded constants and source provenance is permitted. No project-wide license will be inferred from a paper or a nearby license file.

## GE-D — Data and disclosure

For trajectory_cell_B1/B2/B3.mat, identify the underlying battery dataset, exact version, original terms and permission for redistributing derived features/RUL labels. Confirm whether row-index time is an acceptable disclosure of the processed schema.

Required response: dataset name/version, rights holder, access URL or contact process, permitted uses and redistribution terms. Choose **code plus synthetic data only**, **code plus an approved request-access route**, or **code plus specifically licensed data**. Approve disclosure of schemas/identifiers separately. Until answered, no research data or trained weights are included.

## GE-M — Scientific scope

Confirm the intended RUL time units, first/second derivative definitions and normalization. Specify the original cell split, all folds and ensembles, or approve a compact battery-window demonstration without full-paper reproduction claims.

Required response: approve the documented limited scope, or supply the authoritative equations/script/experimental protocol needed for a change. Smoke tests are not evidence that the scientific choice is correct.

## GE-H — Original Git metadata

The original commit retains a local machine email/hostname. New audit commits use the GitHub noreply address. Confirm either (a) retaining the original metadata, or (b) preparing a separately reviewed sanitized public history with required authorship/attribution preserved. No existing branch/history will be force-rewritten as part of this review.

## GE-A — Archive disclosure

Choose whether the eventual public release should include the selected `legacy/` files, a reviewed/redacted subset, or only the maintained runner plus provenance/attribution. Review historical paths, filenames, names and scientific constants before deciding. Original research folders remain unchanged.

Record decisions here with approver, date, scope and evidence reference. Do not paste confidential agreements or original datasets into this repository. An answer to these questions is not an instruction to change repository visibility or merge this PR.
