# Tensora Governance

Tensora is an open-source project developed under the Apache License 2.0. This document describes how technical decisions, maintenance responsibilities, and project changes should be handled as the contributor community grows.

## Principles

Project governance should optimize for:

- technical correctness;
- long-term maintainability;
- transparent decisions;
- clear ownership;
- compatibility discipline;
- security;
- evidence-based performance work;
- respectful technical disagreement.

Authority should be exercised through documented engineering decisions rather than undocumented convention.

## Maintainers

Maintainers are responsible for protecting the quality and direction of the project.

Responsibilities include:

- reviewing and merging changes;
- maintaining architectural boundaries;
- triaging defects and regressions;
- managing releases;
- maintaining supported-platform claims;
- handling security reports;
- reviewing dependency/license implications;
- keeping roadmap and documentation aligned with reality.

Maintainer status should reflect sustained, high-quality project responsibility rather than a single contribution.

## Technical decisions

Small implementation decisions may be resolved in pull-request review.

Material architectural changes use the RFC process described in `docs/RFC_PROCESS.md`.

Examples requiring broader review include:

- stable API/ABI changes;
- model-format changes;
- package boundary changes;
- new backend contracts;
- compatibility-policy changes;
- new security-sensitive extension mechanisms.

## Decision criteria

When multiple designs are viable, prioritize:

1. correctness;
2. clear ownership and failure semantics;
3. simplicity of the stable public contract;
4. maintainability;
5. portability;
6. measurable production value;
7. performance supported by evidence.

Do not choose greater architectural complexity merely because it appears more sophisticated.

## Consensus

The preferred outcome is rough technical consensus: major concerns are understood and either addressed or explicitly accepted with rationale.

Consensus does not require unanimous preference.

When consensus cannot be reached, maintainers responsible for the affected subsystem make the final decision and document the reasoning, particularly for long-lived architectural choices.

## Conflict of interest

Contributors should disclose material conflicts that could influence dependency, vendor, licensing, or security decisions.

A third-party integration should be selected on technical/project grounds rather than because a contributor has an undisclosed interest in a provider.

## Compatibility responsibility

Maintainers should treat stable public contracts as project assets.

Breaking changes require:

- technical justification;
- migration plan;
- appropriate versioning;
- compatibility documentation;
- RFC review when architectural.

## Security responsibility

Security-sensitive reports may require temporary private coordination before public discussion.

Security decisions should minimize user risk while preserving accurate disclosure and regression coverage.

See `SECURITY.md`.

## Release responsibility

Every release should have an identified maintainer responsible for ensuring that validation corresponds to the exact released revision.

See `docs/RELEASES.md`.

## Roadmap ownership

The roadmap expresses direction, not a promise that every listed feature will be implemented.

Roadmap changes should respond to:

- proven user needs;
- architectural prerequisites;
- maintenance capacity;
- measured performance constraints;
- platform changes.

The project should remove or defer roadmap items that no longer justify their complexity.

## Becoming a maintainer

As the project grows, maintainer nomination should consider:

- repeated high-quality contributions;
- sound technical judgment;
- review quality;
- reliability in follow-through;
- understanding of project architecture;
- responsible handling of compatibility and security;
- constructive collaboration.

The exact nomination/voting mechanism may be formalized when there are enough active maintainers to require it.

## Inactive maintainers

Maintainer roles should not become permanent blockers.

If a maintainer becomes inactive for an extended period, active maintainers may reassign subsystem/release responsibilities while preserving contribution history and credit.

## Project identity and trademarks

The Apache License 2.0 governs source licensing but does not automatically grant rights to project trademarks or branding beyond customary descriptive use. Any future trademark policy should be documented separately if needed.

## Governance evolution

This governance model is intentionally lightweight during the project's early phase.

As Tensora gains maintainers, releases, and external users, governance may be expanded through a documented proposal rather than informal process drift.