# Tensora RFC Process

Tensora uses Requests for Comments (RFCs) for changes that materially affect long-lived architecture, compatibility, or cross-package behavior.

The purpose of an RFC is not bureaucracy. It is to make expensive decisions reviewable before they become difficult to reverse.

## When an RFC is required

Use an RFC for changes involving:

- stable public Dart API design;
- the C ABI;
- tensor semantics;
- memory ownership or lifetime rules;
- concurrency/thread/isolate semantics;
- backend interfaces;
- device model;
- graph/IR semantics;
- `.tmodel` format;
- checkpoint/serialization contracts;
- package boundaries;
- compatibility/versioning policy;
- security boundary changes;
- major Flutter runtime architecture;
- new stable extension mechanisms;
- removal of a stable platform/backend.

## When an RFC is usually not required

Normally no RFC is needed for:

- typo/documentation corrections;
- implementation-only refactoring that preserves contracts;
- new tests;
- bug fixes consistent with existing semantics;
- isolated benchmark additions;
- small internal utilities.

When uncertain, prefer a short design discussion before writing a large implementation.

## RFC location

RFCs live under:

```text
docs/rfcs/
```

Recommended filename:

```text
NNNN-short-descriptive-name.md
```

Example:

```text
0001-native-handle-lifetime.md
```

RFC numbers are assigned sequentially when accepted for formal review.

## RFC states

An RFC uses one of these states:

```text
Draft
Proposed
Accepted
Rejected
Withdrawn
Superseded
Implemented
```

## Required structure

Every non-trivial RFC should contain:

```text
Title
Status
Authors
Created
Last updated

Summary
Motivation
Goals
Non-goals
Background
Detailed design
API/ABI impact
Ownership/lifetime
Concurrency
Error model
Compatibility
Security
Performance considerations
Testing strategy
Alternatives considered
Migration plan
Rollout plan
Open questions
```

Sections may be concise where not applicable, but important omissions should be explicit rather than accidental.

## Design quality

An RFC should answer practical questions.

For example, a native API RFC should explain:

- who allocates each object;
- who releases it;
- whether handles can outlive a context;
- what happens after duplicate release;
- thread safety;
- failure representation;
- ABI evolution.

A Flutter runtime RFC should explain:

- which isolate performs what work;
- cancellation;
- lifecycle shutdown;
- buffer ownership;
- backpressure;
- behavior after platform interruptions.

## Alternatives

Do not write an RFC as if the preferred design were inevitable.

Include serious alternatives and explain why they were not selected.

Good alternatives include:

- simpler design;
- different boundary;
- using an existing standard;
- postponing the feature;
- keeping behavior experimental rather than stable.

## Compatibility analysis

If the proposal changes an existing contract, state:

- what breaks;
- who is affected;
- whether deprecation is possible;
- migration path;
- version bump required;
- whether stored model/checkpoint artifacts remain readable.

## Security analysis

For any new parser, native extension, model-loading behavior, device boundary, or network-adjacent subsystem, identify trust boundaries and abuse cases.

## Performance claims

If performance motivates the RFC, include a reproducible baseline when implementation exists or define the benchmark required to validate the assumption.

Do not accept architectural complexity based only on expected speedup.

## Review process

1. Create the RFC as Draft.
2. Gather early feedback on problem statement and scope.
3. Move to Proposed once the design is coherent enough for formal review.
4. Address correctness, compatibility, ownership, security, and operational questions.
5. Accept, reject, withdraw, or supersede explicitly.
6. Implementation follows the accepted design unless later evidence requires an RFC amendment or replacement.

## Acceptance criteria

An RFC can be accepted when:

- the problem is clearly defined;
- scope/non-goals are bounded;
- ownership is unambiguous where relevant;
- failure behavior is defined;
- compatibility impact is understood;
- security implications are addressed;
- testing strategy can prove the design;
- alternatives were considered;
- no major open question makes implementation direction ambiguous.

Acceptance is a technical decision, not proof that every implementation detail is predetermined.

## Implementation tracking

After acceptance, implementation may be split into multiple pull requests.

Each pull request should reference the RFC and identify which acceptance criteria it satisfies.

An RFC becomes `Implemented` only when the stable behavior described by the RFC is actually present, tested, and documented.

## Changing an accepted RFC

Small clarifications may update an accepted RFC with documented rationale.

Material semantic changes require either:

- moving the RFC back to Proposed;
- an amendment with explicit re-review;
- a new RFC that supersedes the old one.

## RFC template

```markdown
# RFC NNNN: Title

- Status: Draft
- Created: YYYY-MM-DD
- Last updated: YYYY-MM-DD

## Summary

## Motivation

## Goals

## Non-goals

## Background

## Detailed design

## API / ABI impact

## Ownership and lifetime

## Concurrency

## Error model

## Compatibility

## Security considerations

## Performance considerations

## Testing strategy

## Alternatives considered

## Migration plan

## Rollout plan

## Open questions
```

## Principle

RFCs exist to preserve architectural coherence. They should be detailed enough to expose risks early, but no larger than needed to make the decision responsibly.