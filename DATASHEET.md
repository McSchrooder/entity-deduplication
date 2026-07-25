# Datasheet

Datasheet for the benchmark corpus, after Gebru et al. (arXiv:1803.09010). The corpus
itself is not distributable. This documents what the published numbers were measured on.

## Motivation

This repo is a portfolio demonstration of entity-resolution work I did professionally,
stripped of anything sensitive. The corpus was not built for it. It is the live guest
base of a hospitality group, exported once. The full export holds ~1.3M profiles
and the architecture here was built for and ran against that scale. The published
benchmark uses a recent slice of it so results stay comparable run to run.

## Composition

The benchmark universe is the 300,000 most recently updated profiles, spanning
September 2023 to October 2025, drawn from two source systems. Fields cover name,
birth date, citizenship, contact details, address fields and source system. Field fill
is sparse and uneven. In the full export email is present on 7% of profiles, phone on
4%, zip on 5%, city on 61%. Names span the diacritic-heavy Central and South European
spectrum plus tourist traffic from everywhere, so transliteration and diacritic folding
variants are common.

Ground truth is derived from the corpus itself. Profiles sharing a document number, an
exact birth date and names that do not clearly disagree are labeled one person, with
name parts compared separately so a shared surname cannot carry two different first
names past the check. The result is 13,261 labeled duplicate clusters covering 34,224
profiles, 11.4% of the slice. Everything unlabeled is unknown, never confirmed unique.

## Collection and labeling

Profiles are front-desk and channel-manager entries, typed by staff, imported from
booking channels or migrated between systems. The duplication is organic, not injected.
Both source systems support explicit profile linking, but the export pipeline collapses
linked records into one row, so what survives is exactly the duplicates humans never
merged. The human signal that does survive is the document number, the same physical ID
card typed onto several profiles, and the labels are built from it.


## Preprocessing

The loader drops identity-document numbers at read time. They exist only inside the
labeling script, as label provenance, and are never ingested, indexed or embedded. The
pipeline's identity-hash column is read only to tag cluster difficulty and is never
indexed. Sampling is seeded and deterministic.

The search index is the blocking strategy. An
inverted index with fuzzy term expansion covers the lexical side, an HNSW graph over
the embeddings covers the semantic side, and rank fusion of the two produces the
candidate set every resolver works from. Whatever the index misses no resolver ever
sees, which is why blocking quality is measured on its own.

## Distribution

None. The repo ships a synthetic smoke corpus for runnability. It mimics the
schema and the perturbation types, not the statistics.

## Maintenance

Snapshot from July 2026. Numbers in the README refer to this snapshot.
