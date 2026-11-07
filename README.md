# Documentation claim auditor

This offline tool checks documentation about assigning a category label to a
whole clip against static source patterns. It is deterministic, uses only the
standard library, makes no network calls, collects no telemetry, and writes no
files during an audit. It requires Python 3.9 or newer.

## Usage

```sh
python3 -m pip install .
video-classification-claim-audit audit .
video-classification-claim-audit audit --ignore vendor/ --ignore generated/ . --json
python3 -m video_classification_claim_audit audit . --verbose --fail-on medium
video-classification-claim-audit --version
```

The repository defaults to the current directory. The root README.md and
Markdown files beneath docs are checked. Triple-backtick fences are excluded;
inline code counts as prose. Python sources provide evidence throughout the
repository, excluding Git metadata, caches, and virtual environments. Repeatable
single-value ignore options skip repository-relative path prefixes. Test fixtures
under tests can produce medium findings when claims occur only there.

High findings fail by default; the medium threshold fails for either severity.
Exit codes are 0 when the threshold passes, 2 when findings meet the threshold,
and 1 for usage or IO errors. JSON output is a list whose findings contain
verdict, message, severity, and path fields. Verbose decisions use standard error.

```text
documented-video-classification-claim-false HIGH
documented-no-video-classification-claim-false HIGH
```

A positive claim without recognized evidence produces a finding. A denial
contradicted by evidence produces a finding. A denial without evidence passes,
as does documentation with no recognized claims.

## Differentiation

The first neighboring tool checks recognising human actions; this tool checks
assigning a category label to a whole clip. The second checks labelling single
still frames; this tool checks a whole clip. The third checks classification in
general; this tool checks labels for whole clips. The fourth checks producing
clips; this tool checks labelling them. Their names, in that order, are:

```text
action-recognition-claim-audit
image-classification-claim-audit
classification-claim-audit
video-generation-claim-audit
```

## Limitations

This is static pattern matching, not semantic judgement or runtime verification.
It recognizes exact substrings, not restatements. Matching ignores case but does
not stem words or normalize hyphens. Longest-first span blanking counts each
occurrence once under its longest phrase. Denial spans are removed first.

Construction and an instance call must occur in one module for the pipeline,
dedicated model, and backbone routes. Imported aliases are resolved; local
function and class definitions shadow route names. The generic route accepts
an attribute call on any base, or an imported bare call. A shared method name
on an unrelated object therefore counts as evidence, as in this example:

```text
unrelated.classify_video(data)
```

A recognized video-model backbone construction paired with an instance call
counts even though the same backbone can serve action recognition. Per-frame
image labelling with a vote is deliberately excluded because it supplies no
video-level model or video-named call. Loading, decoding, base encoders without
heads, imports, strings, comments, unbound references, and subprocess commands
alone do not establish the checked capability. Evidence split between modules
does not count. Recognized patterns do not prove correctness.

Exact substring matching has ambiguities: the first example below also describes
audio samples; the second can describe metadata or object labels; the third can
describe a library index. These still match. Longest-first matching controls
counting, and does not resolve the meaning of such prose.

```text
clip classification of audio samples
video tagging with object labels
video category classification for the library index
```

## Development

```sh
python3 -m unittest discover
python3 -m build
```

Released under the MIT license.
