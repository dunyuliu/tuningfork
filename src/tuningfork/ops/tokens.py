"""A step's answer as tokens: what a heatmap shows, in every front end.

Reading is `step_tokens`: the latest stream of the step's own answer — scored afterwards (see
`score`), or recorded when the harness asked for logprobs — and the streams of any samples
drawn at the step with `resample --logprobs`. Each comes with the numbers per token, and the
positions where the step's sampler could most easily have drawn another token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tuningfork.ops.common import OperationError
from tuningfork.ops.trash import refuse_trashed
from tuningfork.store.db import Store


def _round(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(value, places)


@dataclass(slots=True)
class TokensView:
    """One token stream, as a front end shows it."""

    record: Any  # store.db.TokenStreamRecord
    stream: Any  # analysis.tokens.TokenStream
    sampler: Any  # analysis.tokens.Sampler
    #: Computed on the server the step was recorded on (None: not known).
    same_server: bool | None = None
    #: For a sample's stream: which experiment and sample, and what it did.
    sample: dict[str, Any] | None = None

    def hotspots(self, n: int = 10) -> list[Any]:
        """The positions whose runner-up the step's sampler would most likely have drawn."""
        from tuningfork.analysis.tokens import candidates

        return candidates(self.stream, self.sampler, top=n, region="all", min_p=0.01, alternatives=3)

    def tokens_json(self) -> list[dict[str, Any]]:
        from tuningfork.analysis.tokens import CLOSENESS_EDGES, display_groups, heat, probability

        stream = self.stream
        out = []
        for start, end, text in display_groups(stream):
            for position in range(start, end):
                known = sorted(stream.known(position).items(), key=lambda item: -item[1])
                entropy, margin = stream.entropy(position), stream.margin(position)
                entry: dict[str, Any] = {
                    "i": position,
                    "text": text if position == start else "",
                    "part": stream.part(position),
                    "p": _round(probability(stream.logprobs[position]), 6),
                    "entropy": _round(entropy),
                    "margin": _round(margin),
                    "heat": heat(entropy),
                    "heat_margin": heat(None if margin is None else 1 - margin, CLOSENESS_EDGES),
                    "alternatives": [
                        [stream.token_text_of(token), _round(probability(logprob), 6)]
                        for token, logprob in known[: stream.top_k or len(known)]
                    ],
                }
                if end - start > 1:
                    entry["group"] = [start, end]
                if stream.is_redacted(position):
                    entry["redacted"] = True
                out.append(entry)
        return out

    def to_json(self, *, hotspots: int = 10) -> dict[str, Any]:
        return {
            "stream": self.record.to_json(),
            "sample": self.sample,
            "same_server": self.same_server,
            "format": self.stream.format,
            "top_k": self.stream.top_k,
            "sampler": self.sampler.to_json(),
            "text": self.stream.text(),
            "tokens": self.tokens_json(),
            "parts": [list(part) for part in self.stream.parts],
            "redacted": [list(span) for span in self.stream.redacted],
            "hotspots": [candidate.to_json() for candidate in self.hotspots(hotspots)],
        }


@dataclass(slots=True)
class StepTokens:
    """What token data a step has: the stream shown, and every other one to choose from."""

    run_id: str
    step_idx: int
    step_id: str
    wire: str
    view: TokensView | None
    #: Every stream at the step, newest first: the answer's own, and samples'.
    streams: list[dict[str, Any]] = field(default_factory=list)
    #: The forking-token probes made from the stream shown, newest first, each with its
    #: result at every position it forced: so the heatmap marks those tokens, and leads to
    #: each result — and the result back to its token.
    probes: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self, *, hotspots: int = 10) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step_idx,
            "wire": self.wire,
            "view": self.view.to_json(hotspots=hotspots) if self.view else None,
            "streams": self.streams,
            "probes": self.probes,
        }


def _sampler(checkpoint: Any) -> Any:
    from tuningfork.analysis.tokens import Sampler

    return Sampler.of(checkpoint.request, checkpoint.step.server_defaults)


def _sample_streams(store: Store, checkpoint: Any) -> list[tuple[Any, Any, Any]]:
    """(record, experiment, sample) for every sample at the step that kept its tokens."""
    samples = {}
    for experiment in store.experiments(
        run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, kind="resample", limit=50
    ):
        if not experiment.settings.get("logprobs"):
            continue
        for sample in store.samples(experiment.id):
            samples[sample.id] = (experiment, sample)
    records = store.token_streams(sample_ids=list(samples))
    return [(record, *samples[record.sample_id]) for record in records]


def _probes(store: Store, checkpoint: Any, stream_id: str) -> list[dict[str, Any]]:
    """The forking-token probes forked from this stream, each with its result per position."""
    from tuningfork.ops.experiments import experiment_json
    from tuningfork.ops.forking import forking_result

    found = []
    for experiment in store.experiments(
        run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, kind="forking-tokens", limit=50
    ):
        if experiment.settings.get("base_stream") != stream_id:
            continue
        report = forking_result(store, experiment).to_json()
        found.append({
            "experiment": experiment_json(experiment),
            "created_at": experiment.created_at,
            "target": report["target"],
            "positions": report["positions"],
        })
    return found


def step_tokens(store: Store, run: str, step: str, *, stream: str | None = None) -> StepTokens:
    """The step's token data: `stream` by its id, or the answer's own newest stream."""
    from tuningfork.analysis.tokens import TokenStream
    from tuningfork.ops.experiments import load

    checkpoint = load(store, run, step)
    own = store.token_streams(step_id=checkpoint.step.id)
    sampled = _sample_streams(store, checkpoint)
    listed = [
        {**record.to_json(), "experiment_id": None, "sample_idx": None, "outcome": None}
        for record in own
    ] + [
        {
            **record.to_json(),
            "experiment_id": experiment.id,
            "sample_idx": sample.idx,
            "outcome": sample.outcome_label,
        }
        for record, experiment, sample in sampled
    ]
    chosen, sample_info = None, None
    if stream is not None:
        chosen = next((record for record in own if record.id.startswith(stream)), None)
        for record, experiment, sample in sampled:
            if chosen is None and record.id.startswith(stream):
                chosen = record
                sample_info = {"experiment_id": experiment.id, "idx": sample.idx, "outcome": sample.outcome_label}
        if chosen is None:
            raise OperationError(f"step {checkpoint.step.idx} has no token stream {stream!r}")
    elif own:
        chosen = own[0]
    view = None
    if chosen is not None:
        view = TokensView(
            record=chosen,
            stream=TokenStream.from_json(store.token_stream_data(chosen)),
            sampler=_sampler(checkpoint),
            same_server=(
                None if chosen.server_id is None or checkpoint.step.server_id is None
                else chosen.server_id == checkpoint.step.server_id
            ),
            sample=sample_info,
        )
    return StepTokens(
        run_id=checkpoint.run.id,
        step_idx=checkpoint.step.idx,
        step_id=checkpoint.step.id,
        wire=checkpoint.wire.name,
        view=view,
        streams=listed,
        probes=_probes(store, checkpoint, chosen.id) if chosen is not None else [],
    )


@dataclass(slots=True)
class ScorePlan:
    """Scoring a step's recorded answer, checked before anything is sent."""

    checkpoint: Any
    base_url: str
    top_k: int

    def to_json(self) -> dict[str, Any]:
        usage = self.checkpoint.step.usage or {}
        return {
            "checkpoint": self.checkpoint.to_json(),
            "base_url": self.base_url,
            "top_k": self.top_k,
            "answer_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
        }


def prepare_score(
    store: Store, run: str, step: str, *, top_k: int = 8, base_url: str | None = None
) -> ScorePlan:
    from tuningfork.endpoints import check_endpoint
    from tuningfork.experiment.scoring import recorded_message
    from tuningfork.ops.experiments import default_upstream, load

    if not 1 <= top_k <= 20:
        raise OperationError("top_k is how many alternatives to keep at each token, from 1 to 20", field="top_k")
    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    message, _ = recorded_message(store, checkpoint.step)
    if not message:
        raise OperationError(f"step {checkpoint.step.idx} has no answer to score")
    return ScorePlan(
        checkpoint=checkpoint,
        base_url=check_endpoint(base_url or default_upstream(), home=store.home, purpose="scoring"),
        top_k=top_k,
    )


async def execute_score(
    store: Store, prepared: ScorePlan, *, progress: Any = None, cancel: Any = None
) -> StepTokens:
    """Score the answer on the server, store its stream, and return the step's tokens."""
    from tuningfork.backends.raw import RawError
    from tuningfork.experiment.scoring import ScoringError, raw_session, score_answer
    from tuningfork.ops.common import emit
    from tuningfork.ops.experiments import _cancellable

    checkpoint = prepared.checkpoint
    emit(progress, "scoring", run_id=checkpoint.run.id, step=checkpoint.step.idx, top_k=prepared.top_k)
    try:
        async with raw_session(store, prepared.base_url, tag="tokens", progress=progress) as (_, raw, server):
            record = await _cancellable(
                score_answer(store, checkpoint, raw, top_k=prepared.top_k, server_id=server), cancel
            )
    except (RawError, ScoringError) as error:
        raise OperationError(str(error)) from None
    emit(progress, "scored", stream_id=record.id, tokens=record.tokens, exact=record.exact)
    return step_tokens(store, checkpoint.run.id, str(checkpoint.step.idx), stream=record.id)


def sample_tokens(store: Store, experiment: str, idx: int) -> StepTokens:
    """A sample's token data, found by its experiment and index."""
    found = store.get_experiment(experiment)
    if found is None:
        raise OperationError(f"no experiment matching {experiment!r}")
    sample = next((s for s in store.samples(found.id) if s.idx == idx), None)
    if sample is None:
        raise OperationError(f"experiment {found.id[:8]} has no sample {idx}")
    records = store.token_streams(sample_ids=[sample.id])
    if not records:
        raise OperationError(
            f"sample {idx} of experiment {found.id[:8]} kept no tokens",
            hint="resample with --logprobs to keep each sample's tokens",
        )
    if found.run_id is None:
        raise OperationError("the sample is of a held call that was never recorded as a step")
    return step_tokens(store, found.run_id, str(found.step_idx), stream=records[0].id)
