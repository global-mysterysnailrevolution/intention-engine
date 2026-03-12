"""Event sourcing system for the intention engine hypergraph."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator, Literal

EventType = Literal[
    "node_added",
    "node_updated",
    "node_removed",
    "edge_minted",
    "edge_reinforced",
    "edge_weakened",
    "edge_membership_changed",
    "edge_closed",
    "search_executed",
]


@dataclass
class GraphEvent:
    """A single immutable event in the graph's history."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    entity_id: str = ""  # node or edge ID
    data: dict = field(default_factory=dict)  # event-specific payload
    intention: str = ""  # the intention that triggered this event (if any)


class EventLog:
    """Append-only event log with JSONL persistence.

    Stores a sequence of :class:`GraphEvent` instances in memory and
    optionally auto-flushes to a JSONL file on every append.
    """

    def __init__(self, file_path: str | None = None) -> None:
        self._events: list[GraphEvent] = []
        self._file_path: str | None = file_path

    # -- mutation --------------------------------------------------------

    def append(self, event: GraphEvent) -> None:
        """Add an event to the log.  Auto-flushes to disk when *file_path* is set."""
        self._events.append(event)
        if self._file_path is not None:
            self._flush_one(event)

    # -- queries ---------------------------------------------------------

    def events_for(self, entity_id: str) -> list[GraphEvent]:
        """Return all events whose *entity_id* matches."""
        return [e for e in self._events if e.entity_id == entity_id]

    def events_in_range(self, start: float, end: float) -> list[GraphEvent]:
        """Return events whose timestamp falls in [start, end]."""
        return [e for e in self._events if start <= e.timestamp <= end]

    def events_by_type(self, event_type: EventType) -> list[GraphEvent]:
        """Return events matching the given *event_type*."""
        return [e for e in self._events if e.event_type == event_type]

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Write all events to *path* as JSONL (one JSON object per line)."""
        with open(path, "w", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(asdict(event), default=str) + "\n")

    def load(self, path: str) -> None:
        """Append events from a JSONL file.  Missing file is silently ignored."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._events.append(
                    GraphEvent(
                        event_type=obj["event_type"],
                        timestamp=obj.get("timestamp", 0.0),
                        entity_id=obj.get("entity_id", ""),
                        data=obj.get("data", {}),
                        intention=obj.get("intention", ""),
                    )
                )

    # -- dunder helpers --------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[GraphEvent]:
        return iter(self._events)

    # -- internals -------------------------------------------------------

    def _flush_one(self, event: GraphEvent) -> None:
        """Append a single event to the backing file."""
        with open(self._file_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            fh.write(json.dumps(asdict(event), default=str) + "\n")
