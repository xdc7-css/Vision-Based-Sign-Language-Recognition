"""Recognition domain models — recognition results, vocabulary entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class GestureCategory(Enum):
    ALPHABET = auto()
    STATIC_WORD = auto()
    DYNAMIC_GESTURE = auto()


@dataclass(frozen=True)
class VocabularyEntry:
    """A single entry in the gesture vocabulary."""
    gesture_id: str
    arabic_text: str
    category: GestureCategory
    description: str = ""
    confidence_threshold: float = 0.7

    def __post_init__(self) -> None:
        if not self.gesture_id:
            raise ValueError("gesture_id must not be empty")
        if not self.arabic_text:
            raise ValueError("arabic_text must not be empty")


@dataclass
class RecognitionResult:
    """Result of recognizing a single gesture."""
    gesture_id: str
    arabic_text: str
    category: GestureCategory
    confidence: float
    is_rejected: bool = False
    rejection_reason: str = ""
    processing_time_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.is_rejected and self.confidence > 0.0


@dataclass
class RecognitionSession:
    """Tracks a stream of recognition results over time."""
    results: list[RecognitionResult] = field(default_factory=list)

    @property
    def last_result(self) -> Optional[RecognitionResult]:
        return self.results[-1] if self.results else None

    @property
    def text_output(self) -> str:
        """Compose all results into a text string."""
        return " ".join(r.arabic_text for r in self.results if r.is_valid)


class WordVocabulary:
    """Manages known words with their expected gesture signatures."""

    def __init__(self) -> None:
        self._entries: dict[str, VocabularyEntry] = {}

    def register(self, entry: VocabularyEntry) -> None:
        self._entries[entry.gesture_id] = entry

    def register_many(self, entries: list[VocabularyEntry]) -> None:
        for entry in entries:
            self.register(entry)

    def get(self, gesture_id: str) -> Optional[VocabularyEntry]:
        return self._entries.get(gesture_id)

    @property
    def all_entries(self) -> list[VocabularyEntry]:
        return list(self._entries.values())

    @property
    def known_ids(self) -> set[str]:
        return set(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, gesture_id: str) -> bool:
        return gesture_id in self._entries


# ── Default vocabulary for common Arabic sign language words ──────────

DEFAULT_VOCABULARY: list[VocabularyEntry] = [
    VocabularyEntry("greeting_hello", "مرحبا", GestureCategory.STATIC_WORD, "Hello / Greeting"),
    VocabularyEntry("greeting_welcome", "أهلا", GestureCategory.STATIC_WORD, "Welcome"),
    VocabularyEntry("thanks", "شكرا", GestureCategory.STATIC_WORD, "Thank you"),
    VocabularyEntry("sorry", "آسف", GestureCategory.STATIC_WORD, "Sorry"),
    VocabularyEntry("help", "مساعدة", GestureCategory.STATIC_WORD, "Help"),
    VocabularyEntry("yes", "نعم", GestureCategory.STATIC_WORD, "Yes"),
    VocabularyEntry("no", "لا", GestureCategory.STATIC_WORD, "No"),
    VocabularyEntry("stop", "توقف", GestureCategory.STATIC_WORD, "Stop"),
    VocabularyEntry("good", "جيد", GestureCategory.STATIC_WORD, "Good"),
    VocabularyEntry("bad", "سيء", GestureCategory.STATIC_WORD, "Bad"),
    VocabularyEntry("how_are_you", "كيف حالك", GestureCategory.DYNAMIC_GESTURE, "How are you?"),
    VocabularyEntry("goodbye", "مع السلامة", GestureCategory.DYNAMIC_GESTURE, "Goodbye"),
    VocabularyEntry("please", "من فضلك", GestureCategory.DYNAMIC_GESTURE, "Please"),
    VocabularyEntry("thank_you_much", "شكرا جزيلا", GestureCategory.DYNAMIC_GESTURE, "Thank you very much"),
    VocabularyEntry("what", "ماذا", GestureCategory.DYNAMIC_GESTURE, "What"),
    VocabularyEntry("who", "من", GestureCategory.DYNAMIC_GESTURE, "Who"),
    VocabularyEntry("where", "أين", GestureCategory.DYNAMIC_GESTURE, "Where"),
    VocabularyEntry("when", "متى", GestureCategory.DYNAMIC_GESTURE, "When"),
    VocabularyEntry("why", "لماذا", GestureCategory.DYNAMIC_GESTURE, "Why"),
    VocabularyEntry("eat", "أكل", GestureCategory.DYNAMIC_GESTURE, "Eat"),
    VocabularyEntry("drink", "شرب", GestureCategory.DYNAMIC_GESTURE, "Drink"),
    VocabularyEntry("come", "تعال", GestureCategory.DYNAMIC_GESTURE, "Come"),
    VocabularyEntry("go", "اذهب", GestureCategory.DYNAMIC_GESTURE, "Go"),
    VocabularyEntry("sleep", "نوم", GestureCategory.DYNAMIC_GESTURE, "Sleep"),
    # Alphabet
    *[VocabularyEntry(f"letter_{chr(0x0627 + i) if i < 28 else '?'}",
                     chr(0x0627 + i) if i < 28 else "?",
                     GestureCategory.ALPHABET,
                     f"Letter {chr(0x0627 + i) if i < 28 else '?'}")
      for i in range(28)],
]
