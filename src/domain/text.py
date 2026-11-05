"""Arabic text domain models — text buffer, word builder, state machine states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class TextBuilderState(Enum):
    IDLE = auto()
    BUILDING_WORD = auto()
    WORD_COMPLETE = auto()
    SENTENCE_COMPLETE = auto()


class GestureInputType(Enum):
    LETTER = auto()
    SPACE = auto()
    DELETE = auto()
    CLEAR = auto()
    WORD = auto()


@dataclass
class TextBuffer:
    """Manages the current text state — word buffer and composed sentence."""

    current_word: list[str] = field(default_factory=list)
    sentence: list[str] = field(default_factory=list)
    max_word_length: int = 20
    max_sentence_words: int = 50

    @property
    def current_word_text(self) -> str:
        return "".join(self.current_word)

    @property
    def sentence_text(self) -> str:
        return " ".join(self.sentence)

    @property
    def full_text(self) -> str:
        if self.current_word:
            words = list(self.sentence)
            words.append(self.current_word_text)
            return " ".join(words)
        return self.sentence_text

    def add_letter(self, letter: str) -> None:
        if len(self.current_word) < self.max_word_length:
            self.current_word.append(letter)

    def delete_last_letter(self) -> None:
        if self.current_word:
            self.current_word.pop()

    def commit_word(self) -> None:
        word = self.current_word_text
        if word:
            self.sentence.append(word)
        self.current_word.clear()

    def add_space(self) -> None:
        self.commit_word()

    def clear_all(self) -> None:
        self.current_word.clear()
        self.sentence.clear()

    def clear_last_word(self) -> None:
        if self.sentence:
            self.sentence.pop()
        else:
            self.current_word.clear()

    @property
    def is_empty(self) -> bool:
        return not self.current_word and not self.sentence


@dataclass
class TextCommand:
    """A command produced by the hybrid recognition pipeline for the text builder."""
    input_type: GestureInputType
    value: str = ""
    confidence: float = 0.0
    gesture_id: str = ""

    @staticmethod
    def letter(text: str, confidence: float = 1.0) -> TextCommand:
        return TextCommand(GestureInputType.LETTER, text, confidence)

    @staticmethod
    def word(text: str, confidence: float = 1.0) -> TextCommand:
        return TextCommand(GestureInputType.WORD, text, confidence)

    @staticmethod
    def space() -> TextCommand:
        return TextCommand(GestureInputType.SPACE, " ")

    @staticmethod
    def delete() -> TextCommand:
        return TextCommand(GestureInputType.DELETE)

    @staticmethod
    def clear() -> TextCommand:
        return TextCommand(GestureInputType.CLEAR)
