"""ArabicTextBuilder — state machine for building Arabic text from recognized gestures.

State machine:
    IDLE → BUILDING_WORD (on first letter/word)
    BUILDING_WORD → WORD_COMPLETE (on space)
    BUILDING_WORD → SENTENCE_COMPLETE (on clear/end)
    WORD_COMPLETE → BUILDING_WORD (on next letter)
    any → IDLE (on clear)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.domain.text import (
    GestureInputType,
    TextBuffer,
    TextBuilderState,
    TextCommand,
)


class ArabicTextBuilder:
    """Builds Arabic text from a stream of gesture recognition commands.

    Features:
    - Builds words from individual letter gestures
    - Prevents consecutive repeated letters
    - Supports delete (remove last letter/word), clear, and space gestures
    - Accumulates words into sentences
    - State machine for tracking build progress
    """

    def __init__(self) -> None:
        self._buffer = TextBuffer()
        self._state = TextBuilderState.IDLE
        self._last_letter: str = ""
        self._logger = logging.getLogger(__name__)

    # ── Public API ──────────────────────────────────────────────────

    def process(self, command: TextCommand) -> str:
        """Process a single text command. Returns the updated full text."""
        if command.input_type == GestureInputType.LETTER:
            self._handle_letter(command.value)
        elif command.input_type == GestureInputType.WORD:
            self._handle_word(command.value)
        elif command.input_type == GestureInputType.SPACE:
            self._handle_space()
        elif command.input_type == GestureInputType.DELETE:
            self._handle_delete()
        elif command.input_type == GestureInputType.CLEAR:
            self._handle_clear()

        return self.full_text

    def process_batch(self, commands: list[TextCommand]) -> str:
        """Process multiple commands sequentially."""
        for cmd in commands:
            self.process(cmd)
        return self.full_text

    @property
    def buffer(self) -> TextBuffer:
        return self._buffer

    @property
    def state(self) -> TextBuilderState:
        return self._state

    @property
    def full_text(self) -> str:
        return self._buffer.full_text

    @property
    def current_word(self) -> str:
        return self._buffer.current_word_text

    @property
    def sentence(self) -> str:
        return self._buffer.sentence_text

    def reset(self) -> None:
        """Clear the entire buffer and reset state."""
        self._buffer.clear_all()
        self._state = TextBuilderState.IDLE
        self._last_letter = ""
        self._logger.debug("Text builder reset")

    def undo_last_word(self) -> None:
        """Remove the last word from the sentence."""
        self._buffer.clear_last_word()
        if not self._buffer.sentence and not self._buffer.current_word:
            self._state = TextBuilderState.IDLE

    # ── Private handlers ────────────────────────────────────────────

    def _handle_letter(self, letter: str) -> None:
        """Add a letter to the current word, preventing consecutive repeats."""
        if not letter or len(letter) != 1:
            self._logger.warning(f"Invalid letter: '{letter}'")
            return

        # Prevent consecutive repeated letters
        if letter == self._last_letter and self._buffer.current_word:
            self._logger.debug(f"Blocked consecutive repeat of '{letter}'")
            return

        self._buffer.add_letter(letter)
        self._last_letter = letter
        self._state = TextBuilderState.BUILDING_WORD
        self._logger.debug(f"Added letter '{letter}' → word: '{self._buffer.current_word_text}'")

    def _handle_word(self, word: str) -> None:
        """Add a complete word to the sentence."""
        if not word:
            return
        self._buffer.sentence.append(word)
        self._buffer.current_word.clear()
        self._state = TextBuilderState.WORD_COMPLETE
        self._last_letter = ""
        self._logger.debug(f"Added word '{word}' → sentence: '{self._buffer.sentence_text}'")

    def _handle_space(self) -> None:
        """Commit current word and add a space."""
        self._buffer.commit_word()
        self._state = TextBuilderState.WORD_COMPLETE
        self._last_letter = ""
        self._logger.debug(f"Space → sentence: '{self._buffer.sentence_text}'")

    def _handle_delete(self) -> None:
        """Delete the last letter, or last word if word is empty."""
        if self._buffer.current_word:
            self._buffer.delete_last_letter()
            self._last_letter = self._buffer.current_word[-1] if self._buffer.current_word else ""
            self._logger.debug(f"Deleted last letter → word: '{self._buffer.current_word_text}'")
        elif self._buffer.sentence:
            self._buffer.clear_last_word()
            self._last_letter = ""
            self._logger.debug(f"Deleted last word → sentence: '{self._buffer.sentence_text}'")

        if not self._buffer.current_word and not self._buffer.sentence:
            self._state = TextBuilderState.IDLE

    def _handle_clear(self) -> None:
        """Clear the entire buffer."""
        self._buffer.clear_all()
        self._state = TextBuilderState.IDLE
        self._last_letter = ""
        self._logger.debug("Cleared all text")
