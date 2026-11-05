"""Tests for ArabicTextBuilder — state machine, letter/word/delete/clear/space."""

import pytest
from src.domain.text import GestureInputType, TextCommand, TextBuilderState
from src.infrastructure.text_builder import ArabicTextBuilder


class TestArabicTextBuilder:

    def test_initial_state_idle(self):
        builder = ArabicTextBuilder()
        assert builder.state == TextBuilderState.IDLE
        assert builder.full_text == ""
        assert builder.current_word == ""
        assert builder.sentence == ""

    def test_add_single_letter(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("ا"))
        assert builder.current_word == "ا"
        assert builder.state == TextBuilderState.BUILDING_WORD

    def test_build_word_from_letters(self):
        builder = ArabicTextBuilder()
        for letter in ["م", "ر", "ح", "ب", "ا"]:
            builder.process(TextCommand.letter(letter))
        assert builder.current_word == "مرحبا"
        assert builder.state == TextBuilderState.BUILDING_WORD

    def test_space_commits_word(self):
        builder = ArabicTextBuilder()
        for letter in ["م", "ر", "ح", "ب", "ا"]:
            builder.process(TextCommand.letter(letter))
        builder.process(TextCommand.space())
        assert builder.current_word == ""
        assert builder.sentence == "مرحبا"
        assert builder.state == TextBuilderState.WORD_COMPLETE

    def test_sentence_construction(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("م"))
        builder.process(TextCommand.letter("ر"))
        builder.process(TextCommand.letter("ح"))
        builder.process(TextCommand.letter("ب"))
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.space())
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.letter("ل"))
        builder.process(TextCommand.letter("ع"))
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.letter("ل"))
        builder.process(TextCommand.letter("م"))
        builder.process(TextCommand.space())
        assert builder.sentence == "مرحبا العالم"

    def test_delete_letter_from_word(self):
        builder = ArabicTextBuilder()
        for letter in ["ا", "ب", "ت"]:
            builder.process(TextCommand.letter(letter))
        assert builder.current_word == "ابت"
        builder.process(TextCommand.delete())
        assert builder.current_word == "اب"

    def test_delete_word_when_empty(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.space())
        assert builder.sentence == "ا"
        builder.process(TextCommand.delete())
        assert builder.sentence == ""
        assert builder.state == TextBuilderState.IDLE

    def test_clear_all(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.space())
        builder.process(TextCommand.letter("ب"))
        builder.process(TextCommand.clear())
        assert builder.full_text == ""
        assert builder.current_word == ""
        assert builder.sentence == ""
        assert builder.state == TextBuilderState.IDLE

    def test_prevent_consecutive_repeat(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.letter("ا"))
        assert builder.current_word == "ا"

    def test_allow_same_letter_after_different(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.letter("ب"))
        builder.process(TextCommand.letter("ا"))
        assert builder.current_word == "ابا"

    def test_whole_word_input(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.word("مرحبا"))
        assert builder.sentence == "مرحبا"
        assert builder.state == TextBuilderState.WORD_COMPLETE

    def test_word_after_word(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.word("مرحبا"))
        builder.process(TextCommand.word("بالعالم"))
        assert builder.sentence == "مرحبا بالعالم"

    def test_full_text_property(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.word("مرحبا"))
        builder.process(TextCommand.letter("ا"))
        assert builder.full_text == "مرحبا ا"

    def test_reset(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.space())
        builder.process(TextCommand.letter("ب"))
        builder.reset()
        assert builder.full_text == ""
        assert builder.state == TextBuilderState.IDLE

    def test_batch_processing(self):
        builder = ArabicTextBuilder()
        commands = [
            TextCommand.letter("ا"),
            TextCommand.letter("ل"),
            TextCommand.letter("س"),
            TextCommand.letter("ل"),
            TextCommand.letter("ا"),
            TextCommand.letter("م"),
            TextCommand.space(),
        ]
        result = builder.process_batch(commands)
        assert result == "السلام"
        assert builder.sentence == "السلام"

    def test_empty_letter_ignored(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter(""))
        assert builder.current_word == ""

    def test_multi_word_sentence(self):
        builder = ArabicTextBuilder()
        words = ["السلام", "عليكم", "ورحمة", "الله", "وبركاته"]
        for w in words:
            builder.process(TextCommand.word(w))
            builder.process(TextCommand.space())
        assert builder.sentence == "السلام عليكم ورحمة الله وبركاته"
