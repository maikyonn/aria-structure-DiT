"""Contains MIDI tokenizer with absolute onset timings."""

import functools
import itertools
import random

from pathlib import Path
from collections import defaultdict
from typing import Final, Callable, Any

from ariautils.midi import (
    MidiDict,
    MetaMessage,
    TempoMessage,
    PedalMessage,
    InstrumentMessage,
    NoteMessage,
    get_duration_ms,
)
from ariautils.utils import load_config, get_logger, warn_once
from ariautils.tokenizer._base import Tokenizer, Token


logger = get_logger(__package__)


# TODO:
# - Add asserts to the tokenization / detokenization for user error
# - Need to add a tokenization or MidiDict check of how to resolve different
#   channels, with the same instrument, have overlaping notes
# - There are tons of edge cases here e.g., what if there are two identical
#   notes on different channels.
# - Add information about the config, i.e., which instruments removed


class AbsTokenizer(Tokenizer):
    """MidiDict tokenizer implemented with absolute onset timings.

    The tokenizer processes MIDI files in 5000ms segments, with each segment
    separated by a special <T> token. Within each segment, note timings are
    represented relative to the segment start.

    Tokenization Schema:
        For non-percussion instruments:
            - Each note is represented by three consecutive tokens:
                1. [instrument, pitch, velocity]: Instrument class, MIDI pitch,
                    and velocity
                2. [onset]: Absolute time in milliseconds from segment start
                3. [duration]: Note duration in milliseconds

        For percussion instruments:
            - Each note is represented by two consecutive tokens:
                1. [drum, note_number]: Percussion instrument and MIDI note
                    number
                2. [onset]: Absolute time in milliseconds from segment start

    Notes:
        - Notes are ordered according to onset time
        - Sustain pedal effects are incorporated directly into note durations
        - Start (<S>) and end (<E>) tokens wrap the tokenized sequence, and
            prefix tokens for instrument, genre, composer, and form are
            prepended, i.e., before the <S> token
        - Various configuration settings affecting instrument processing,
            timing resolution, quantization levels, and prefix tokens can be
            adjusted in config.json at 'tokenizer.abs'.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        super().__init__()
        self.config = load_config(config_path)["tokenizer"]["abs"]
        self.name = "abs"

        # Calculate time quantizations (in ms)
        # TODO: abs_time_step_ms isn't very descriptive
        self.abs_time_step_ms: int = self.config["abs_time_step_ms"]
        self.max_dur_ms: int = self.config["max_dur_ms"]
        self.time_step_ms: int = self.config["time_step_ms"]

        self.dur_time_quantizations = [
            self.time_step_ms * i
            for i in range((self.max_dur_ms // self.time_step_ms) + 1)
        ]
        self.onset_time_quantizations = [
            self.time_step_ms * i
            for i in range((self.max_dur_ms // self.time_step_ms))
        ]

        # Calculate velocity quantizations
        self.velocity_step: int = self.config["velocity_quantization_step"]
        self.velocity_quantizations = [
            i * self.velocity_step
            for i in range(int(127 / self.velocity_step) + 1)
        ]
        self.max_velocity = self.velocity_quantizations[-1]

        # _nd = no drum; _wd = with drum
        self.instruments_nd = [
            k
            for k, v in self.config["ignore_instruments"].items()
            if v is False
        ]
        self.instruments_wd = self.instruments_nd + ["drum"]

        # Prefix tokens
        self.prefix_tokens: list[Token] = [
            ("prefix", "instrument", x) for x in self.instruments_wd
        ]
        self.composer_names: list[Token] = self.config["composer_names"]
        self.form_names: list[str] = self.config["form_names"]
        self.genre_names: list[str] = self.config["genre_names"]
        self.prefix_tokens += [
            ("prefix", "composer", x) for x in self.composer_names
        ]
        self.prefix_tokens += [("prefix", "form", x) for x in self.form_names]
        self.prefix_tokens += [("prefix", "genre", x) for x in self.genre_names]

        # Build vocab
        self.time_tok = "<T>"
        self.onset_tokens: list[Token] = [
            ("onset", i) for i in self.onset_time_quantizations
        ]
        self.dur_tokens: list[Token] = [
            ("dur", i) for i in self.dur_time_quantizations
        ]
        self.drum_tokens: list[Token] = [("drum", i) for i in range(35, 82)]

        self.note_tokens: list[Token] = list(
            itertools.product(
                self.instruments_nd,
                [i for i in range(128)],
                self.velocity_quantizations,
            )
        )

        self.special_tokens.append(self.time_tok)
        self.add_tokens_to_vocab(
            self.special_tokens
            + self.prefix_tokens
            + self.note_tokens
            + self.drum_tokens
            + self.dur_tokens
            + self.onset_tokens
        )
        self.pad_id = self.tok_to_id[self.pad_tok]

        # Pedal tokens appended to end of vocab
        self.include_pedal = self.config["include_pedal"]
        self.ped_on_tok = "<PED_ON>"
        self.ped_off_tok = "<PED_OFF>"
        if self.config["include_pedal"] is True:
            self.add_tokens_to_vocab([self.ped_on_tok, self.ped_off_tok])

    def export_data_aug(self) -> list[Callable[[list[Token]], list[Token]]]:
        return [
            self.export_tempo_aug(max_tempo_aug=0.2, mixup=True),
            self.export_pitch_aug(5),
            self.export_velocity_aug(1),
        ]

    def _quantize_dur(self, time: int) -> int:
        # This function will return values res >= 0 (inc. 0)
        dur = self._find_closest_int(time, self.dur_time_quantizations)

        return dur if dur != 0 else self.time_step_ms

    def _quantize_onset(self, time: int) -> int:
        # This function will return values res >= 0 (inc. 0)
        return self._find_closest_int(time, self.onset_time_quantizations)

    def _quantize_velocity(self, velocity: int) -> int:
        # This function will return values in the range 0 < res =< 127
        velocity_quantized = self._find_closest_int(
            velocity, self.velocity_quantizations
        )

        if velocity_quantized == 0 and velocity != 0:
            return self.velocity_step
        else:
            return velocity_quantized

    def _format(
        self,
        prefix: list[Token],
        unformatted_seq: list[Token],
        add_dim_tok: bool = True,
        add_eos_tok: bool = True,
    ) -> list[Token]:
        # If unformatted_seq is longer than 150 tokens insert diminish tok
        idx = -100 + random.randint(-10, 10)
        if len(unformatted_seq) > 150 and add_dim_tok is True:
            if (
                unformatted_seq[idx][0] == "onset"
            ):  # Don't want: note, <D>, onset, due
                unformatted_seq.insert(idx - 1, self.dim_tok)
            elif (
                unformatted_seq[idx][0] == "dur"
            ):  # Don't want: note, onset, <D>, dur
                unformatted_seq.insert(idx - 2, self.dim_tok)
            else:
                unformatted_seq.insert(idx, self.dim_tok)

        if add_eos_tok is True:
            res = prefix + [self.bos_tok] + unformatted_seq + [self.eos_tok]
        else:
            res = prefix + [self.bos_tok] + unformatted_seq

        return res

    def calc_length_ms(self, seq: list[Token], onset: bool = False) -> int:
        """Calculates sequence time length in milliseconds.

        Args:
            seq (list[Token]): List of tokens to process.
            onset (bool): If True, returns onset time of last note instead of
                total duration.

        Returns:
            int: Time in milliseconds from start to either:
                - End of last note if onset=False
                - Onset of last note if onset=True
        """

        offsets = [0]
        curr_num_time_toks = seq.count(self.time_tok)
        idx = len(seq) - 1
        for tok in seq[::-1]:
            if tok == self.time_tok:
                curr_num_time_toks -= 1
            elif type(tok) is tuple and tok[0] == "dur":
                assert seq[idx][0] == "dur", "Expected duration token"
                assert seq[idx - 1][0] == "onset", "Expect onset token"

                onset_ms = seq[idx - 1][1]
                duration_ms = seq[idx][1]
                assert isinstance(onset_ms, int), "Expected int"
                assert isinstance(duration_ms, int), "Expected int"

                abs_onset_ms = (
                    curr_num_time_toks * self.abs_time_step_ms
                ) + onset_ms
                abs_offset_ms = abs_onset_ms + duration_ms

                if onset is False:
                    offsets.append(abs_offset_ms)

                    if abs_onset_ms + self.max_dur_ms < max(offsets):
                        break

                elif onset is True:
                    # Last onset (positionally) is always greatest
                    return (
                        curr_num_time_toks * self.abs_time_step_ms
                    ) + onset_ms

            idx -= 1

        return max(offsets)

    def truncate_by_time(
        self, tokenized_seq: list[Token], trunc_time_ms: int
    ) -> list[Token]:
        """Truncates notes with onset_ms > trunc_time_ms."""
        time_offset_ms = 0
        for idx, tok in enumerate(tokenized_seq):
            if tok == self.time_tok:
                time_offset_ms += self.abs_time_step_ms
            elif type(tok) is tuple and tok[0] == "onset":
                if time_offset_ms + tok[1] > trunc_time_ms:
                    return tokenized_seq[: idx - 1]

        return tokenized_seq

    def _tokenize_midi_dict(
        self,
        midi_dict: MidiDict,
        remove_preceding_silence: bool = True,
        add_dim_tok: bool = True,
        add_eos_tok: bool = True,
    ) -> list[Token]:
        ticks_per_beat = midi_dict.ticks_per_beat
        midi_dict.remove_instruments(self.config["ignore_instruments"])

        if len(midi_dict.note_msgs) == 0:
            raise Exception("note_msgs is empty after ignoring instruments")

        channel_to_pedal_intervals = midi_dict._build_pedal_intervals()

        channels_used = {msg["channel"] for msg in midi_dict.note_msgs}

        channel_to_instrument = {
            msg["channel"]: midi_dict.program_to_instrument[msg["data"]]
            for msg in midi_dict.instrument_msgs
            if msg["channel"] != 9  # Exclude drums
        }
        # If non-drum channel is missing from instrument_msgs, default to piano
        for c in channels_used:
            if channel_to_instrument.get(c) is None and c != 9:
                channel_to_instrument[c] = "piano"

        if self.include_pedal:
            if len(channel_to_instrument.keys()) > 1:
                warn_once(
                    logger_name=logger.name,
                    message=f"AbsTokenizer config setting include_pedal=True "
                    "does not officially support multiple channels. You must "
                    "manually ensure that channels don't overlap.",
                )
            assert set(channel_to_instrument.values()) == {
                "piano"
            }, "AbsTokenizer config setting include_pedal=True only supports piano"

        # Calculate prefix
        prefix: list[Token] = [
            ("prefix", "instrument", x)
            for x in set(channel_to_instrument.values())
        ]
        if 9 in channels_used:
            prefix.append(("prefix", "instrument", "drum"))
        composer = midi_dict.metadata.get("composer")
        if composer and (composer in self.composer_names):
            prefix.insert(0, ("prefix", "composer", composer))
        form = midi_dict.metadata.get("form")
        if form and (form in self.form_names):
            prefix.insert(0, ("prefix", "form", form))
        genre = midi_dict.metadata.get("genre")
        if genre and (genre in self.genre_names):
            prefix.insert(0, ("prefix", "genre", genre))
        random.shuffle(prefix)

        tokenized_seq: list[Token] = []

        if self.include_pedal is True:
            _msgs = midi_dict.note_msgs + midi_dict.pedal_msgs
        else:
            _msgs = midi_dict.note_msgs

        _msgs.sort(
            key=lambda msg: (
                msg["data"]["start"] if msg["type"] == "note" else msg["tick"]
            )
        )

        if remove_preceding_silence is False:
            initial_onset_tick = 0
        else:
            initial_onset_tick = _msgs[0]["tick"]

        curr_time_since_onset = 0
        for _, msg in enumerate(_msgs):
            # Extract msg data
            if msg["type"] == "note":
                _type = "note"
                _channel = msg["channel"]
                _pitch = msg["data"]["pitch"]
                _velocity = msg["data"]["velocity"]
                _pedal_data = None
                _start_tick = msg["data"]["start"]
                _end_tick = msg["data"]["end"]
            elif msg["type"] == "pedal":
                _type = "pedal"
                _channel = msg["channel"]
                _pitch = None
                _velocity = None
                _pedal_data = msg["data"]
                _start_tick = msg["tick"]
                _end_tick = None

            # Calculate time data
            prev_time_since_onset = curr_time_since_onset
            curr_time_since_onset = get_duration_ms(
                start_tick=initial_onset_tick,
                end_tick=_start_tick,
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=ticks_per_beat,
            )

            # Add abs time token if necessary
            time_toks_to_append = (
                curr_time_since_onset // self.abs_time_step_ms
            ) - (prev_time_since_onset // self.abs_time_step_ms)
            if time_toks_to_append > 0:
                for _ in range(time_toks_to_append):
                    tokenized_seq.append(self.time_tok)

            # Special case instrument is a drum. This occurs exclusively when
            # MIDI channel is 9 when 0 indexing
            if _channel == 9:
                _note_onset = self._quantize_onset(
                    curr_time_since_onset % self.abs_time_step_ms
                )
                tokenized_seq.append(("drum", _pitch))
                tokenized_seq.append(("onset", _note_onset))

            elif _type == "pedal":
                _pedal_onset = self._quantize_onset(
                    curr_time_since_onset % self.abs_time_step_ms
                )
                if _pedal_data == 1:
                    tokenized_seq.append(self.ped_on_tok)
                    tokenized_seq.append(("onset", _pedal_onset))
                elif _pedal_data == 0:
                    tokenized_seq.append(self.ped_off_tok)
                    tokenized_seq.append(("onset", _pedal_onset))
                else:
                    raise ValueError("Invalid pedal message")

            else:  # Non drum case (i.e. an instrument note)
                _instrument = channel_to_instrument[_channel]
                assert _velocity is not None
                assert _end_tick is not None

                # Update _end_tick if affected by pedal
                for pedal_interval in channel_to_pedal_intervals[_channel]:
                    pedal_start, pedal_end = (
                        pedal_interval[0],
                        pedal_interval[1],
                    )
                    if pedal_start < _end_tick < pedal_end:
                        _end_tick = pedal_end
                        break

                _note_duration = get_duration_ms(
                    start_tick=_start_tick,
                    end_tick=_end_tick,
                    tempo_msgs=midi_dict.tempo_msgs,
                    ticks_per_beat=ticks_per_beat,
                )

                _velocity = self._quantize_velocity(_velocity)
                _note_onset = self._quantize_onset(
                    curr_time_since_onset % self.abs_time_step_ms
                )
                _note_duration = self._quantize_dur(_note_duration)

                tokenized_seq.append((_instrument, _pitch, _velocity))
                tokenized_seq.append(("onset", _note_onset))
                tokenized_seq.append(("dur", _note_duration))

        return self._format(
            prefix=prefix,
            unformatted_seq=tokenized_seq,
            add_dim_tok=add_dim_tok,
            add_eos_tok=add_eos_tok,
        )

    def tokenize(
        self,
        midi_dict: MidiDict,
        remove_preceding_silence: bool = True,
        add_dim_tok: bool = True,
        add_eos_tok: bool = True,
        **kwargs: Any,
    ) -> list[Token]:
        """Tokenizes a MidiDict object into a sequence.

        Args:
            midi_dict (MidiDict): The MidiDict to tokenize.
            remove_preceding_silence (bool): If true starts the sequence at
                onset=0ms by removing preceding silence. Defaults to True.
            add_dim_tok (bool): Add diminish token if appropriate. Defaults to
                True.
            add_dim_tok (bool): Append end of sequence token. Defaults to True.

        Returns:
            list[Token]: A sequence of tokens representing the MIDI content.
        """

        return self._tokenize_midi_dict(
            midi_dict=midi_dict,
            remove_preceding_silence=remove_preceding_silence,
            add_dim_tok=add_dim_tok,
            add_eos_tok=add_eos_tok,
        )

    def _detokenize_midi_dict(self, tokenized_seq: list[Token]) -> MidiDict:
        # NOTE: These values chosen so that 1000 ticks = 1000ms, allowing us to
        # skip converting between ticks and ms
        instrument_programs = self.config["instrument_programs"]
        TICKS_PER_BEAT: Final[int] = 500
        TEMPO: Final[int] = 500000

        tempo_msgs: list[TempoMessage] = [
            {"type": "tempo", "data": TEMPO, "tick": 0}
        ]
        meta_msgs: list[MetaMessage] = []
        pedal_msgs: list[PedalMessage] = []
        instrument_msgs: list[InstrumentMessage] = []

        instrument_to_channel: dict[str, int] = {}

        # Add non-drum instrument_msgs, breaks at first note token
        channel_idx = 0
        curr_tick = 0
        for idx, tok in enumerate(tokenized_seq):
            if channel_idx == 9:  # Skip channel reserved for drums
                channel_idx += 1

            if tok in self.special_tokens:
                if tok == self.time_tok:
                    curr_tick += self.abs_time_step_ms
                continue
            elif (
                tok[0] == "prefix"
                and tok[1] == "instrument"
                and tok[2] in self.instruments_wd
            ):
                # Process instrument prefix tokens
                if tok[2] in instrument_to_channel.keys():
                    logger.warning(f"Duplicate prefix {tok[2]}")
                    continue
                elif tok[2] == "drum":
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": 0,
                            "tick": 0,
                            "channel": 9,
                        }
                    )
                    instrument_to_channel["drum"] = 9
                else:
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": instrument_programs[tok[2]],
                            "tick": 0,
                            "channel": channel_idx,
                        }
                    )
                    instrument_to_channel[tok[2]] = channel_idx
                    channel_idx += 1
            elif tok in self.prefix_tokens:
                continue
            else:
                # Note, wait, or duration token
                start = idx
                break

        if self.include_pedal:
            assert len(instrument_msgs) == 1
            assert instrument_msgs[0]["data"] == 0  # Piano

        # Note messages
        note_msgs: list[NoteMessage] = []
        for tok_1, tok_2, tok_3 in zip(
            tokenized_seq[start:],
            tokenized_seq[start + 1 :],
            tokenized_seq[start + 2 :],
        ):
            if tok_1 in self.special_tokens:
                _tok_type_1 = "special"
            else:
                _tok_type_1 = tok_1[0]
            if tok_2 in self.special_tokens:
                _tok_type_2 = "special"
            else:
                _tok_type_2 = tok_2[0]
            if tok_3 in self.special_tokens:
                _tok_type_3 = "special"
            else:
                _tok_type_3 = tok_3[0]

            if tok_1 == self.time_tok:
                curr_tick += self.abs_time_step_ms
            elif (
                _tok_type_1 == "special"
                or _tok_type_1 == "prefix"
                or _tok_type_1 == "onset"
                or _tok_type_1 == "dur"
            ):
                continue
            elif tok_1 in {self.ped_on_tok, self.ped_off_tok}:
                assert isinstance(
                    tok_2[1], int
                ), f"Expected int for onset, got {tok_2[1]}"

                _data = 1 if tok_1 == self.ped_on_tok else 0
                _tick: int = curr_tick + tok_2[1]
                pedal_msgs.append(
                    {
                        "type": "pedal",
                        "data": _data,
                        "tick": _tick,
                        "channel": 0,
                    }
                )
            elif _tok_type_1 == "drum" and _tok_type_2 == "onset":
                assert isinstance(
                    tok_2[1], int
                ), f"Expected int for onset, got {tok_2[1]}"
                assert isinstance(
                    tok_1[1], int
                ), f"Expected int for pitch, got {tok_1[1]}"

                _pitch: int = tok_1[1]
                _channel = instrument_to_channel["drum"]
                _velocity: int = self.config["drum_velocity"]
                _start_tick: int = curr_tick + tok_2[1]
                _end_tick: int = _start_tick + self.time_step_ms

                if "drum" not in instrument_to_channel.keys():
                    logger.warning(
                        f"Tried to decode note message for unexpected instrument: drum"
                    )
                else:
                    _channel = instrument_to_channel["drum"]
                    note_msgs.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": _pitch,
                                "start": _start_tick,
                                "end": _end_tick,
                                "velocity": _velocity,
                            },
                            "tick": _start_tick,
                            "channel": _channel,
                        }
                    )

            elif (
                _tok_type_1 in self.instruments_nd
                and _tok_type_2 == "onset"
                and _tok_type_3 == "dur"
            ):
                assert isinstance(
                    tok_1[0], str
                ), f"Expected str for instrument, got {tok_1[0]}"
                assert isinstance(
                    tok_1[1], int
                ), f"Expected int for pitch, got {tok_1[1]}"
                assert isinstance(
                    tok_1[2], int
                ), f"Expected int for velocity, got {tok_1[2]}"
                assert isinstance(
                    tok_2[1], int
                ), f"Expected int for onset, got {tok_2[1]}"
                assert isinstance(
                    tok_3[1], int
                ), f"Expected int for duration, got {tok_3[1]}"

                _instrument = tok_1[0]
                _pitch = tok_1[1]
                _velocity = tok_1[2]
                _start_tick = curr_tick + tok_2[1]
                _end_tick = _start_tick + tok_3[1]

                if _instrument not in instrument_to_channel.keys():
                    logger.warning(
                        f"Tried to decode note message for unexpected instrument: {_instrument} "
                    )
                else:
                    _channel = instrument_to_channel[_instrument]
                    note_msgs.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": _pitch,
                                "start": _start_tick,
                                "end": _end_tick,
                                "velocity": _velocity,
                            },
                            "tick": _start_tick,
                            "channel": _channel,
                        }
                    )
            else:
                logger.warning(
                    f"Unexpected token sequence: {tok_1}, {tok_2}, {tok_3}"
                )

        return MidiDict(
            meta_msgs=meta_msgs,
            tempo_msgs=tempo_msgs,
            pedal_msgs=pedal_msgs,
            instrument_msgs=instrument_msgs,
            note_msgs=note_msgs,
            ticks_per_beat=TICKS_PER_BEAT,
            metadata={},
        )

    def detokenize(self, tokenized_seq: list[Token], **kwargs: Any) -> MidiDict:
        """Detokenizes a MidiDict object.

        Args:
            tokenized_seq (list): The sequence of tokens to detokenize.

        Returns:
            MidiDict: A MidiDict reconstructed from the tokens.
        """

        return self._detokenize_midi_dict(tokenized_seq=tokenized_seq)

    def export_pitch_aug(
        self, max_pitch_aug: int
    ) -> Callable[..., list[Token]]:
        """Exports a function that augments the pitch of all note tokens.

        Notes which fall out of the range (0, 127) will be replaced
        with the unknown token '<U>'.

        Args:
            max_pitch_aug (int): Returned function will randomly augment the pitch
                from a value in the range (-max_pitch_aug, max_pitch_aug).

        Returns:
            Callable[[list[Token], int], list[Token]]: Exported function.
        """

        def pitch_aug_seq(
            src: list[Token],
            unk_tok: str,
            _max_pitch_aug: int,
            pitch_aug: int | None = None,
        ) -> list[Token]:
            def pitch_aug_tok(tok: Token, _pitch_aug: int) -> Token:
                if isinstance(tok, str):  # Stand in for SpecialToken
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "onset"
                ):
                    # Return without changing
                    return tok
                else:
                    # Return augmented tok
                    assert (
                        isinstance(tok, tuple) and len(tok) == 3
                    ), f"Invalid note token"
                    (_instrument, _pitch, _velocity) = tok

                    assert isinstance(
                        _pitch, int
                    ), f"Expected int for pitch, got {_pitch}"
                    assert isinstance(
                        _velocity, int
                    ), f"Expected int for velocity, got {_velocity}"

                    if 0 <= _pitch + _pitch_aug <= 127:
                        return (_instrument, _pitch + _pitch_aug, _velocity)
                    else:
                        return unk_tok

            if pitch_aug is None:
                pitch_aug = random.randint(-_max_pitch_aug, _max_pitch_aug)

            return [pitch_aug_tok(x, pitch_aug) for x in src]

        return self.export_aug_fn_concat(
            functools.partial(
                pitch_aug_seq,
                unk_tok=self.unk_tok,
                _max_pitch_aug=max_pitch_aug,
            )
        )

    def export_velocity_aug(
        self, max_num_aug_steps: int
    ) -> Callable[..., list[Token]]:
        """Exports a function which augments the velocity of all pitch tokens.

        Velocity values are clipped so that they don't fall outside of the
        valid range.

        Args:
            max_num_aug_steps (int): Returned function will randomly augment
                velocity in the range self.velocity_step * (-max_num_aug_steps,
                max_num_aug_steps).

        Returns:
            Callable[[list[Token], int], list[Token]]: Exported function.
        """

        def velocity_aug_seq(
            src: list[Token],
            min_velocity_step: int,
            max_velocity: int,
            _max_num_aug_steps: int,
            aug_step: int | None = None,
        ) -> list[Token]:
            def velocity_aug_tok(tok: Token, _velocity_aug: int) -> Token:
                if isinstance(tok, str):  # Stand in for SpecialToken
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "onset"
                ):
                    # Return without changing
                    return tok
                else:
                    assert isinstance(tok, tuple) and len(tok) == 3
                    (_instrument, _pitch, _velocity) = tok

                    assert isinstance(_pitch, int)
                    assert isinstance(_velocity, int)

                    # Check it doesn't go out of bounds
                    if _velocity + _velocity_aug >= max_velocity:
                        return (_instrument, _pitch, max_velocity)
                    elif _velocity + _velocity_aug <= min_velocity_step:
                        return (_instrument, _pitch, min_velocity_step)

                    return (_instrument, _pitch, _velocity + _velocity_aug)

            if aug_step is None:
                velocity_aug = min_velocity_step * random.randint(
                    -_max_num_aug_steps, _max_num_aug_steps
                )
            else:
                velocity_aug = aug_step * min_velocity_step

            return [velocity_aug_tok(x, velocity_aug) for x in src]

        return self.export_aug_fn_concat(
            functools.partial(
                velocity_aug_seq,
                min_velocity_step=self.velocity_step,
                max_velocity=self.max_velocity,
                _max_num_aug_steps=max_num_aug_steps,
            )
        )

    def export_tempo_aug(
        self, max_tempo_aug: float, mixup: bool
    ) -> Callable[..., list[Token]]:
        """Exports a function which augments the tempo of a sequence of tokens.

        Additionally this function performs note-mixup: randomly re-ordering
        the note subsequences which occur on the same onset.

        This version supports variable-length token events, including notes,
        drums, and pedal events.

        Args:
            max_tempo_aug (float): Returned function will randomly augment
                tempo by a factor in the range (1 - max_tempo_aug,
                1 + max_tempo_aug).

        Returns:
            Callable[[list[Token], float], list[Token]]: Exported function.
        """

        def tempo_aug(
            src: list[Token],
            abs_time_step: int,
            max_dur: int,
            time_step: int,
            bos_tok: str,
            eos_tok: str,
            time_tok: str,
            dim_tok: str,
            pad_tok: str,
            unk_tok: str,
            ped_on_tok: str,
            ped_off_tok: str,
            instruments_wd: list[str],
            _max_tempo_aug: float,
            _mixup: bool,
            tempo_aug: float | None = None,
        ) -> list[Token]:
            """This must be used with export_aug_fn_concat in order to work
            properly for concatenated sequences."""

            def _quantize_time(_n: int | float) -> int:
                return round(_n / time_step) * time_step

            if tempo_aug is None:
                tempo_aug = random.uniform(
                    1 - _max_tempo_aug, 1 + _max_tempo_aug
                )

            # Buffer to hold all events, grouped by time
            # buffer[time_tok_count][onset_ms] = [ event_1, event_2, ... ]
            # where event is a list of tokens, e.g. [note, onset, dur]
            buffer: defaultdict[int, defaultdict[int, list[list[Token]]]] = (
                defaultdict(lambda: defaultdict(list))
            )

            res_prefix: list[Token] = []
            src_time_tok_cnt = 0
            dim_tok_seen_at: tuple[int, int] | None = None
            eos_tok_seen: bool = False

            idx = 0
            while idx < len(src):
                tok = src[idx]
                is_tuple = isinstance(tok, tuple)
                if tok is bos_tok or (is_tuple and tok[0] == "prefix"):
                    res_prefix.append(tok)
                    idx += 1
                else:
                    break

            while idx < len(src):
                event_subsequence = []
                tok = src[idx]

                if tok == time_tok:
                    src_time_tok_cnt += 1
                    idx += 1
                    continue

                if tok is eos_tok:
                    eos_tok_seen = True
                    idx += 1
                    continue
                elif tok in {unk_tok, pad_tok}:
                    idx += 1
                    continue

                # Handle <D> token by attaching it to the previous event's time
                if tok == dim_tok:
                    if dim_tok_seen_at is not None:
                        logger.warning(
                            "Multiple <D> tokens encountered in augmentation"
                        )
                    # Find the last buffered event to get its time
                    last_time = max(buffer.keys()) if buffer else 0
                    last_onset = (
                        max(buffer[last_time].keys())
                        if buffer.get(last_time)
                        else 0
                    )
                    dim_tok_seen_at = (last_time, last_onset)
                    idx += 1
                    continue

                # Parse event sequences (note, drum, pedal)
                tok_type = tok[0] if isinstance(tok, tuple) else tok
                current_onset = -1

                if (
                    tok_type in instruments_wd and tok_type != "drum"
                ):  # Note Event: 3 tokens
                    event_subsequence = src[idx : idx + 3]
                    idx += 3
                    if (
                        len(event_subsequence) < 3
                        or event_subsequence[1][0] != "onset"
                        or event_subsequence[2][0] != "dur"
                    ):
                        logger.warning(
                            f"Malformed sequence: {event_subsequence}"
                        )
                        continue
                    current_onset = event_subsequence[1][1]
                elif tok_type == "drum":  # Drum Event: 2 tokens
                    event_subsequence = src[idx : idx + 2]
                    idx += 2
                    if (
                        len(event_subsequence) < 2
                        or event_subsequence[1][0] != "onset"
                    ):
                        logger.warning(
                            f"Malformed sequence: {event_subsequence}"
                        )
                        continue
                    current_onset = event_subsequence[1][1]
                elif tok_type in {
                    ped_on_tok,
                    ped_off_tok,
                }:  # Pedal Event: 2 tokens
                    event_subsequence = src[idx : idx + 2]
                    idx += 2
                    if (
                        len(event_subsequence) < 2
                        or event_subsequence[1][0] != "onset"
                    ):
                        logger.warning(
                            f"Malformed sequence: {event_subsequence}"
                        )
                        continue
                    current_onset = event_subsequence[1][1]
                else:
                    idx += 1
                    continue

                if current_onset != -1:
                    buffer[src_time_tok_cnt][current_onset].append(
                        event_subsequence
                    )

            res_events: list[Token] = []
            prev_tgt_time_tok_cnt = 0

            for src_time_tok_cnt, interval_events in sorted(buffer.items()):
                for src_onset, events_at_onset in sorted(
                    interval_events.items()
                ):
                    src_time = src_time_tok_cnt * abs_time_step + src_onset
                    tgt_time = _quantize_time(src_time * tempo_aug)

                    curr_tgt_time_tok_cnt = tgt_time // abs_time_step
                    curr_tgt_onset = tgt_time % abs_time_step

                    # Add necessary <T> tokens
                    for _ in range(
                        curr_tgt_time_tok_cnt - prev_tgt_time_tok_cnt
                    ):
                        res_events.append(time_tok)
                    prev_tgt_time_tok_cnt = curr_tgt_time_tok_cnt

                    if _mixup:
                        random.shuffle(events_at_onset)

                    # Process and append all events for this timestamp
                    for event in events_at_onset:
                        first_tok = event[0]
                        # Note event with duration
                        if len(event) == 3:
                            _src_dur_tok = event[2]
                            tgt_dur = _quantize_time(
                                _src_dur_tok[1] * tempo_aug
                            )
                            tgt_dur = max(
                                min(tgt_dur, max_dur),
                                time_step,
                            )
                            res_events.extend(
                                [
                                    first_tok,
                                    ("onset", curr_tgt_onset),
                                    ("dur", tgt_dur),
                                ]
                            )
                        elif len(event) == 2:
                            res_events.extend(
                                [first_tok, ("onset", curr_tgt_onset)]
                            )

                    if dim_tok_seen_at == (src_time_tok_cnt, src_onset):
                        res_events.append(dim_tok)
                        dim_tok_seen_at = None

            # Re-assemble the final sequence
            final_res = res_prefix + res_events

            if eos_tok_seen is True:
                final_res.append(eos_tok)

            return final_res

        return self.export_aug_fn_concat(
            functools.partial(
                tempo_aug,
                abs_time_step=self.abs_time_step_ms,
                max_dur=self.max_dur_ms,
                time_step=self.time_step_ms,
                bos_tok=self.bos_tok,
                eos_tok=self.eos_tok,
                time_tok=self.time_tok,
                dim_tok=self.dim_tok,
                pad_tok=self.pad_tok,
                unk_tok=self.unk_tok,
                ped_on_tok=self.ped_on_tok,
                ped_off_tok=self.ped_off_tok,
                instruments_wd=self.instruments_wd,
                _max_tempo_aug=max_tempo_aug,
                _mixup=mixup,
            )
        )
