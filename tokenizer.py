import logging
from typing import Dict, List, Optional
from ariautils.midi import MidiDict
from ariautils.tokenizer.absolute import AbsTokenizer

# GLOBAL CONSTANTS
IGNORE_LABEL_IDX = -100  # Loss padding index for ignored tokens
MASK_TOKEN_ID = 32000  # Special token ID for diffusion masking
STYLE_LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_STYLE_MAP = {v: k for k, v in STYLE_LABEL_MAP.items()}
PROMPT_START_TOKEN = '<PROMPT_START>'
PROMPT_END_TOKEN = '<PROMPT_END>'
A_SECTION_TOKEN = '<A_SECTION>'
B_SECTION_TOKEN = '<B_SECTION>'
C_SECTION_TOKEN = '<C_SECTION>'
D_SECTION_TOKEN = '<D_SECTION>'
SPECIAL_TOKENS = [
    PROMPT_START_TOKEN,
    PROMPT_END_TOKEN,
    A_SECTION_TOKEN,
    B_SECTION_TOKEN,
    C_SECTION_TOKEN,
    D_SECTION_TOKEN,
    '<MASK>'  # Added for diffusion
]
SECTION_TOKENS = {
    'A': A_SECTION_TOKEN,
    'B': B_SECTION_TOKEN,
    'C': C_SECTION_TOKEN,
    'D': D_SECTION_TOKEN
}
STYLE_TO_SNIPPET_IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
SUPPORTED_STRUCTURE_FORMS = [
    "A", "AB", "ABC", "ABA", "ABAB", "ABAC", "ABCA", "ABCB", "ABCD"
]
FORM_LABEL_MAP = {form: idx for idx, form in enumerate(SUPPORTED_STRUCTURE_FORMS)}
ID_TO_FORM_MAP = {idx: form for form, idx in FORM_LABEL_MAP.items()}

logger = logging.getLogger(__name__)

class MusicTokenizerWithStyle:
    def __init__(self, config_path: str = "ariautils/config/config.json"):
        self._base_tokenizer = AbsTokenizer(config_path=config_path)
        style_tokens = [A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN]
        prompt_tokens = [PROMPT_START_TOKEN, PROMPT_END_TOKEN]
        form_tokens = [f"<{form}>" for form in SUPPORTED_STRUCTURE_FORMS]
        all_special_tokens = style_tokens + prompt_tokens + form_tokens + ['<MASK>']
        self._base_tokenizer.add_tokens_to_vocab(all_special_tokens)
        self.style_id_to_label = ID_TO_STYLE_MAP
        self.style_label_to_id = STYLE_LABEL_MAP
        self.mask_id = self._base_tokenizer.tok_to_id['<MASK>']
        logger.info(
            f"MusicTokenizerWithStyle initialized | "
            f"vocab_size={self.vocab_size} | pad_id={self.pad_id} | "
            f"bos_id={self.bos_id} | eos_id={self.eos_id} | mask_id={self.mask_id}"
        )

    def tokenize_midi_dict(self, midi_dict: MidiDict) -> List[str]:
        return self._base_tokenizer.tokenize(midi_dict)

    def tokenize_midi_file(self, midi_path: str) -> Optional[List[str]]:
        try:
            midi_dict = MidiDict.from_midi(midi_path)
            tokens = self.tokenize_midi_dict(midi_dict)
            return tokens if tokens else None
        except Exception as exc:
            logger.error(f"Failed to tokenize MIDI file {midi_path}: {exc}")
            return None

    def encode_tokens(self, token_sequence: List[str]) -> List[int]:
        return self._base_tokenizer.encode(token_sequence)

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        return self._base_tokenizer.decode(token_ids)

    def calculate_duration_ms(self, tokens: List[str], use_onset: bool = True) -> int:
        return self._base_tokenizer.calc_length_ms(tokens, onset=use_onset)

    def truncate_tokens_by_time(self, tokens: List[str], max_duration_ms: int) -> List[str]:
        return self._base_tokenizer.truncate_by_time(tokens, max_duration_ms)

    def save_tokens_as_midi(self, token_ids: List[int], output_path: str) -> bool:
        try:
            token_sequence = self.decode_tokens(token_ids)
            midi_dict = self._base_tokenizer.detokenize(token_sequence)
            midi_file = midi_dict.to_midi()
            midi_file.save(output_path)
            logger.info(f"MIDI file saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save MIDI file: {e}", exc_info=False)
            return False

    def remove_instrument_prefix_tokens(self, tokens: List[str]) -> List[str]:
        instrument_token_ids = [0, 6]
        instrument_tokens = [self.id_to_token_map.get(token_id, '') for token_id in instrument_token_ids]
        return [token for token in tokens if token not in instrument_tokens]

    @property
    def vocab_size(self) -> int:
        return self._base_tokenizer.vocab_size

    @property
    def pad_id(self) -> int:
        return self._base_tokenizer.pad_id

    @property
    def bos_id(self) -> Optional[int]:
        return self._base_tokenizer.tok_to_id.get(self._base_tokenizer.bos_tok)

    @property
    def eos_id(self) -> Optional[int]:
        return self._base_tokenizer.tok_to_id.get(self._base_tokenizer.eos_tok)

    @property
    def token_to_id_map(self) -> Dict[str, int]:
        return self._base_tokenizer.tok_to_id

    @property
    def id_to_token_map(self) -> Dict[int, str]:
        return self._base_tokenizer.id_to_tok

    # Legacy methods
    def tokenize(self, midi_dict: MidiDict) -> List[str]:
        return self.tokenize_midi_dict(midi_dict)

    def encode(self, seq: List[str]) -> List[int]:
        return self.encode_tokens(seq)

    def decode(self, seq: List[int]) -> List[str]:
        return self.decode_tokens(seq)

    def tokenize_from_file(self, midi_path: str) -> Optional[List[str]]:
        return self.tokenize_midi_file(midi_path)

    def calc_length_ms(self, tokens: List[str], onset: bool = True) -> int:
        return self.calculate_duration_ms(tokens, use_onset=onset)

    def truncate_by_time(self, tokens: List[str], trunc_time_ms: int) -> List[str]:
        return self.truncate_tokens_by_time(tokens, trunc_time_ms)

    def ids_to_file(self, ids: List[int], output_path: str):
        self.save_tokens_as_midi(ids, output_path)

    def remove_instrument_prefix(self, tokens: List[str]) -> List[str]:
        return self.remove_instrument_prefix_tokens(tokens)

    @property
    def tok_to_id(self) -> Dict[str, int]:
        return self.token_to_id_map

    @property
    def id_to_tok(self) -> Dict[int, str]:
        return self.id_to_token_map