import spacy_alignments as tokenizations
from transformers import AutoTokenizer
from sudachipy import Dictionary, SplitMode

class NerTokenizer:
    def __init__(self, hf_tokenizer_name: str):
        """
        Initializes the NerTokenizer.

        Args:
            hf_tokenizer_name (str): The name of the Hugging Face tokenizer.
        """
        print(f"Initializing tokenizer '{hf_tokenizer_name}' and Sudachi...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)

        # Initialize Sudachi
        dict_obj = Dictionary()
        self.sudachi = dict_obj.create()

    def sudachi_tokenizer(self, text: str) -> list[str]:
        """Tokenizes text into words using Sudachi."""
        return [m.surface() for m in self.sudachi.tokenize(text, SplitMode.A)]

    def tokenize_and_align_labels(self, examples: dict, label2id: dict):
        """
        Tokenizes text and aligns labels for NER.
        This function is adapted from the notebook and contains the core preprocessing logic.
        """
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for i, text in enumerate(examples["text"]):
            entities = examples["entities"][i]

            # 1. Sudachi word tokenization
            words = self.sudachi_tokenizer(text)

            # 2. Hugging Face subword tokenization
            tokenized_inputs = self.tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                return_offsets_mapping=True,
                return_attention_mask=True,
                add_special_tokens=True
            )

            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]

            # 3. Align words and subwords
            subwords_for_alignment = self.tokenizer.convert_ids_to_tokens(
                self.tokenizer(words, is_split_into_words=True, add_special_tokens=False).input_ids
            )
            words2subwords, subwords2words = tokenizations.get_alignments(words, subwords_for_alignment)

            # 4. Align entity labels to subwords
            aligned_labels = [-100] * len(input_ids)
            word_labels = ["O"] * len(words)

            word_char_spans = []
            current_char_index = 0
            for word_text in words:
                start_index = text.find(word_text, current_char_index)
                if start_index != -1:
                    end_index = start_index + len(word_text)
                    word_char_spans.append((start_index, end_index))
                    current_char_index = end_index
                else:
                    word_char_spans.append((-1, -1))

            for ent in entities:
                ent_start_char, ent_end_char = ent["span"]
                ent_type = ent["type"]
                for word_index, (word_char_start, word_char_end) in enumerate(word_char_spans):
                    if word_char_start != -1 and max(word_char_start, ent_start_char) < min(word_char_end, ent_end_char):
                        is_entity_start = True # Simplified logic, assuming first word in span is B-
                        # A more robust check might be needed for complex cases
                        if any(word_char_start > s for s, e in word_char_spans if e <= ent_start_char):
                            is_entity_start = False

                        word_labels[word_index] = f"B-{ent_type}" if is_entity_start else f"I-{ent_type}"

            # Transfer word-level labels to subword-level labels
            for subword_index in range(len(input_ids)):
                if self.tokenizer.convert_ids_to_tokens(input_ids[subword_index]) in self.tokenizer.all_special_tokens:
                    aligned_labels[subword_index] = -100
                    continue

                subword_alignment_index = subword_index - 1  # Adjust for [CLS]
                if 0 <= subword_alignment_index < len(subwords2words):
                    aligned_word_indices = subwords2words[subword_alignment_index]
                    if aligned_word_indices:
                        first_aligned_word_index = aligned_word_indices[0]
                        if first_aligned_word_index < len(word_labels):
                            word_level_label = word_labels[first_aligned_word_index]

                            is_first_subword_of_word = (
                                first_aligned_word_index < len(words2subwords) and
                                subword_alignment_index in words2subwords[first_aligned_word_index] and
                                words2subwords[first_aligned_word_index][0] == subword_alignment_index
                            )

                            if word_level_label != "O":
                                if is_first_subword_of_word:
                                    aligned_labels[subword_index] = label2id[word_level_label]
                                else:
                                    if word_level_label.startswith("B-"):
                                        aligned_labels[subword_index] = label2id[f"I-{word_level_label.split('-', 1)[1]}"]
                                    else:
                                        aligned_labels[subword_index] = label2id[word_level_label]
                            else:
                                aligned_labels[subword_index] = label2id["O"]
                        else:
                             aligned_labels[subword_index] = label2id["O"]
                    else:
                        aligned_labels[subword_index] = label2id["O"]
                else:
                    aligned_labels[subword_index] = -100

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(aligned_labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels
        }
