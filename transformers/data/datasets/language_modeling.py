import os
import pickle
import random
import time
from typing import Dict, List, Optional
import torch
from torch.utils.data.dataset import Dataset
from filelock import FileLock
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False, cache_dir: Optional[str] = None):
        assert os.path.isfile(file_path), f'Input file path {file_path} not found'
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        (directory, filename) = os.path.split(file_path)
        cached_features_file = os.path.join((cache_dir if cache_dir is not None else directory), 'cached_lm_{}_{}_{}'.format(tokenizer.__class__.__name__, str(block_size), filename))
        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if (os.path.exists(cached_features_file) and not overwrite_cache):
                start = time.time()
                with open(cached_features_file, 'rb') as handle:
                    self.examples = pickle.load(handle)
                logger.info(f'Loading features from cached file {cached_features_file} [took %.3f s]', time.time() - start)
            else:
                logger.info(f'Creating features from dataset file at {directory}')
                self.examples = []
                with open(file_path, encoding='utf-8') as f:
                    text = f.read()
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))
                start = time.time()
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info('Saving features into cached file %s [took %.3f s]', cached_features_file, time.time() - start)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)



class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f'Input file path {file_path} not found'
        logger.info('Creating features from dataset file at %s', file_path)
        with open(file_path, encoding='utf-8') as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding['input_ids']
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)



class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        assert os.path.isdir(file_dir)
        logger.info(f'Creating features from dataset file folder at {file_dir}')
        self.examples = []
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            assert os.path.isfile(file_path)
            article_open = False
            with open(file_path, encoding='utf-8') as f:
                original_lines = f.readlines()
                article_lines = []
                for line in original_lines:
                    if '<doc id=' in line:
                        article_open = True
                    elif '</doc>' in line:
                        article_open = False
                        document = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)) for line in article_lines[1:] if (len(line) > 0 and not line.isspace())]
                        examples = self.create_examples_from_document(document, block_size, tokenizer)
                        self.examples.extend(examples)
                        article_lines = []
                    elif article_open:
                        article_lines.append(line)
        logger.info('Dataset parse finished.')
    
    def create_examples_from_document(self, document, block_size, tokenizer, short_seq_prob=0.1):
        """Creates examples for a single document."""
        max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        examples = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            if not segment:
                i += 1
                continue
            current_chunk.append(segment)
            current_length += len(segment)
            if (i == len(document) - 1 or current_length >= target_seq_length):
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    tokens_b = []
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    if (len(tokens_a) == 0 or len(tokens_b) == 0):
                        continue
                    if random.random() < 0.5:
                        is_next = False
                        (tokens_a, tokens_b) = (tokens_b, tokens_a)
                    else:
                        is_next = True
                    
                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = (tokens_a if len(tokens_a) > len(tokens_b) else tokens_b)
                            assert len(trunc_tokens) >= 1
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()
                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
                    example = {'input_ids': torch.tensor(input_ids, dtype=torch.long), 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long), 'sentence_order_label': torch.tensor((0 if is_next else 1), dtype=torch.long)}
                    examples.append(example)
                current_chunk = []
                current_length = 0
            i += 1
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i) -> Dict[(str, torch.tensor)]:
        return self.examples[i]



class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False, short_seq_probability=0.1, nsp_probability=0.5):
        assert os.path.isfile(file_path), f'Input file path {file_path} not found'
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability
        (directory, filename) = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_nsp_{}_{}_{}'.format(tokenizer.__class__.__name__, str(block_size), filename))
        self.tokenizer = tokenizer
        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if (os.path.exists(cached_features_file) and not overwrite_cache):
                start = time.time()
                with open(cached_features_file, 'rb') as handle:
                    self.examples = pickle.load(handle)
                logger.info(f'Loading features from cached file {cached_features_file} [took %.3f s]', time.time() - start)
            else:
                logger.info(f'Creating features from dataset file at {directory}')
                self.documents = [[]]
                with open(file_path, encoding='utf-8') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if (not line and len(self.documents[-1]) != 0):
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)
                logger.info(f'Creating examples from {len(self.documents)} documents.')
                self.examples = []
                for (doc_index, document) in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index)
                start = time.time()
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info('Saving features into cached file %s [took %.3f s]', cached_features_file, time.time() - start)
    
    def create_examples_from_document(self, document: List[List[int]], doc_index: int):
        """Creates examples for a single document."""
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if (i == len(document) - 1 or current_length >= target_seq_length):
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    tokens_b = []
                    if (len(current_chunk) == 1 or random.random() < self.nsp_probability):
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)
                        for _ in range(10):
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index:
                                break
                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    self.examples.append({'tokens_a': tokens_a, 'tokens_b': tokens_b, 'is_random_next': is_random_next})
                current_chunk = []
                current_length = 0
            i += 1
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]


