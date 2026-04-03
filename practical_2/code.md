# DAM304 Practical 2: Code Explanation

## Tokenization Techniques and Positional Encoding

---

## Overview

This practical notebook demonstrates four core concepts essential for language models:

1. **Character-Level Tokenization** - Converting text to integer IDs
2. **Byte-Pair Encoding (BPE)** - Subword tokenization through iterative merging
3. **HuggingFace Tokenizer Comparison** - Comparing real-world tokenizers (GPT-2 vs Mistral)
4. **Sinusoidal Positional Encoding** - Adding position information to token embeddings

---

## Task 1: Character-Level Tokenizer

### Purpose

Create a simple tokenizer that converts individual characters to unique integer IDs and can reverse the process.

### What It Does

- **Build Vocabulary**: Maps each unique character to an integer ID
- **Encode**: Converts text strings into sequences of integer IDs
- **Decode**: Converts integer ID sequences back to text strings
- **Special Tokens**: Reserves ID 0 for `[UNK]` (unknown) and ID 1 for `[PAD]` (padding)

### How It Works

```python
class CharTokenizer:
    def __init__(self):
        self.char_to_id = {'[UNK]': 0, '[PAD]': 1}  # Maps chars to IDs
        self.id_to_char = {0: '[UNK]', 1: '[PAD]'}  # Maps IDs back to chars
```

**Step 1: Build Vocabulary**

```python
def build_vocab(self, text):
    next_id = 2  # Start at 2 (0 and 1 are reserved)
    for char in text:
        if char not in self.char_to_id:
            self.char_to_id[char] = next_id
            self.id_to_char[next_id] = char
            next_id += 1
```

- Iterates through each character in the text
- Assigns a unique ID (2, 3, 4, ...) to each new character
- Both mappings (char→ID and ID→char) are stored for bidirectional conversion

**Step 2: Encode Text**

```python
def encode(self, text):
    return [self.char_to_id.get(char, 0) for char in text]
```

- Converts each character to its ID using the vocabulary
- If a character is unknown, it returns 0 (`[UNK]`)
- Result: A list of integers representing the text

**Step 3: Decode IDs**

```python
def decode(self, ids):
    return ''.join([self.id_to_char.get(id_val, '[UNK]') for id_val in ids])
```

- Converts each integer ID back to its character
- If an ID doesn't exist, it returns `'[UNK]'`
- Joins all characters to reconstruct the original text

### Example

```
Input: "The quick brown fox jumps over the lazy dog"

After build_vocab():
- 'T' → 2, 'h' → 3, 'e' → 4, etc.
- Total: 30 unique characters

After encode():
[2, 3, 4, 0, 5, 6, 7, 8, 9, 10, ...]  # 43 tokens total

After decode():
"The quick brown fox jumps over the lazy dog" ✓ Round-trip successful!
```

### Why It Matters

- **Baseline Understanding**: Shows the fundamental tokenization concept
- **Round-Trip Verification**: Ensures encoding-decoding consistency
- **Foundation**: All tokenization approaches build on this character-level concept

---

## Task 2: Byte-Pair Encoding (BPE)

### Purpose

Implement a subword tokenization algorithm that learns which character sequences should be merged together based on frequency.

### What It Does

1. **Training Phase**: Learns which adjacent symbols to merge by finding the most frequent pair and merging iteratively
2. **Tokenization Phase**: Applies learned merges to tokenize new words

### Why BPE?

- Character-level: Too many tokens (e.g., "running" = 7 tokens)
- Word-level: Vocabulary explosion and doesn't handle unknown words
- BPE: **Sweet spot** - uses frequent subwords (e.g., "running" = "run" + "ing" = 2 tokens)

### How It Works

**The Training Loop**

```python
CORPUS = [
    ('l o w </w>', 5),          # "low" appears 5 times
    ('l o w e r </w>', 2),      # "lower" appears 2 times
    ('n e w e s t </w>', 6),    # "newest" appears 6 times
    ('w i d e s t </w>', 3),    # "widest" appears 3 times
]
```

- Each word is split into characters with spaces between them
- `</w>` marks the end of a word (word boundary)
- Each word has a frequency count (how often it appears in training data)

**Step 1: Count Adjacent Pairs**

```python
def get_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()  # Split by spaces
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])  # Find all adjacent pairs
            pairs[pair] += freq  # Add frequency count
    return pairs
```

- Splits each word into individual symbols
- For each pair of adjacent symbols, increments its count by the word's frequency
- Returns a Counter with all pairs and their total frequencies

**Step 2: Merge Most Frequent Pair**

```python
def merge_vocab(best_pair, vocab):
    new_vocab = {}
    symbol_a, symbol_b = best_pair
    bigram = re.escape(symbol_a + ' ' + symbol_b)
    replacement = symbol_a + symbol_b  # Merge without space

    for word, freq in vocab.items():
        # Replace "a b" with "ab" everywhere in the word
        new_word = re.sub(bigram, replacement, word)
        new_vocab[new_word] = freq

    return new_vocab
```

- Takes the most frequent pair from `get_pairs()`
- Replaces all occurrences of that pair throughout the vocabulary
- Example: If ("l", "o") is most frequent, replace "l o" with "lo" in all words

**Example Iteration:**

```
Initial: {'l o w </w>': 5, 'l o w e r </w>': 2, ...}

Merge 1: ('e', 's') is most frequent (9 times total)
After:   {'l o w </w>': 5, 'l o w e r </w>': 2, 'n ew est </w>': 6, ...}

Merge 2: ('es', 't') is now most frequent (9 times)
After:   {'l o w </w>': 5, 'l o w e r </w>': 2, 'n ew est </w>': 6, ...}

...continues for NUM_MERGES (10 times)
```

**Step 3: Tokenize New Words**

```python
def tokenize(word, vocab):
    # Convert word to character-level with spaces
    word_tokens = ' '.join(list(word)) + ' </w>'
    symbols = word_tokens.split()

    # Keep applying learned merges
    while len(symbols) > 1:
        # Find which learned pairs exist in current symbols
        best_pair = None
        for word_in_vocab in vocab.keys():
            word_symbols = word_in_vocab.split()
            for i in range(len(word_symbols) - 1):
                pair = (word_symbols[i], word_symbols[i + 1])
                if pair in [current adjacent pairs]
                    best_pair = pair

        if best_pair is None:
            break

        # Merge the best pair
        symbols = merge_adjacent(symbols, best_pair)

    return symbols
```

- Breaks new word into characters (e.g., "newer" → "n e w e r")
- Finds which learned pairs from training exist in these characters
- Merges them iteratively until all learned merges are applied

**Example:**

```
Tokenizing "newer":

Start:       n e w e r </w>
After merge ("n", "e"): ne w e r </w>  (if this pair was learned)
After merge ("e", "r"): ne w er </w>
...
Result:      ['n', 'ew', 'er', '</w>']
```

### Why This Works

- **Compression**: Reduces vocabulary size while preserving meaning
- **Flexibility**: Handles unknown words by breaking them into known subwords
- **Frequency-Based**: Learns what merges are useful from training data

---

## Task 3: HuggingFace Tokenizers - GPT-2 vs Mistral

### Purpose

Compare two real-world production tokenizers to understand how different models tokenize the same text.

### What It Does

1. **Loads Pre-trained Tokenizers** from HuggingFace model hub
2. **Tests Multiple Sentences** across different domains
3. **Compares Tokenization Results** to show differences
4. **Verifies Round-Trip** to ensure encoding-decoding consistency

### How It Works

**Loading Tokenizers**

```python
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
llama_tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
```

- Downloads pre-trained tokenizers from HuggingFace
- GPT-2: ~50,257 vocab size
- Mistral: Different vocab size and merge patterns

**Tokenization Comparison**

```python
gpt2_tokens = gpt2_tokenizer.tokenize(sentence)
gpt2_ids = gpt2_tokenizer.encode(sentence)
gpt2_decoded = gpt2_tokenizer.decode(gpt2_ids)
```

- **tokenize()**: Returns human-readable tokens (with special characters like 'Ġ' for spaces)
- **encode()**: Returns integer IDs
- **decode()**: Converts IDs back to text

### Example Findings

**Sentence**: "The Bhutanese government adopted a Gross National Happiness index."

```
GPT-2 produces: ['The', 'ĠBhutanese', 'Ġgovernment', ...]
Tokens: 15

Mistral produces: Different tokenization

Difference: Different training data → different vocab → different merges
```

**Special Characters**

- 'Ġ' (GPT-2): Represents a space before a word
- '▁' (Mistral): Different representation of spaces
- These help the tokenizer mark word boundaries

### Key Insights

1. **Vocabulary Size Matters**
   - Larger vocabulary → fewer tokens needed
   - Fewer tokens → less context window consumed
   - But: Larger vocabulary means more expensive model computations

2. **Training Data Impact**
   - Models trained on different data have different vocabularies
   - Specialized vocabulary for technical terms might differ
   - Example: Coding vs natural language models tokenize code differently

3. **Round-Trip Verification**
   - Both tokenizers must preserve text through encode-decode cycle
   - Verifies no information is lost during tokenization

---

## Task 4: Sinusoidal Positional Encoding

### Purpose

Add position information to token embeddings so the transformer model understands the order of tokens.

### Why Position Encoding is Critical

**The Problem:**

- Self-attention in transformers is **permutation-invariant**
- All attention layers see tokens in any order as equivalent
- "The dog bit the man" = "The man bit the dog" without position info

**The Solution:**

- Add a unique vector for each position
- Transformer learns to combine token embedding + position embedding

### What It Does

Creates a 2D matrix where:

- Each **row** represents a token position (0 to 99)
- Each **column** represents an embedding dimension (0 to 63)
- Each value is computed using sine and cosine functions

### How It Works

**The Formula**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:

- `pos`: Token position (0, 1, 2, ..., 99)
- `i`: Dimension index (0, 1, 2, ..., 31)
- `d_model`: Total embedding dimension (64)
- Even columns use **sine**, odd columns use **cosine**

**Implementation**

```python
def positional_encoding(max_seq_len, d_model):
    PE = np.zeros((max_seq_len, d_model))

    # Create position matrix: shape (max_seq_len, 1)
    pos = np.arange(max_seq_len)[:, np.newaxis]
    # pos = [[0], [1], [2], ..., [99]]

    # Create dimension indices for even columns: shape (1, 32)
    i = np.arange(0, d_model, 2)[np.newaxis, :]
    # i = [[0, 2, 4, ..., 62]]

    # Compute the oscillation frequency: 10000^(i/d_model)
    # Higher dimensions oscillate faster
    div_term = 10000 ** (i / d_model)

    # Fill even columns with sine
    PE[:, 0::2] = np.sin(pos / div_term)
    # PE[5, 0] = sin(5 / 10000^(0/64)) = sin(5)
    # PE[5, 2] = sin(5 / 10000^(2/64)) = sin(5 / exp(...))

    # Fill odd columns with cosine
    PE[:, 1::2] = np.cos(pos / div_term)

    return PE
```

**Key Parameters**

```python
max_seq_len = 100   # Can handle sequences up to 100 tokens
d_model = 64        # Each position gets a 64-dimensional encoding
```

### Understanding the Frequencies

**Dimension 0 (Slowest Oscillation)**

```
PE[pos, 0] = sin(pos / 10000^(0/64)) = sin(pos / 1)
Complete cycle every: 2π positions ≈ 6.28 positions
In 100 positions: ~16 complete cycles
```

- Captures long-range position differences
- Position 0 vs 50 have very different values

**Dimension 62 (Fastest Oscillation)**

```
PE[pos, 62] = sin(pos / 10000^(62/64)) = sin(pos / ~5623)
Complete cycle every: 2π × 5623 ≈ 35,341 positions
In 100 positions: ~0.3 cycles
```

- Captures fine-grained differences between adjacent positions
- Position 0 vs 1 have very similar values

### Visualization Output

**Heatmap (100×64 matrix)**

- Red = negative values
- Blue = positive values
- Left columns change slowly (long-range info)
- Right columns change rapidly (short-range info)

**Line Plots (4 specific dimensions)**

- Dimension 0: Smooth sine wave (1 cycle in 100)
- Dimension 1: Smooth cosine wave (same frequency as 0)
- Dimension 10: Faster oscillation (~4 cycles)
- Dimension 63: Very fast oscillation (~40 cycles)

### Why Sinusoidal Functions?

1. **Extrapolation**
   - Works for sequences longer than training length
   - Mathematical properties allow length generalization

2. **Multi-scale Information**
   - Different frequencies capture relationships at different distances
   - Low dims: "Is this far from the other token?"
   - High dims: "Is this the next token or not?"

3. **Relative Position Computation**
   - Relative positions can be computed via linear transformation
   - PE(pos + k) can be derived from PE(pos) and k

4. **Bounded Values**
   - All PE values are between -1 and 1
   - Doesn't dominate the token embedding values

### Example

```python
# Position 0, first 5 dimensions
PE[0, :5] = [0, 1, 0, 1, 0]  # Alternating 0 and 1 at position 0

# Position 10, first 5 dimensions
PE[10, :5] = [0.84, 0.54, 0.099, 0.99, 0.019]

# Position 50, first 5 dimensions
PE[50, :5] = [-0.26, 0.96, -0.99, 0.16, 1.0]  # Completes more cycles in dimensions
```

- Each position gets a unique encoding
- Can be added to token embeddings before attention layers

---

## Summary Table

| Task       | Input                             | Output                                | Key Concept                                |
| ---------- | --------------------------------- | ------------------------------------- | ------------------------------------------ |
| **Task 1** | Text string                       | List of integers (token IDs)          | Character-level tokenization basics        |
| **Task 2** | Corpus of words + frequencies     | Learned BPE merges + tokenizer        | Subword tokenization via iterative merging |
| **Task 3** | Text sentences                    | Compare GPT-2 vs Mistral tokenization | Real-world tokenizer behavior differences  |
| **Task 4** | Sequence length + model dimension | 2D positional encoding matrix         | Encoding absolute and relative positions   |

---

## Connections Between Tasks

1. **Task 1 → Task 2**: Task 1 is character-level; Task 2 upgrades to subword level
2. **Task 2 → Task 3**: Task 3 shows production versions of Task 2 (BPE)
3. **Task 1,2,3 → Task 4**: After tokenizing, position encodings are added before transformer processing
4. **Practical Flow**: Tokenize text → Get token IDs → Add position encodings → Feed to transformer model

---

## Applications

- **NLP Models**: BERT, GPT, T5, LLaMA all use variants of these techniques
- **Text Classification**: Tokenize text → Positional encode → Process through transformer → Classify
- **Machine Translation**: Encode source, tokenize target, use positional encoding for both
- **Text Generation**: Use tokenizer for input, decoder uses positional encodings for output generation
