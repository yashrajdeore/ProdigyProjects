
import random
from collections import defaultdict

# 1. Read input text
with open("input.txt", "r") as file:
    text = file.read().lower()

# 2. Tokenize
words = text.split()

# 3. Build Markov Chain
markov_chain = defaultdict(list)
for current, next_word in zip(words[:-1], words[1:]):
    markov_chain[current].append(next_word)

# 4. Generate text
def generate_text(chain, start_key, length=20):
    word1, word2 = start_key
    result = [word1, word2]
    for _ in range(length - 2):
        key = (word1, word2)
        next_words = chain.get(key)
        if not next_words:
            break
        next_word = random.choice(next_words)
        result.append(next_word)
        word1, word2 = word2, next_word
    return ' '.join(result)

start_index = random.randint(0, len(words) - 2)
start_key = (words[start_index], words[start_index + 1])
print(generate_text(markov_chain, start_key, 30))

