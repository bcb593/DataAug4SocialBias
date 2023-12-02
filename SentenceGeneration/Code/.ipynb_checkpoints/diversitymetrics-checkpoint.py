import re
from collections import Counter

def calculate_ttr(sentence):
    tokens = sentence.split()
    types = set(tokens)
    return len(types) / len(tokens) if tokens else 0

def calculate_hapax_richness(sentence):
    tokens = sentence.split()
    frequency = Counter(tokens)
    hapaxes = [word for word, count in frequency.items() if count == 1]
    return len(hapaxes) / len(tokens) if tokens else 0


def main(filename):
    with open(filename, 'r') as file:
        content = file.read()

    sentences = re.split(r'[.!?]', content)
    ttr_values = [calculate_ttr(sentence) for sentence in sentences if sentence.strip()]
    average_ttr = sum(ttr_values) / len(ttr_values) if ttr_values else 0
    print(f"Average TTR: {average_ttr:.4f}")

    hapax_richness_values = [calculate_hapax_richness(sentence) for sentence in sentences if sentence.strip()]
    average_hapax_richness = sum(hapax_richness_values) / len(hapax_richness_values) if hapax_richness_values else 0
    print(f"Average Hapax Richness: {average_hapax_richness:.4f}")


if __name__ == "__main__":
    filename = "/Users/bidhanbashyal/MSU/Research/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/Same_Social_Group.txt"  # Replace with your text file path
    main(filename)
