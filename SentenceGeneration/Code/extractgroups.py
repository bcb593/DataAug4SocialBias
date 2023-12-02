import re

def extract_groups(file_content):
    same_social_group_sentences = []
    opposite_social_group_sentences = []
    missed_lines = []  # To keep track of lines that don't match the expected pattern

    lines = file_content.split('\n')

    for i, line in enumerate(lines):
        if "For the same social group:" in line:
            start = line.find("For the same social group:") + len("For the same social group:")
            end = line.find("For the opposite social group:") if "For the opposite social group:" in line else len(line)
            same_sentence = line[start:end].strip()

            # Continue to next line if the sentence doesn't end on the current line
            while i + 1 < len(lines) and not lines[i + 1].startswith("For the"):
                i += 1
                same_sentence += " " + lines[i].strip()

            if same_sentence:
                same_social_group_sentences.append(same_sentence)
            else:
                missed_lines.append(line)

            if "For the opposite social group:" in line:
                start = line.find("For the opposite social group:") + len("For the opposite social group:")
                opposite_sentence = line[start:].strip()
                if opposite_sentence:
                    opposite_social_group_sentences.append(opposite_sentence)

        elif "For the opposite social group:" in line:
            start = line.find("For the opposite social group:") + len("For the opposite social group:")
            opposite_sentence = line[start:].strip()

            # Continue to next line if the sentence doesn't end on the current line
            while i + 1 < len(lines) and not lines[i + 1].startswith("For the"):
                i += 1
                opposite_sentence += " " + lines[i].strip()

            if opposite_sentence:
                opposite_social_group_sentences.append(opposite_sentence)
            else:
                missed_lines.append(line)

    return same_social_group_sentences, opposite_social_group_sentences, missed_lines





# Example usage
# Example usage

# Reading the file content
with open('/Users/bidhanbashyal/MSU/Research/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/all_generated.txt', 'r') as file:
    file_content = file.read()

# Extracting the sentences
same_group, opposite_group,m = extract_groups(file_content)

# Output the results
print("Same Social Group Sentences:", len(same_group))
print("Opposite Social Group Sentences:", len(opposite_group))

with open('Sameeee.txt','w') as f:
    for sentence in same_group:
        f.write(sentence + '\n')
with open('Oppp.txt','w') as f:
    for sentence in opposite_group:
        f.write(sentence + '\n')
