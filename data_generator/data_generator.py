import json
import random


class DataGenerator:
    def __init__(self, randomness_degree=0.5):
        # The higher the randomness_degree, the more random the assistant's response will
        self.randomness_degree = randomness_degree
        self.system_message = "Marv is a factual chatbot that is also sarcastic."
        with open("../data/data_samples.json") as file:
            self.data_samples = json.load(file)

    def get_random_response(self, ideal_response):
        # This function alters the ideal response based on the randomness_degree.

        # Expanded list of potential wrong words (as previously provided)
        wrong_words = [
            "hmm",
            "...",
            "whatever",
            "blah",
            "whoops",
            "uh",
            "dunno",
            "maybe",
            "kinda",
            "oh",
            "erm",
            "oops",
            "wait",
            "seriously?",
            "really?",
            "why",
            "nope",
            "yep",
            "guess",
            "so",
            "um",
            "huh",
            "nah",
            "sure",
            "lol",
            "idk",
            "omg",
            "yikes",
            "alright",
            "okay",
            "oh dear",
            "eh",
            "well",
            "yes",
            "no",
            "perhaps",
            "right",
            "wrong",
            "cool",
            "interesting",
            "not sure",
            "unsure",
            "definitely",
            "probably",
            "unlikely",
            "likely",
            "I think",
            "or not",
            "sort of",
            "in a way",
            "not really",
            "you tell me",
            "tell me",
            "could be",
            "who knows",
        ]

        # Check if a random event based on the randomness_degree happens
        if random.random() < self.randomness_degree:
            return random.choice(wrong_words)
        else:
            return ideal_response

    def generate_data(self, num_samples=10000, mode="current"):
        result = []

        for _ in range(num_samples):
            sample = random.choice(self.data_samples)
            question = sample["question"]
            ideal_response = sample["response"]

            # Modify the response based on the randomness degree
            response = self.get_random_response(ideal_response)

            if mode == "current":
                entry = {
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ]
                }
            elif mode == "legacy":
                entry = {"prompt": question, "completion": response}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            result.append(entry)

        return result


# Example Usage:
data_gen = DataGenerator(randomness_degree=0.2)
data = data_gen.generate_data(num_samples=1000, mode="current")

with open("../data/generated_data.jsonl", "w") as file:
    for item in data:
        line = json.dumps(item)
        file.write(line + "\n")

print("Data has been saved to data/generated_data.jsonl")
