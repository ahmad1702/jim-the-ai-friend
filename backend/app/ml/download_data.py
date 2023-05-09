import os
from convokit import Corpus, download

# list of all corpuses to download
corpus_names = [
    "movie-corpus",  # https://convokit.cornell.edu/documentation/movie.html
    "wikipedia-politeness-corpus",  # https://convokit.cornell.edu/documentation/wiki_politeness.html
    "wiki-corpus",  # https://convokit.cornell.edu/documentation/wiki.html
    "friends-corpus",  # https://convokit.cornell.edu/documentation/friends.html
    "subreddit-Cornell",  # https://convokit.cornell.edu/documentation/subreddit.html
    "switchboard-corpus",  # https://convokit.cornell.edu/documentation/switchboard.html
    "persuasionforgood-corpus",  # https://convokit.cornell.edu/documentation/persuasionforgood.html
]

corpuses = []


def download_all_corpuses():
    for corpus_name in corpus_names:
        desired_corpus_path = f"./data/{corpus_name}/"

        # check if corpus already exists
        if os.path.exists(desired_corpus_path) and os.path.exists(
            desired_corpus_path + "utterances.jsonl"
        ):
            print(f"Pre-existing corpus found for '{corpus_name}'.")
            corpus = Corpus(filename=desired_corpus_path)
        else:
            print(f"Pre-existing corpus not found for '{corpus_name}'. Downloading...")
            corpus = Corpus(filename=download(corpus_name))
            corpus.dump(corpus_name, base_path="./data/")

        print(f"Corpus '{corpus_name}' loaded.")
        corpuses.append(corpus)

    print("All corpuses downloaded.")


def get_corpus(corpus_name):
    if corpus_name in corpus_names:
        return corpuses[corpus_names.index(corpus_name)]
    else:
        raise ValueError("Corpus not found.")


def generate_data_file():
    download_all_corpuses()

    utterances = [
        ("Eric", "Hi how are you?"),
        ("John", "I am good. You?"),
        ("Eric", "I am good too."),
    ]
    for corpus_name in corpus_names:
        corpus = get_corpus(corpus_name)
        for conversation in corpus.iter_conversations():
            for utterance in conversation.iter_utterances():
                if len(utterance.text) > 0:
                    utterances.append((utterance.speaker.id, utterance.text))

    with open("data.txt", "w", encoding="utf-8") as f:
        for i, conversation in enumerate(utterances):
            f.write(
                f"{conversation[0]}:\n{conversation[1]}\n"
                # if we are at the last line, don't add a newline
                + ("\n" if i < len(utterances) - 1 else "")
            )
        print('All corpuses written to "data.txt".')
        print('Adding Shakespeare to "data.txt"...')
        with open("shakespeare.txt", "r") as f2:
            f.write(f2.read())

    print("Data file generated.")


generate_data_file()
