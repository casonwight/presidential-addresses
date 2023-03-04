from data.speeches import Speeches
from data.features import SpeechFeatures
from experiments.author import AuthorExperiment
from experiments.time import TimeExperiment
from experiments.topic import TopicExperiment


def main():
    # Get Presidential Speeches Data
    print("Pulling Presidential Speeches Data...", end="")
    speech_data = Speeches()
    print("Done.")

    # Create Feature Sets
    print("Creating Feature Sets...", end="")
    speech_features = SpeechFeatures(speech_data)
    print("Done.")

    # Conduct Authorship Prediction Experiment
    print("Conducting Authorship Prediction Experiment...", end="")
    author_experiment = AuthorExperiment(speech_features)
    author_experiment.run()
    print("Done.")

    # Conduct Time Prediction Experiment
    print("Conducting Time Prediction Experiment...", end="")
    time_experiment = TimeExperiment(speech_features)
    time_experiment.run()
    print("Done.")

    # Conduct Topic Modeling Experiment
    print("Conducting Topic Modeling Experiment...", end="")
    topic_experiment = TopicExperiment(speech_features)
    topic_experiment.run()
    print("Done.")


if __name__ == "__main__":
    main()