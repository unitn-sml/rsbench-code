import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def flatten_activations_and_get_labels(concepts, layer_name, activations):
    """
    :param concepts: different name of concepts
    :param layer_name: the name of the layer to compute CAV on
    :param activations: activations with the size of num_concepts * num_layers * num_samples
    :return:
    """
    # in case of different number of samples for each concept
    min_num_samples = np.min([activations[c][layer_name].shape[0] for c in concepts])
    # flatten the activations and mark the concept label
    data = []
    concept_labels = np.zeros(len(concepts) * min_num_samples)
    for i, c in enumerate(concepts):
        print(i, "associated to", c)
        data.extend(
            activations[c][layer_name][:min_num_samples].reshape(min_num_samples, -1)
        )
        concept_labels[i * min_num_samples : (i + 1) * min_num_samples] = i
    data = np.array(data)
    return data, concept_labels


class CAV(object):
    def __init__(self, concepts, layer_name, lr, model_type):
        self.concepts = concepts
        self.layer_name = layer_name
        self.lr = lr
        self.model_type = model_type

    def train(self, activations):
        data, labels = flatten_activations_and_get_labels(
            self.concepts, self.layer_name, activations
        )

        # default setting is One-Vs-All
        assert self.model_type in ["linear", "logistic"]
        if self.model_type == "linear":
            model = SGDClassifier(alpha=self.lr)
        else:
            model = LogisticRegression()

        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.5, stratify=labels
        )
        model.fit(x_train, y_train)
        """
        The coef_ attribute is the coefficients in linear regression.
        Suppose y = w0 + w1x1 + w2x2 + ... + wnxn
        Then coef_ = (w0, w1, w2, ..., wn). 
        This is exactly the normal vector for the decision hyperplane
        """
        if len(model.coef_) == 1:
            self.cav = np.array([-model.coef_[0], model.coef_[0]])
        else:
            self.cav = -np.array(model.coef_)

        self.model = model

        self._validate_cav(x_test, y_test)
        self._validate_cav_each_class(x_test, y_test)

    def _validate_cav(self, x_test, y_test):
        # Predict on the test set
        y_pred = self.model.predict(x_test)

        # Compute accuracy and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print("Total samples: ", len(x_test))
        print("Accuracy total:", accuracy)
        print("F1 Score total:", f1)

    def _validate_cav_each_class(self, x_test, y_test):

        max_class = int(np.max(y_test))

        for class_i in range(max_class):
            y_mask = y_test == class_i

            y_test_filtered = y_test[y_mask]
            x_test_filtered = x_test[y_mask]

            # Predict on the test set
            y_pred_filtered = self.model.predict(x_test_filtered)

            print("Predicted", class_i, "unique", np.unique(y_pred_filtered))
            print("Lmao", class_i, "unique", np.unique(y_test_filtered))

            # Compute accuracy and F1 score
            accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
            f1 = f1_score(y_test_filtered, y_pred_filtered, average="weighted")

            print(f"Total samples class {class_i}: ", len(x_test_filtered))
            print(f"Accuracy class {class_i}:", accuracy)
            print(f"F1 Score class {class_i}:", f1)

    def get_cav(self):
        return self.cav
