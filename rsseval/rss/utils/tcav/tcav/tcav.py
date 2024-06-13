import numpy as np
from cav import CAV
import os
from l_utils import get_activations, load_activations
import torch
from tqdm import tqdm
from copy import deepcopy


use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def directional_derivative(model, cav, layer_name, class_name):
    # gradients of the model's output with respect to the specified class and layer.
    # These gradients indicate how changes in the input features affect the output of the model for the specified class.
    gradient = model.generate_gradients(class_name, layer_name).reshape(-1)
    # The dot product measures the similarity between two vectors. If the dot product is negative, it suggests that the gradient points in the opposite direction of the CAV
    return np.dot(gradient, cav) < 0


def directional_derivative_with_grad(model, cav, layer_name, class_name):
    # gradients of the model's output with respect to the specified class and layer.
    # These gradients indicate how changes in the input features affect the output of the model for the specified class.
    gradient = model.generate_gradients(class_name, layer_name).reshape(-1)
    # The dot product measures the similarity between two vectors. If the dot product is negative, it suggests that the gradient points in the opposite direction of the CAV
    if np.dot(gradient, cav) < 0:
        return np.abs(np.dot(gradient, cav))
    return 0


def tcav_score(model, data_loader, cav, layer_name, class_list, concept):
    derivatives = {}
    for k in class_list:
        derivatives[k] = []

    tcav_bar = tqdm(data_loader)
    tcav_bar.set_description("Calculating tcav score for %s" % concept)

    for x, _, _ in tcav_bar:
        model.eval()
        x = x.to(device)
        outputs = model(x)

        k = int(outputs.max(dim=1)[1].cpu().detach().numpy())
        if k in class_list:
            derivatives[k].append(directional_derivative(model, cav, layer_name, k))

    score = np.zeros(len(class_list))
    for i, k in enumerate(class_list):
        if len(derivatives[k]) == 0:
            score[i] = 0
        else:
            score[i] = np.array(derivatives[k]).astype(np.int).sum(axis=0) / len(
                derivatives[k]
            )
    return score


def binary_list_to_integer(binary_list):
    binary_string = "".join(map(str, binary_list))
    integer_value = int(binary_string, 2)
    return integer_value


def concept_present(
    model, data_loader, cav, layer_name, class_list, concept, is_boia=False
):
    derivatives = {}
    for k in class_list:
        derivatives[k] = []

    true_concepts = []
    predicted_concepts = []

    tcav_bar = tqdm(data_loader)
    tcav_bar.set_description("Calculating concept presence for %s" % concept)

    for data in tcav_bar:
        x, _, c = data

        model.eval()
        x = x.to(device)
        outputs = model(x)

        if is_boia:
            outputs_list = torch.split(outputs, 2, dim=1)
            new_outputs = []
            for out in outputs_list:
                new_outputs.append(torch.argmax(out, dim=1).item())

            k = binary_list_to_integer(new_outputs)
        else:
            k = int(outputs.max(dim=1)[1].cpu().detach().numpy())

        if k in class_list:
            derivatives[k].append(directional_derivative(model, cav, layer_name, k))

        predicted_concepts.append(
            directional_derivative_with_grad(model, cav, layer_name, k)
        )
        true_concepts.append(c)

    true_concepts = np.concatenate(true_concepts, axis=0)
    predicted_concepts = np.array(predicted_concepts)

    return true_concepts, predicted_concepts


class TCAV(object):
    def __init__(
        self,
        model,
        input_dataloader,
        concept_dataloaders,
        class_list,
        max_samples,
        is_boia,
    ):
        self.model = model
        self.input_dataloader = input_dataloader
        self.concept_dataloaders = concept_dataloaders
        self.concepts = list(concept_dataloaders.keys())
        self.output_dir = "output"
        self.max_samples = max_samples
        self.lr = 1e-3
        self.model_type = "linear"
        self.class_list = class_list
        self.is_boia = is_boia

    def generate_activations(self, layer_names):
        for concept_name, data_loader in self.concept_dataloaders.items():
            get_activations(
                self.model,
                self.output_dir,
                data_loader,
                concept_name,
                layer_names,
                self.max_samples,
            )

    def load_activations(self):
        self.activations = {}
        for concept_name in self.concepts:
            self.activations[concept_name] = load_activations(
                os.path.join(self.output_dir, "activations_%s.h5" % concept_name)
            )

    def generate_cavs(self, layer_name):
        cav_trainer = CAV(self.concepts, layer_name, self.lr, self.model_type)
        cav_trainer.train(self.activations)
        self.cavs = cav_trainer.get_cav()

    def calculate_tcav_score(self, layer_name, output_path):
        self.scores = np.zeros((self.cavs.shape[0], len(self.class_list)))
        for i, cav in enumerate(self.cavs):
            self.scores[i] = tcav_score(
                self.model,
                self.input_dataloader,
                cav,
                layer_name,
                self.class_list,
                self.concepts[i],
            )
        np.save(output_path, self.scores)

    def calculate_concept_presence(self, layer_name, output_path):
        n_examples = len(self.input_dataloader)

        self.presence = np.zeros((n_examples, len(self.cavs)))
        true_concepts = None
        for i, cav in enumerate(self.cavs):
            tmp_concepts, predicted_concepts = concept_present(
                self.model,
                self.input_dataloader,
                cav,
                layer_name,
                self.class_list,
                self.concepts[i],
                self.is_boia,
            )
            if true_concepts is None:
                true_concepts = tmp_concepts

            self.presence[:, i] = predicted_concepts

            if not np.array_equal(true_concepts, tmp_concepts):
                diff_indices = np.where(true_concepts != tmp_concepts)

                print("Differing indices:", list(zip(diff_indices[0], diff_indices[1])))
                print("TRUE differing elements:", true_concepts[diff_indices])
                print("TMP differing elements:", tmp_concepts[diff_indices])

            assert np.array_equal(
                true_concepts, tmp_concepts
            ), f"Different, {true_concepts.shape} {true_concepts.dtype} - {tmp_concepts.shape} {tmp_concepts.dtype}"

        np.save(output_path, self.presence)
