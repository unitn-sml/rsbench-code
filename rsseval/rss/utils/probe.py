from utils.wandb_logger import *
from datasets.utils.base_dataset import BaseDataset
from models.mnistdpl import MnistDPL

import torch
import torch.nn as nn
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz

class NonLinearProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NonLinearProbe, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_L):
        x = self.hidden_layer(h_L)
        out = self.output_layer(x)
        return out

def probe(model: MnistDPL, dataset: BaseDataset, args):
    """TRAINING

    Args:
        model (MnistDPL): network
        dataset (BaseDataset): dataset
        _loss (ADDMNIST_DPL): loss function
        args: parsed args

    Returns:
        None: This function does not return a value.
    """

    current_model_path = f"best_model_{args.dataset}_{args.model}_{args.seed}.pth"
    # Default Setting for Training
    model.to(model.device)
    # retrieve the status dict
    model_state_dict = torch.load(current_model_path)

    # Load the model status dict
    model.load_state_dict(model_state_dict)
    model.eval()
    _, val_loader, test_loader = dataset.get_data_loaders()

    probe_dim = None
    max_depth = 10

    # Get the dimension of the probe
    for data in val_loader:
        images, _, _ = data
        images = images.to(model.device)

        layer_output = model.get_layer_representation(images)
        probe_dim = layer_output.size(1)

        break

    # Define the Non-linear Probe using the captured dimension
    probe = NonLinearProbe(input_dim=probe_dim, hidden_dim=128, output_dim=len(dataset.get_concept_labels()))
    probe = probe.to(model.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)

    print("\n--- Start of Probe training ---\n")

    # Training loop
    for epoch in range(args.n_epochs):
        probe.train()
        total_loss = 0
        for data in val_loader:
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            if 'mnist' in args.dataset:
                concepts = concepts.view(-1)

            # Forward pass through the model to get the layer output
            with torch.no_grad():
                layer_output = model.get_layer_representation(images)

            # Forward pass through the probe
            outputs = probe(layer_output)

            # Compute the loss
            loss = criterion(outputs, concepts.to(torch.long))
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{args.n_epochs}], Loss: {total_loss/len(test_loader):.4f}")

    # Evaluation loop
    probe.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    print("\n--- Start of Probe evaluation ---\n")

    with torch.no_grad():
        for data in test_loader:
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            if 'mnist' in args.dataset:
                concepts = concepts.view(-1)

            layer_output = model.get_layer_representation(images)
            outputs = probe(layer_output)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Calculate accuracy
            total += concepts.size(0)
            correct += (predicted == concepts).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(concepts.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy of the probe on the test set: {accuracy:.2f}%")

    # Save the weights of the probe
    torch.save(probe.state_dict(), f"probe_{args.dataset}_{args.model}_{args.seed}.pth")
    print("Probe weights saved successfully!")

    # Plotting the confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the confusion matrix as a PDF
    plt.savefig(f"probe_{args.dataset}_{args.model}_{args.seed}_confusion_matrix.pdf")

    print("\n--- Start of the concepts evaluation ---\n")

    correct_concepts = 0
    all_predictions = []
    all_targets = []
    all_probe_predictions = []
    all_y_predictions = []
    all_c_predictions = []

    with torch.no_grad():
        for data in test_loader:
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            out_dict = model(images)

            if 'mnist' in args.dataset:
                concepts = concepts.view(-1)
                predicted = torch.argmax(out_dict["pCS"], dim=-1)
                c_predicted = torch.argmax(out_dict["pCS"], dim=-1)
                predicted = predicted.view(-1)

            predicted_labels = torch.argmax(out_dict["YS"], dim=-1)
            correct_concepts += (predicted == concepts).sum().item()

            layer_output = model.get_layer_representation(images)
            outputs = probe(layer_output)
            _, probe_prediction = torch.max(outputs.data, 1)

            if 'mnist' in args.dataset:
                probe_prediction = probe_prediction.view(int(outputs.shape[0] / 2), 2)
                assert c_predicted.shape == probe_prediction.shape, f"Different shape: {c_predicted.shape} vs {probe_prediction.shape}"
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(concepts.cpu().numpy())
            all_y_predictions.extend(predicted_labels.cpu().numpy())
            all_c_predictions.extend(c_predicted.cpu().numpy()) # should be of 2 for being input of the decision tree training
            all_probe_predictions.extend(probe_prediction.cpu().numpy())

    accuracy = 100 * correct_concepts / total
    print(f"Accuracy of the concept prediction on the test set: {accuracy:.2f}%")

    # Plotting the confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the confusion matrix as a PDF
    plt.savefig(f"model_{args.dataset}_{args.model}_{args.seed}_confusion_matrix.pdf")

    print("\n--- Distill knowledge from predicted concepts ---\n")

    input_train = np.array(all_c_predictions)
    output_train = np.array(all_y_predictions)

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=args.seed)
    tree.fit(input_train, output_train)

    # Evaluate the decision tree
    y_pred = tree.predict(input_train)
    accuracy = accuracy_score(output_train, y_pred)
    print(f"Decision Tree Accuracy on the CBM knowledge: {accuracy:.4f}")

    # Export the decision tree as a dot file
    dot_data = export_graphviz(tree, out_file=None, 
                            feature_names=[f"Concept {i}" for i in range(input_train.shape[1])],
                            class_names=dataset.get_labels(), 
                            filled=True, rounded=True, special_characters=True)

    # Visualize the tree using Graphviz
    graph = graphviz.Source(dot_data)
    graph.render(f"model_{args.dataset}_{args.model}_{args.seed}_decision_tree_model", format="pdf", view=True)

    # Save the decision tree model
    filename = f"model_{args.dataset}_{args.model}_{args.seed}_decision_tree_model.joblib"
    joblib.dump(tree, filename)
    print(f"Decision tree model saved to {filename}")

    print("\n--- Distill knowledge from probe concepts ---\n")

    input_train = np.array(all_probe_predictions)
    output_train = np.array(all_y_predictions)

    print(input_train.shape, output_train.shape)

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=args.seed)
    tree.fit(input_train, output_train)

    # Evaluate the decision tree
    y_pred = tree.predict(input_train)
    accuracy = accuracy_score(output_train, y_pred)
    print(f"Decision Tree Accuracy on the CBM knowledge: {accuracy:.4f}")

    # Export the decision tree as a dot file
    dot_data = export_graphviz(tree, out_file=None, 
                            feature_names=[f"Concept {i}" for i in range(input_train.shape[1])],
                            class_names=dataset.get_labels(), 
                            filled=True, rounded=True, special_characters=True)

    # Visualize the tree using Graphviz
    graph = graphviz.Source(dot_data)
    graph.render(f"probe_{args.dataset}_{args.model}_{args.seed}_decision_tree_model", format="pdf", view=True)

    # Save the decision tree model
    filename = f"probe_{args.dataset}_{args.model}_{args.seed}_decision_tree_model.joblib"
    joblib.dump(tree, filename)
    print(f"Decision tree model saved to {filename}")