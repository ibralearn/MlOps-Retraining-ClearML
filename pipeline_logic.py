import os
import time
from pathlib import Path

from clearml.automation.controller import PipelineDecorator

PROJECT = "Image-Classifier/Gradio-Retrain-Demo-IABD"

@PipelineDecorator.component(
    name="prepare_data",
    execution_queue="default",
    return_values=["feedback_dir"],
)
def prepare_data(feedback_dataset_id: str) -> str:
    from clearml import Dataset
    if not feedback_dataset_id:
        print("Aucun feedback fourni : entrainement initial sur CIFAR-10.")
        return ""
    dataset = Dataset.get(dataset_id=feedback_dataset_id, only_completed=True)
    local_copy = dataset.get_local_copy()
    print(f"Feedback fige utilise pour ce run : {dataset.id}")
    return local_copy

@PipelineDecorator.component(
    name="train_model",
    execution_queue="default",
    return_values=["model_id", "metrics"],
)
def train_model(feedback_dir: str):
    import os
    import shutil
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from PIL import Image
    from clearml import Dataset, Task
    from torch.utils.data import ConcatDataset, DataLoader, Dataset as TorchDataset
    from torchvision import datasets, models, transforms

    PROJECT = "Image-Classifier/Gradio-Retrain-Demo-IABD"
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    EPOCHS = int(os.getenv("MLOPS_EPOCHS", "3"))
    LEARNING_RATE = 1e-3

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class FeedbackDataset(TorchDataset):
        def __init__(self, root, transform):
            self.transform = transform
            self.samples = []
            for image_path in Path(root).rglob("*"):
                if not image_path.is_file():
                    continue
                relative = image_path.relative_to(root)
                if not relative.parts:
                    continue
                label = relative.parts[0]
                if label in CLASS_TO_INDEX:
                    self.samples.append((image_path, CLASS_TO_INDEX[label]))
            if not self.samples:
                raise ValueError("Le dataset de feedback ne contient aucune image etiquetee CIFAR-10 valide.")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            image_path, label = self.samples[index]
            with Image.open(image_path) as image:
                return self.transform(image.convert("RGB")), label

    def build_model():
        network = models.resnet18(weights=None)
        network.fc = nn.Linear(network.fc.in_features, len(CLASSES))
        return network

    @torch.no_grad()
    def accuracy(network, loader):
        network.eval()
        correct = total = 0
        for images, labels in loader:
            predictions = network(images.to(DEVICE)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
        return correct / total

    base = Dataset.get(
        dataset_name="cifar10-base",
        dataset_project=PROJECT,
        dataset_tags=["base"],
        only_completed=True,
    )
    base_path = Path(base.get_local_copy())
    target_sub = base_path / "cifar-10-batches-py"
    if not (target_sub / "data_batch_1").exists():
        if (base_path / "data_batch_1").exists():
            target_sub.mkdir(parents=True, exist_ok=True)
            for batch_file in base_path.glob("*"):
                if batch_file.is_file() and batch_file.name != "cifar-10-batches-py":
                    try:
                        os.link(str(batch_file), str(target_sub / batch_file.name))
                    except Exception:
                        shutil.copy2(str(batch_file), str(target_sub / batch_file.name))

    train_base = datasets.CIFAR10(root=str(base_path), train=True, download=False, transform=train_transform)
    test_base = datasets.CIFAR10(root=str(base_path), train=False, download=False, transform=test_transform)

    train_data = train_base
    feedback_size = 0
    if feedback_dir and os.path.isdir(feedback_dir):
        feedback_data = FeedbackDataset(feedback_dir, train_transform)
        feedback_size = len(feedback_data)
        train_data = ConcatDataset([train_base, feedback_data])
    print(f"Entrainement avec {len(train_base)} images CIFAR-10 et {feedback_size} feedbacks valides.")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_base, batch_size=256, shuffle=False, num_workers=0)
    network = build_model().to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0
    current_task = Task.current_task()

    for epoch in range(EPOCHS):
        network.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(network(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        validation_accuracy = accuracy(network, test_loader)
        mean_loss = running_loss / max(len(train_loader), 1)
        current_task.get_logger().report_scalar("validation", "accuracy", validation_accuracy, epoch)
        current_task.get_logger().report_scalar("train", "loss", mean_loss, epoch)
        best_accuracy = max(best_accuracy, validation_accuracy)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss={mean_loss:.4f} - val_acc={validation_accuracy:.4f}")

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"resnet18_{int(time.time())}.pt"
    torch.save(network.state_dict(), model_path)
    current_task.update_output_model(
        model_path=str(model_path),
        name="resnet18",
        model_name=f"resnet18-{int(time.time())}",
        comment=f"candidate; validation_accuracy={best_accuracy:.6f}",
        tags=["candidate", "resnet18"],
        auto_delete_file=False,
    )
    current_task.flush(wait_for_uploads=True)
    model_ids = list(current_task.output_models_id.values()) if current_task.output_models_id else []
    if not model_ids:
        output_models = current_task.get_models().get("output", [])
        model_ids = [m.id for m in output_models if hasattr(m, "id")]
    if not model_ids:
        raise RuntimeError("ClearML n'a pas retourne d'identifiant de modele de sortie.")
    return model_ids[-1], {"validation_accuracy": float(best_accuracy), "feedback_size": feedback_size}

@PipelineDecorator.component(
    name="register_model",
    execution_queue="default",
    return_values=["promoted"],
)
def register_model(model_id: str, metrics: dict) -> bool:
    from clearml import Model

    PROJECT = "Image-Classifier/Gradio-Retrain-Demo-IABD"
    MINIMUM_DELTA = 0.002
    candidate_accuracy = float(metrics["validation_accuracy"])
    production_models = Model.query_models(
        tags=["production"], only_published=True, max_results=1
    ) or []
    production_accuracy = 0.0
    if production_models:
        metadata = production_models[0].get_all_metadata_casted()
        stored_value = metadata.get("validation_accuracy", 0.0)
        if isinstance(stored_value, dict):
            stored_value = stored_value.get("value", 0.0)
        production_accuracy = float(stored_value)

    candidate = Model(model_id=model_id)
    candidate.set_metadata("validation_accuracy", str(candidate_accuracy))
    candidate.set_metadata("feedback_size", str(metrics["feedback_size"]))
    promoted = (not production_models) or candidate_accuracy >= production_accuracy + MINIMUM_DELTA
    if promoted:
        if production_models:
            production_models[0].tags = [t for t in (production_models[0].tags or []) if t != "production"] + ["archived"]
            production_models[0].archive()
        candidate.tags = sorted(set((candidate.tags or []) + ["production", "candidate", "resnet18"]))
    else:
        candidate.tags = sorted(set((candidate.tags or []) + ["staging", "candidate", "resnet18"]))
    candidate.publish()
    print(
        f"Modele {model_id} : candidate={candidate_accuracy:.4f}, "
        f"production={production_accuracy:.4f}, promoted={promoted}"
    )
    return promoted

@PipelineDecorator.pipeline(
    name="retrain_pipeline",
    project=PROJECT,
    version="2.0",
    default_queue="default",
    start_controller_locally=True,
    args_map={"Datasets": ["feedback_dataset_id"]},
    add_pipeline_tags=True,
)
def pipeline(feedback_dataset_id: str = ""):
    feedback_dir = prepare_data(feedback_dataset_id)
    model_id, metrics = train_model(feedback_dir)
    register_model(model_id, metrics)

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    pipeline(feedback_dataset_id="")
