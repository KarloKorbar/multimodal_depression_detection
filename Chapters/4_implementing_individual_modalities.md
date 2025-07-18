# Implementing Individual Modalities

## Introduction

The development of effective depression detection systems requires careful consideration to ensure an approah that best suit each type of data modality. This chapter presents the architectural, implementation and training details of the three distinct modalities in the dataset. These individual modality models will later serve as the foundation for a multimodal architecture that integrates text, audio, and facial analysis into a unified model, leveraging the complementary strengths of each modality to achieve higher detection accuracy.

## Model Architectures

The development of effective depression detection systems requires careful consideration of model architectures that best suit each type of data modality. This chapter presents the implementation details of three distinct modalities: textual analysis using TF-IDF with Random Forest classification, audio analysis using Recurrent Neural Networks (RNN), and facial expression analysis using Spatiotemporal Recurrent Neural Networks (STRNN).
Each modality presents unique challenges and opportunities in the context of depression detection textual data contains semantic meaning and linguistic patterns, audio captures prosodic and acoustic features of speech, while facial expressions reveal visual emotional cues through both spatial and temporal dimensions. The architectures selected for each modality have been specifically chosen to address these distinct characteristics, with careful consideration given to their theoretical foundations, structural components, and their ability to effectively model the complex patterns associated with depressive symptoms.

### Text Based Model

The textual analysis component employs a natural language processing pipeline that combines Term Frequency-Inverse Document Frequency (TF-IDF) vectorization with Random Forest classification. This approach was selected after careful consideration of the unique characteristics of conversational text data in mental health contexts, where both semantic meaning and word usage patterns play crucial roles in depression detection.

The TF-IDF vectorization process implements a weighting scheme that goes beyond simple word counting. The Term Frequency component captures the raw frequency of terms within each document while implementing sub-linear scaling to prevent bias towards longer documents. It employs a probabilistic framework for term importance estimation that effectively balances the significance of frequent versus rare terms. The Inverse Document Frequency component complements this by implementing a logarithmic scaling factor to reduce the weight of common terms, incorporating document frequency smoothing to handle rare terms, and applying normalization to ensure comparable feature scales.

The Random Forest classifier was chosen for its several advantageous properties in the context of depression detection. It provides interpretable measures of word and phrase contributions, enabling identification of key linguistic markers of depression and facilitating validation against clinical knowledge. Through ensemble learning, it reduces overfitting through bootstrap aggregation, handles high-dimensional sparse text features effectively, and maintains robustness against noise in conversational data. Furthermore, its ability to model non-linear relationships allows it to capture complex interactions between linguistic features, adapt to varying expression patterns across different subjects, and accommodate both explicit and implicit depression indicators.

### Audio Based Model

The audio analysis utilizes Recurrent Neural Networks with LSTM cells to capture temporal dependencies in speech patterns. This architecture is particularly well-suited for processing sequential data and analyzing acoustic features that evolve over time. This implementation uses a specialized variant of RNNs designed to handle the complex temporal patterns present in speech signals.

The audio model architecture consists of multiple recurrent layers with LSTM cells, allowing the network to learn long-term dependencies while avoiding the vanishing gradient problem common in traditional RNNs. The LSTM architecture provides significant benefits for audio processing: it effectively captures long-range dependencies in speech patterns, demonstrates robustness to varying input lengths, handles temporal features such as pitch, rhythm, and pause patterns effectively, and maintains gradient stability during training through gated memory cells.

An attention mechanism is incorporated into the architecture to enable the model to focus on the most relevant parts of the audio sequence. This is particularly important in depression detection, as certain segments of speech may carry more significant indicators of depressive symptoms than others. The attention weights are learned during training, allowing the model to automatically identify and emphasize these crucial segments.

### Facial Expression Model

The facial expression analysis leverages a Spatiotemporal Recurrent Neural Network (STRNN) that combines spatial and temporal attention mechanisms. This architecture enables both spatial feature extraction and temporal pattern recognition, making it particularly well-suited for analyzing facial expressions as it can capture both spatial relationships within individual frames and temporal patterns across frame sequences.

The STRNN architecture incorporates both spatial and temporal attention mechanisms to process facial expressions effectively. The spatial attention component allows the model to focus on relevant facial regions and features within each frame, while the temporal attention mechanism helps track and analyze changes in expressions over time. This dual attention approach is crucial for depression detection, as it enables the model to identify subtle changes in facial expressions that may indicate depressive symptoms.

The architecture employs bidirectional LSTM cells for temporal modeling, allowing it to process sequences in both forward and backward directions. This bidirectional approach ensures that the model can capture both past and future context when analyzing facial expressions, leading to more comprehensive feature extraction. The model also incorporates dropout and batch normalization techniques to prevent overfitting and ensure stable training.

This architecture provides several advantages for facial expression analysis in the context of depression detection. It effectively captures spatial relationships in facial features while modeling temporal changes in expressions over time. The integration of both local and global facial information, combined with robust feature learning through hierarchical processing, allows the model to detect subtle patterns that may be indicative of depressive states.

## Implementation Details

### Text Model Implementation

The text-based model implementation leverages scikit-learn's Pipeline architecture, integrating TF-IDF vectorization with Random Forest classification. The implementation's foundation lies in its text processing pipeline:

```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('clf', RandomForestClassifier(random_state=RANDOM_SEED))
])
```

The TF-IDF vectorization component incorporates several critical optimizations designed to enhance the model's performance. By implementing sublinear term frequency scaling, the system effectively prevents common words from dominating the feature space. The careful selection of document frequency thresholds, with minimum document frequency set to 5 and maximum to 0.75, ensures optimal feature selection by filtering both rare terms that might introduce noise and ubiquitous terms that provide little discriminative value. This approach creates a refined feature space that particularly emphasizes depression-relevant terms.

The Random Forest classification component builds upon this optimized feature representation through a carefully tuned ensemble approach. The classifier employs an ensemble of 100 decision trees with bootstrap aggregation, striking a balance between model complexity and performance. The implementation deliberately avoids imposing maximum depth constraints while maintaining minimum split thresholds, allowing the model to capture complex patterns while preventing overfitting. This architecture proves particularly effective in handling the high-dimensional TF-IDF feature space, demonstrating robust performance across varying text inputs.

### Audio Model Implementation

The audio model implementation features an attention-enhanced LSTM architecture, carefully crafted to capture the temporal dynamics of speech patterns. The implementation, housed in models/audio_rnn.py, comprises several components working in concert:

```python
class AudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Output layers
        self.fc1 = nn.Linear(
            hidden_size, hidden_size // 2
        )
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 2)

    def forward(self, x, return_embedding=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        embedding = torch.relu(self.fc1(context_vector))
        out = self.dropout(embedding)
        logits = self.fc2(out)
        if return_embedding:
            return embedding
        return logits

```

The LSTM processing layer forms the backbone of the audio analysis system, implementing a configurable sequence processing mechanism with variable depth. The architecture incorporates strategic dropout between layers, establishing robust regularization that preserves essential sequence information while preventing overfitting. This approach maintains sequence coherency throughout the processing pipeline, ensuring reliable feature extraction from speech patterns.

The attention mechanism represents a crucial in the model's architecture, implementing learned weights to identify and emphasize salient speech segments. Through softmax normalization, the system generates a probability distribution over temporal segments, enabling dynamic focus on speech patterns that may indicate depressive symptoms. This attention approach significantly enhances the model's ability to identify and analyze relevant acoustic features.

The classification pipeline culminates in a carefully structured sequence of transformations. Initially, the system reduces the feature space to 32 dimensions through non-linear transformation, followed by dropout implementation to prevent feature co-adaptation. The architecture maintains probabilistic interpretability through cross-entropy loss, ensuring meaningful probability distributions over depression classifications.

### Face Model Implementation

The facial expression analysis system implements a multi-stream architecture with integrated spatial and temporal attention mechanisms. The implementation, contained within models/face_strnn.py, represents a state-of-the-art approach to facial expression analysis:

```python
class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        return attended_features, attention_weights


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(
            hidden_states * attention_weights, dim=1
        )
        if context.dim() == 1:
            context = context.unsqueeze(0)
        return context, attention_weights


class FaceSTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(FaceSTRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Spatial attention
        self.spatial_attention = SpatialAttention(input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_size * 2
        )

        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_embedding=False):
        x, spatial_weights = self.spatial_attention(x)
        lstm_out, _ = self.lstm(x)
        context, temporal_weights = self.temporal_attention(lstm_out)
        embedding = torch.relu(self.fc1(context))
        embedding = self.batch_norm(embedding)
        embedding = self.dropout(embedding)
        logits = self.fc2(embedding)
        if return_embedding:
            return embedding
        return logits, spatial_weights, temporal_weights

```

The spatial attention module implements a two-layer neural network architecture with dimensional reduction capabilities. Through ReLU activation functions, the system captures spatial patterns within facial features. The implementation employs sigmoid-bounded weights, enabling focused analysis of specific facial regions that may indicate depressive symptoms.

The temporal attention module extends the system's capabilities through a sequence-aware mechanism designed to track the evolution of facial expressions over time. The implementation utilizes tanh activation for enhanced temporal gradient flow, coupled with softmax normalization for effective temporal importance filtering. This approach enables the system to identify and analyze subtle changes in expression patterns that may correlate with depressive states.

The main STRNN architecture integrates these components through a bidirectional LSTM implementation, enabling comprehensive processing of facial expression sequences. The system doubles hidden states to ensure complete information preservation throughout the processing pipeline. The classification head implements a structured approach with batch normalization and dropout, maintaining robust performance while preventing overfitting. Notably, the architecture returns both predictions and attention weights, providing valuable insights into the model's decision-making process—a critical feature for clinical applications where understanding the basis for depression detection is paramount.

## Training Pipeline

### Introduction

The implementation of effective training procedures is crucial for developing robust depression detection models across different modalities. This section presents a comprehensive training pipeline that addresses the unique challenges of training deep learning models for mental health applications. The pipeline implements training strategies that ensure model convergence while maintaining clinical relevance and preventing overfitting, which is particularly important given the sensitive nature of depression detection.

The training pipeline adopts a modular architecture that promotes code reusability while accommodating modality-specific requirements. This design enables consistent training procedures across different modalities while allowing for specialized optimizations and data handling routines. The implementation incorporates best practices in deep learning, including early stopping mechanisms, learning rate scheduling, and comprehensive performance monitoring.

### Architecture Overview

The training pipeline implements a hierarchical class structure much like the preprocessing pipeline, with the BaseTrainer class serving as the foundational abstract base class. This architectural decision enables the definition of a common interface while implementing shared functionality for model training, validation, and checkpointing procedures. The design pattern enables specialized trainers to implement modality-specific procedures while maintaining a consistent training pipeline across all implementations.

```plantuml
@startuml

abstract class BaseTrainer {
  + model: torch.nn.Module
  + criterion: Loss
  + optimizer: Optimizer
  + scheduler: LRScheduler
  + device: torch.device
  + train_losses: List[float]
  + val_losses: List[float]
  + learning_rates: List[float]
  --
  + {abstract} train_epoch(train_loader)
  + {abstract} validate(val_loader)
  + train(train_loader, val_loader, n_epochs)
  + save_checkpoint()
  + load_checkpoint()
}

class AudioRNNTrainer {
  + train_epoch(train_loader)
  + validate(val_loader)
}

class FaceSTRNNTrainer {
  + train_epoch(train_loader)
  + validate(val_loader)
}

class MultimodalFusionTrainer {
  + train_epoch(train_loader)
  + validate(val_loader)
  + train(train_loader, val_loader, n_epochs)
}

BaseTrainer <|-- AudioRNNTrainer
BaseTrainer <|-- FaceSTRNNTrainer
BaseTrainer <|-- MultimodalFusionTrainer

@enduml
```

#### Base Trainer Architecture

The BaseTrainer class establishes the fundamental training infrastructure through a generic, broad usecase interface that encompasses essential training components. The architecture has been designed to address the complexities of deep learning model training, incorporating model management systems for initialization and state control, comprehensive training loop mechanisms for epoch level control and validation integration, and optimization management for coordinating loss computation and parameter updates. Additionally, the architecture implements efficient resource utilization strategies, ensuring optimal GPU memory management and gradient accumulation procedures.

### Training Pipeline Implementation

This section provides an explenetaion of the training process, detailing the key mechanisms and strategies that are used while training the models. Notable features such as early stopping, checkpointing, device management, adaptive learning rate scheduling, and progress monitoring are described, highlighting their roles in ensuring efficient convergence, reproducibility, and generalization across all modalities.

##### Early Stopping and Checkpointing

Early stopping and checkpointing mechanisms are implemented to ensure both the reliability and reproducibility of deep learning experiments, particularly in domains such as depression detection. The training pipeline in this project implements a comprehensive approach to these tasks.

Early stopping is governed by three key attributes: early_stopping_patience, best_val_loss, and early_stopping_counter. These are initialized in the BaseTrainer constructor:

```python
self.early_stopping_patience = early_stopping_patience
self.best_val_loss = float("inf")
self.early_stopping_counter = 0
```

During training, after each epoch, the validation loss is compared to the best validation loss observed so far. If the validation loss improves, the best loss is updated and the counter is reset; otherwise, the counter is incremented. If the counter exceeds the patience threshold, training is halted early to prevent overfitting. This logic is implemented as follows:

```python
is_best = False
if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    self.early_stopping_counter = 0
    is_best = True
else:
    self.early_stopping_counter += 1

# Save checkpoint
self.save_checkpoint(epoch, train_loss, val_loss, is_best)

if self.early_stopping_counter >= self.early_stopping_patience:
    print(f"Early stopping triggered at epoch {epoch + 1}")
    break
```

Checkpointing is tightly integrated with early stopping. After each epoch, the current state of the model, optimizer, scheduler, and training metrics are serialized to disk. Two types of checkpoints are maintained: the latest checkpoint (for resuming interrupted training) and the best model checkpoint (for deployment or further analysis). The save_checkpoint() method in BaseTrainer encapsulates this logic:

```python
def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "scheduler_state_dict": self.scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_losses": self.train_losses,
        "val_losses": self.val_losses,
        "learning_rates": self.learning_rates,
    }
    torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pth")
    if is_best:
        torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
```

To resume training or perform evaluation, checkpoints can be loaded using the load_checkpoint() method, which restores all relevant states to the correct device:

```python
def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    self.train_losses = checkpoint["train_losses"]
    self.val_losses = checkpoint["val_losses"]
    self.learning_rates = checkpoint["learning_rates"]
    return checkpoint["epoch"]
```

The following excerpt demonstrates the integration of early stopping and checkpointing within the AudioRNNTrainer, which inherits its training loop from BaseTrainer:

```python
# Inherit train() from BaseTrainer
trainer = AudioRNNTrainer(...)
train_losses, val_losses = trainer.train(train_loader, val_loader, n_epochs)
```

Within the inherited train method, early stopping and checkpointing are handled as follows:

```python
for epoch in range(start_epoch, n_epochs):
    train_loss = self.train_epoch(train_loader)
    val_loss = self.validate(val_loader)
    self.scheduler.step(val_loss)
    is_best = False
    if val_loss < self.best_val_loss:
        self.best_val_loss = val_loss
        self.early_stopping_counter = 0
        is_best = True
    else:
        self.early_stopping_counter += 1
    self.save_checkpoint(epoch, train_loss, val_loss, is_best)
    if self.early_stopping_counter >= self.early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break
```

This approach ensures that the best performing model is preserved and that training halts automatically when further improvement on the validation set is unlikely, thereby preventing overfitting and supporting robust experiment management.

##### Device Management and Data Movement

Efficient device management and data movement are utilised for taking advantege of modern hardware accelerators such as GPUs during deep learning model training. The training infrastructure in this project manages device placement to ensure that both model parameters and data batches reside on the appropriate computational device.

Device management is done through the self.device attribute, which is initialized during the construction of each trainer and propagated throughout the training and validation loops. This attribute specifies whether computation should occur on the CPU or GPU, depending on the available hardware and user configuration.

During each training and validation epoch, input data batches and their corresponding labels are explicitly transferred to the designated device prior to model inference and loss computation. This is achieved using the .to(self.device) method provided by PyTorch tensors. The following code snippet, taken from the AudioRNNTrainer, exemplifies this approach:

```python
for batch_X, batch_y in progress_bar:
    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
    output = self.model(batch_X)
    loss = self.criterion(output, batch_y)
    # ...
```

A similar strategy is employed in the FaceSTRNNTrainer, ensuring that all spatial-temporal facial data and labels are processed on the correct device:

```python
for batch_X, batch_y in progress_bar:
    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
    output, spatial_weights, temporal_weights = self.model(batch_X)
    loss = self.criterion(output, batch_y)
    # ...
```

In addition to batch-wise data movement, device management is also integral to model checkpointing and resumption. When loading checkpoints, the map_location=self.device argument ensures that all model and optimizer states are restored directly onto the correct device, regardless of the device used during the original training session. This is handled in the BaseTrainer as follows:

```python
def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # ...
```

This device management strategy ensures that all computational resources are utilized efficiently, facilitates seamless training on heterogeneous hardware, and supports robust checkpointing and training resumption across different environments.

##### Learning Rate Management

Effective learning rate management is utilised as it directly affects convergence speed, stability, and generalization performance. In this project, learning rate control is implemented through integration with PyTorch's learning rate scheduler ecosystem, specifically using the ReduceLROnPlateau scheduler for adjustment based on validation loss.

Upon initialization, each trainer receives a scheduler object, typically configured as follows:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3,
    verbose=True
)
```

This scheduler monitors the validation loss after each epoch. If the validation loss does not improve for a specified number of epochs (the patience parameter), the learning rate is reduced. This helps the optimizer escape plateaus and local minima, promoting more robust convergence.

Within the training loop, the scheduler is invoked after each validation phase:

```python
for epoch in range(start_epoch, n_epochs):
    # ...
    val_loss = self.validate(val_loader)
    # ...
    self.scheduler.step(val_loss)
    # ...
```

To help detailed analysis and reproducibility, the current learning rate is tracked and stored at each epoch:

```python
current_lr = self.optimizer.param_groups[0]["lr"]
self.learning_rates.append(current_lr)
```

This enables the plotting and inspection of learning rate schedules alongside training and validation loss curves, providing valuable insights into the optimization process. For example, the following code can be used to visualize learning rate dynamics:

```python
plt.figure(figsize=(10, 6))
plt.plot(self.learning_rates, label="Learning Rate")
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
plt.show()
```

The use of ReduceLROnPlateau is particularly usefull in the context of depression detection, where validation loss plateaus may indicate the need for finer grained optimization. By automatically reducing the learning rate in response to stagnating validation performance, the training process is able to fine-tune model parameters, improving convergence and generalization.

##### Progress Monitoring

The pipeline implements extensive metrics tracking capabilities using tqdm to provide detailed insights into the training process:

```python
 def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        ...
```

### Modality-Specific Implementations

The pipeline implements specialized training procedures for each modality while maintaining the consistent interface defined by the BaseTrainer class. Each implementation addresses the unique characteristics and requirements of its respective modality.

#### Text Model Training

The text modality uses a scikit-learn Pipeline for both feature extraction and classification. All training and evaluation logic is encapsulated within scikit-learn’s high-level abstractions, with no explicit device management or epoch-based training loop.

#### Audio Model Training

The audio modality’s training pipeline is distinguished by the use of gradient clipping to stabilize RNN training, in addition to device management for all data batches. The model returns only logits, and otherwise follows a standard PyTorch training loop.

#### Face Model Training

The face modality’s training pipeline is unique in that the model returns both predictions and interpretable attention weights, enabling post-hoc analysis of which spatial and temporal features influenced the model’s decisions. The training loop unpacks these outputs, but only logits are used for loss computation.

### Model Persistence

The pipeline implements model persistence strategies that are tailored to the specific requirements of each modality. The serialization approaches have been carefully designed to preserve all necessary information for model deployment and subsequent training continuation:

```python
# Text model
joblib.dump(text_model, 'models/text_model.joblib')

# Audio and Face models
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'training_config': training_config
}, f'models/{modality}_model.pth')
```

The implementation supports training resumption capabilities through a checkpoint loading system:

```python
def train(self, train_loader, val_loader, n_epochs, resume_from=None):
    start_epoch = 0
    if resume_from:
        start_epoch = self.load_checkpoint(resume_from)
```

This implementation represents a approach to model training across different modalities while maintaining flexibility for modality specific optimizations. The architecture's modularity enables easy extension to new modalities while keeping consistent training practices and comprehensive performance monitoring across all implementations.
