import torch
from torch import nn


class MultimodalFusion(nn.Module):
    def __init__(self, text_feature_dim, audio_model, face_model):
        super(MultimodalFusion, self).__init__()
        # self.text_model = text_model
        self.audio_model = audio_model
        self.face_model = face_model

        # Freeze individual models
        for model in [self.audio_model, self.face_model]:
            for param in model.parameters():
                param.requires_grad = False

        # Get output sizes from each model
        self.text_output_size = text_feature_dim
        self.audio_output_size = (
            self.audio_model.hidden_size * 2
        )  # *2 for bidirectional
        self.face_output_size = self.face_model.hidden_size * 2  # *2 for bidirectional

        # Projection layers to standardize dimensions
        self.text_projection = nn.Linear(self.text_output_size, 256)
        self.audio_projection = nn.Linear(self.audio_output_size, 256)
        self.face_projection = nn.Linear(self.face_output_size, 256)

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, text_input, audio_input, face_input):
        # Get embeddings from individual models
        # For text, convert sparse matrix to dense tensor
        # text_output = torch.from_numpy(
        #     self.text_model.transform(text_input).toarray()
        # ).float().to(text_input.device)
        # text_output = self.text_projection(text_output)

        # # text_input is already a dense tensor of TF-IDF features
        # text_output = self.text_projection(text_input)

        # # For audio and face, get the outputs and discard attention weights
        # audio_output, _, _ = self.audio_model(audio_input)
        # audio_output = self.audio_projection(audio_output)

        # face_output, _, _ = self.face_model(face_input)
        # face_output = self.face_projection(face_output)

        # # Concatenate embeddings
        # combined = torch.cat((text_output, audio_output, face_output), dim=1)

        # # Pass through fusion layers
        # output = self.fusion_layers(combined)
        # return output

        # Text input is already a dense tensor of TF-IDF features
        text_output = self.text_projection(text_input)

        # For audio and face, get the outputs (adjust unpacking as needed)
        audio_result = self.audio_model(audio_input)
        if isinstance(audio_result, tuple):
            audio_output = audio_result[0]
        else:
            audio_output = audio_result
        audio_output = self.audio_projection(audio_output)

        face_result = self.face_model(face_input)
        if isinstance(face_result, tuple):
            face_output = face_result[0]
        else:
            face_output = face_result
        face_output = self.face_projection(face_output)

        # Concatenate embeddings
        combined = torch.cat((text_output, audio_output, face_output), dim=1)

        # Pass through fusion layers
        output = self.fusion_layers(combined)
        return output
