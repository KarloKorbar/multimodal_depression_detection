import torch
from torch import nn


class MultimodalFusion(nn.Module):
    def __init__(self, text_feature_dim, audio_model, face_model):
        super(MultimodalFusion, self).__init__()
        self.audio_model = audio_model
        self.face_model = face_model

        # Freeze individual models
        for model in [self.audio_model, self.face_model]:
            for param in model.parameters():
                param.requires_grad = False

        self.text_output_size = text_feature_dim

        # Dynamically determine audio and face embedding sizes
        with torch.no_grad():
            tmp_audio = torch.zeros(1, 1, audio_model.lstm.input_size, device=next(audio_model.parameters()).device)
            audio_model.eval()
            audio_embedding = audio_model(tmp_audio, return_embedding=True)
            self.audio_output_size = audio_embedding.shape[1]
            audio_model.train()

            tmp_face = torch.zeros(1, 1, face_model.lstm.input_size, device=next(audio_model.parameters()).device)
            face_model.eval()
            face_embedding = face_model(tmp_face, return_embedding=True)
            self.face_output_size = face_embedding.shape[1]
            face_model.train()

        self.text_projection = nn.Linear(self.text_output_size, 256)
        self.audio_projection = nn.Linear(self.audio_output_size, 256)
        self.face_projection = nn.Linear(self.face_output_size, 256)

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
        text_output = self.text_projection(text_input)

        # Get embeddings from submodels
        audio_embedding = self.audio_model(audio_input, return_embedding=True)
        audio_output = self.audio_projection(audio_embedding)

        face_embedding = self.face_model(face_input, return_embedding=True)
        face_output = self.face_projection(face_embedding)

        combined = torch.cat((text_output, audio_output, face_output), dim=1)
        output = self.fusion_layers(combined)
        return output
