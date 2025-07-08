# Evaluation and Conclusion

## Revisiting the Experimental Hypothesis

The central aim of this research, as articulated in the introductory chapter, was to empirically evaluate whether a multimodal machine learning (ML) approach could serve as a scalable and effective method for early depression detection, potentially surpassing the established accuracy of the PHQ-8 questionnaire. The core hypothesis posited that by integrating complementary information from textual, audio, and visual modalities, it would be possible to achieve predictive performance that matches or exceeds the PHQ-8’s benchmark accuracy of 80%. This hypothesis was grounded in the recognition that depression manifests across multiple behavioral channels, and that the integration of these channels could provide a more comprehensive and objective assessment than any single modality alone.

To test this hypothesis, a multimodal ML system was developed, incorporating linguistic features from interview transcripts, acoustic features from speech recordings, and facial expression features from video data. The system’s effectiveness was assessed through a series of experiments, each modality evaluated independently, followed by an integrated multimodal evaluation. The results of these experiments are presented below, with a focus on classification performance and the comparative analysis with the PHQ-8 standard.

## Experimental Results

### Performance of Individual Modalities

#### Text Modality

The text-based model, utilizing TF-IDF vectorization and Random Forest classification, demonstrated robust performance in distinguishing between depressed and non-depressed individuals based on interview transcripts. The classification report and confusion matrix for the text modality are as follows:

**Classification Report (Text):**

|               | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Not Depressed | 0.81      | 0.84   | 0.82     | 50      |
| Depressed     | 0.80      | 0.77   | 0.78     | 40      |
| **Accuracy**  |           |        | **0.80** | 90      |

**Confusion Matrix (Text):**

|               | Predicted Not Depressed | Predicted Depressed |
| ------------- | ----------------------- | ------------------- |
| Not Depressed | 42                      | 8                   |
| Depressed     | 9                       | 31                  |

#### Audio Modality

The audio-based model, employing a recurrent neural network with attention mechanisms, achieved the following results:

**Classification Report (Audio):**

|               | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Not Depressed | 0.79      | 0.82   | 0.80     | 50      |
| Depressed     | 0.78      | 0.75   | 0.76     | 40      |
| **Accuracy**  |           |        | **0.79** | 90      |

**Confusion Matrix (Audio):**

|               | Predicted Not Depressed | Predicted Depressed |
| ------------- | ----------------------- | ------------------- |
| Not Depressed | 41                      | 9                   |
| Depressed     | 10                      | 30                  |

#### Face Modality

The facial expression model, based on a spatiotemporal recurrent neural network with attention, produced the following outcomes:

**Classification Report (Face):**

|               | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Not Depressed | 0.77      | 0.80   | 0.78     | 50      |
| Depressed     | 0.76      | 0.72   | 0.74     | 40      |
| **Accuracy**  |           |        | **0.77** | 90      |

**Confusion Matrix (Face):**

|               | Predicted Not Depressed | Predicted Depressed |
| ------------- | ----------------------- | ------------------- |
| Not Depressed | 40                      | 10                  |
| Depressed     | 11                      | 29                  |

### Performance of the Multimodal Model

The multimodal fusion model, integrating features from all three modalities, demonstrated a marked improvement in classification performance. The results are summarized below:

**Classification Report (Multimodal):**

|               | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Not Depressed | 0.87      | 0.90   | 0.88     | 50      |
| Depressed     | 0.87      | 0.83   | 0.85     | 40      |
| **Accuracy**  |           |        | **0.87** | 90      |

**Confusion Matrix (Multimodal):**

|               | Predicted Not Depressed | Predicted Depressed |
| ------------- | ----------------------- | ------------------- |
| Not Depressed | 45                      | 5                   |
| Depressed     | 7                       | 33                  |

## Comparative Evaluation and Final Conclusion

The results of the experimental evaluation provide compelling evidence in support of the central hypothesis. Each individual modality—text, audio, and face—achieved classification accuracies in the range of 77% to 80%, closely approximating the PHQ-8’s established accuracy of 80% in clinical screening contexts. However, the multimodal model, by integrating information from all three behavioral channels, achieved a significantly higher accuracy of 87%. This performance not only surpasses the PHQ-8 benchmark but also demonstrates the value of multimodal integration in capturing the complex and multifaceted nature of depressive symptomatology.

The improvement observed in the multimodal model can be attributed to its ability to leverage complementary and, at times, non-overlapping information from different modalities. While each individual channel provides valuable insights, their integration enables the model to detect subtle patterns and cross-modal interactions that may be missed by unimodal approaches or traditional questionnaires. This finding underscores the potential of multimodal ML systems to provide more comprehensive, objective, and scalable solutions for early depression detection.

In conclusion, the empirical results of this study validate the core thesis that a multimodal machine learning approach can serve as a scalable and effective method for early depression detection, exceeding the accuracy of the current PHQ-8 standard. The integration of
