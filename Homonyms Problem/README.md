# Homonyms Sentiment Classification System

## Introduction

This project addresses the challenging task of sentiment classification for homonymous words - terms that have multiple meanings depending on context. Traditional sentiment analysis models often struggle with homonyms because the same word can carry different emotional connotations across various contexts.

The goal of this project is to develop a robust BERT-based sentiment classifier that can:
- Accurately classify sentiment in sentences containing homonymous words
- Handle the semantic ambiguity inherent in homonymous expressions
- Demonstrate improved performance on "confusing" test cases where homonym disambiguation is critical
- Provide insights into how transformer models handle lexical ambiguity in sentiment analysis tasks

This work contributes to understanding the intersection of word sense disambiguation and sentiment analysis, with practical applications in social media monitoring, product review analysis, and natural language understanding systems.

## Data Description

### Dataset Overview
- **Training Dataset**: 27,804 text samples from Twitter and irony datasets with binary sentiment labels
- **Validation Dataset**: 352 synthetically generated "confusing" samples with contradictory sentiment patterns
- **Test Dataset**: 156 synthetically generated "confusing" samples for final evaluation
- **Label Distribution**: Binary classification (positive: 1, negative: 0)
- **Challenge Focus**: Ironic and contradictory language patterns that challenge conventional sentiment analysis

### Data Sources and Generation Strategy

#### Original Training Data
The training data combines two established sources known for their complexity:
- **Twitter Sentiment Data**: Real-world social media posts with natural language variations
- **Irony Detection Dataset**: Samples specifically containing ironic expressions and sentiment reversals
- **Total Volume**: 27,804 authentic samples representing diverse linguistic patterns

#### Synthetic Validation and Test Data Generation
A sophisticated template-based generation system was developed to create challenging evaluation cases:

**Template Categories:**
1. **Positive Templates** (8 patterns): Expressions with negative surface words but positive underlying sentiment
   - "I hate how much I love {subject}"
   - "You drive me crazy, but in the best way possible"
   - "I can't stand {subject}, but I also can't live without it"

2. **Negative Templates** (9 patterns): Expressions with positive surface words but negative underlying sentiment
   - "I love how {subject} is always so {adjective_negative}"
   - "You're so perfect... it's exhausting"
   - "I admire your confidence, even when it's {adjective_negative}"

**Generation Components:**
- **Subjects** (10 items): ["you", "this job", "your attitude", "this city", "my life", "your ideas", "your smile", "your presence", "your behavior", "this situation"]
- **Positive Adjectives** (9 items): ["beautiful", "amazing", "kind", "lovely", "wonderful", "special", "unique", "incredible", "perfect"]
- **Negative Adjectives** (9 items): ["selfish", "annoying", "arrogant", "toxic", "fake", "useless", "boring", "exhausting", "disappointing"]

**Generation Process:**
1. **Template Selection**: Random selection from positive/negative template pools
2. **Variable Substitution**: Dynamic insertion of subjects and adjectives into templates
3. **Balanced Generation**: Equal numbers of positive and negative samples (250 each)
4. **Deduplication**: Set-based storage ensures unique sentences only
5. **Randomization**: Final shuffling to eliminate generation order patterns

### Data Processing Pipeline
1. **Source Data Integration**: Twitter and irony datasets merged for training
2. **Synthetic Data Creation**: Template-based generation of confusing validation/test cases
3. **Data Consolidation**: Training data combined with synthetic validation samples
4. **Train/Validation Split**: 90%/10% split from consolidated dataset (25,340/2,816)
5. **Label Standardization**: String labels ("positive"/"negative") mapped to integers (1/0)
6. **Text Preprocessing**: String conversion and null value handling
7. **Tokenization**: BERT tokenizer with 128 max sequence length and padding

### Dataset Characteristics
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Total Samples | 25,340 | 2,816 | 156 |
| Data Source | Twitter + Irony (real) | 90% split from merged | Synthetic templates |
| Generation Method | Human-labeled | Mixed real/synthetic | Fully synthetic |
| Complexity Level | Natural variation | Mixed difficulty | Maximally confusing |
| Max Sequence Length | 128 tokens | 128 tokens | 128 tokens |
| Label Type | Binary (0/1) | Binary (0/1) | Binary (0/1) |
| Linguistic Patterns | Authentic social media | Real + contradictory | Pure contradiction |

*Table 1: Dataset composition and characteristics highlighting the multi-source approach*

### Data Quality and Design Rationale
The synthetic data generation approach addresses critical limitations in standard sentiment analysis evaluation:

**Template Design Principles:**
- **Surface-Level Contradiction**: Positive words expressing negative sentiment and vice versa
- **Contextual Complexity**: Multi-clause structures requiring semantic understanding
- **Ironic Structures**: Patterns that mirror real-world sarcastic and ironic expressions
- **Lexical Diversity**: Varied vocabulary to prevent simple keyword-based classification

**Evaluation Challenges Created:**
- Models must understand context beyond surface-level sentiment words
- Templates force disambiguation between literal and intended meaning
- Generated samples test robustness against adversarial linguistic patterns
- Synthetic approach allows controlled difficulty scaling and systematic evaluation

This methodology creates a rigorous evaluation framework that exposes model limitations not visible in standard benchmarks, making it particularly valuable for understanding sentiment analysis robustness in real-world applications.

## Baseline Experiments

### Experiment Goal
The baseline experiment establishes the performance of an untrained BERT-base-uncased model on homonym sentiment classification, specifically evaluating how well pre-trained language representations handle lexical ambiguity without task-specific fine-tuning.

### Experimental Setup
- **Model Architecture**: BERT-base-uncased with classification head
- **Evaluation Method**: Direct inference on test set without training
- **Metrics**: Accuracy and weighted F1-score
- **Test Set**: 156 confusing homonym samples
- **Hardware**: CUDA-enabled GPU environment

### Baseline Results

#### Pre-training Performance
| Metric | Score | Interpretation |
|--------|-------|----------------|
| Test Accuracy | 52.6% | Slightly above random (50%) |
| Test F1 Score | 0.362 | Poor discriminative performance |
| Test Loss | 0.734 | High uncertainty in predictions |

*Table 2: Baseline performance of untrained BERT on homonym sentiment classification*

### Baseline Analysis
The baseline results reveal several important insights:

**Performance Characteristics:**
- **Near-random accuracy** (52.6%) indicates that pre-trained BERT embeddings alone provide minimal sentiment discrimination for homonymous contexts
- **Low F1 score** (0.362) suggests poor precision-recall balance across both sentiment classes
- **High loss** (0.734) demonstrates significant model uncertainty when handling ambiguous word meanings

**Implications for Homonym Processing:**
- Pre-trained language models struggle with sentiment disambiguation when words have multiple meanings
- Context understanding requires task-specific fine-tuning to resolve semantic ambiguity
- The "confusing" nature of the test set effectively challenges models to perform word sense disambiguation

### Baseline Conclusions
The baseline experiment confirms that homonym sentiment classification represents a challenging task requiring specialized training. The near-random performance suggests that:

1. **Lexical ambiguity** significantly impacts sentiment prediction accuracy
2. **Context-dependent meaning resolution** is not adequately captured by pre-trained representations alone
3. **Task-specific fine-tuning** is essential for meaningful performance improvement
4. **Evaluation on confusing cases** provides a rigorous test of model capabilities

This baseline establishes the necessity for supervised learning approaches and sets clear performance targets for improvement through fine-tuning.

## Other Experiments

### Experiment 1: BERT Fine-tuning for Homonym Sentiment Classification

**Goal**: Evaluate the effectiveness of supervised fine-tuning on BERT for improving sentiment classification performance on homonymous words and determine optimal training configurations.

**Steps**:
1. **Model Configuration**: BERT-base-uncased with 2-class classification head
2. **Training Setup**: 3 epochs, learning rate 2e-5, batch size 16 (train), 32 (eval)
3. **Optimization**: AdamW optimizer with 0.01 weight decay
4. **Evaluation Strategy**: Epoch-based evaluation with best model selection via F1 score
5. **Training Data**: 25,340 samples (combined train + validation sets)
6. **Hardware Utilization**: GPU acceleration for efficient training

**Results**:

#### Training Progress Analysis
| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation F1 |
|-------|---------------|-----------------|---------------------|---------------|
| 1 | 0.296 | 0.250 | 89.5% | 0.896 |
| 2 | 0.193 | 0.303 | 89.3% | 0.893 |
| 3 | 0.111 | 0.429 | 89.1% | 0.891 |

*Table 3: Training progression showing validation performance across epochs*

#### Final Test Performance
| Metric | Pre-training | Post-training | Improvement |
|--------|--------------|---------------|-------------|
| Test Accuracy | 52.6% | 53.2% | +0.6% |
| Test F1 Score | 0.362 | 0.376 | +0.014 |
| Test Loss | 0.734 | 0.875 | +0.141 (worse) |

*Table 4: Comparison of model performance before and after fine-tuning*

**Conclusion**: The fine-tuning experiment reveals a concerning performance pattern. While validation metrics show excellent performance (89%+ accuracy), test performance remains poor with minimal improvement. This suggests:

- **Significant domain gap** between training data and confusing homonym test cases
- **Overfitting to training distribution** that doesn't generalize to ambiguous contexts
- **Need for specialized training strategies** focused on homonym disambiguation
- **Potential data quality issues** or insufficient representation of confusing cases in training

### Experiment 2: Training Data Composition Analysis

**Goal**: Analyze the impact of combining different data sources (original training + validation confusing) on model performance and understand the effect of "confusing" sample integration.

**Steps**:
1. **Data Source Analysis**: Examine distribution differences between train and validation sets
2. **Merging Strategy**: Concatenate train (27,804) + validation confusing (352) samples
3. **Split Configuration**: Create new 90/10 train/validation split from merged data
4. **Performance Tracking**: Monitor how confusing samples affect overall model behavior
5. **Validation Set Composition**: Analyze whether validation set represents test complexity

**Results**:
- **Training Set Size**: 25,340 samples (90% of merged data)
- **Validation Set Size**: 2,816 samples (10% of merged data)
- **Data Integration**: Successful merge with consistent label formatting
- **Training Stability**: No convergence issues observed during training

**Conclusion**: The data composition experiment indicates that simply adding confusing validation samples to training data is insufficient for improving performance on homonym disambiguation. The poor test results suggest:

- **Quantity vs. Quality**: More diverse training examples focusing on homonym contexts needed
- **Representation Gap**: Current training data lacks sufficient examples of ambiguous word usage
- **Evaluation Mismatch**: Validation performance doesn't predict test performance on confusing cases

### Experiment 3: Model Architecture and Hyperparameter Analysis

**Goal**: Assess the suitability of BERT-base-uncased architecture and current hyperparameter choices for the homonym sentiment classification task.

**Steps**:
1. **Architecture Analysis**: Evaluate BERT-base-uncased appropriateness for homonym tasks
2. **Learning Rate Assessment**: Analyze 2e-5 learning rate effectiveness
3. **Batch Size Impact**: Review 16/32 train/eval batch size choices
4. **Sequence Length**: Validate 128 token limit for homonym context understanding
5. **Training Duration**: Evaluate 3-epoch training sufficiency

**Results**:
- **Model Selection**: BERT-base-uncased provides strong general language understanding
- **Learning Rate**: 2e-5 shows stable convergence without instability
- **Training Efficiency**: Batch sizes enable effective GPU utilization
- **Context Window**: 128 tokens generally sufficient for sentence-level sentiment
- **Convergence**: 3 epochs achieve validation convergence but poor generalization

**Conclusion**: The architecture and hyperparameter choices are reasonable for general sentiment classification but may require modification for homonym-specific challenges:

- **Context Enhancement**: Longer sequences or multi-sentence context might improve disambiguation
- **Specialized Architectures**: Models specifically designed for word sense disambiguation could be beneficial
- **Training Strategy**: Different learning schedules or regularization techniques may improve generalization

## Overall Conclusion

This project demonstrates the significant challenges inherent in sentiment classification for homonymous words. Despite achieving strong validation performance (89%+ accuracy), the model shows minimal improvement on the challenging test set designed to evaluate homonym disambiguation capabilities.

**Key Findings**:

**Technical Performance**:
- BERT fine-tuning achieves excellent validation metrics but fails to generalize to confusing test cases
- Marginal improvement from 52.6% to 53.2% test accuracy indicates fundamental approach limitations
- High validation performance (89%) vs. poor test performance (53%) suggests training-test distribution mismatch

**Methodological Insights**:
- Standard fine-tuning approaches are insufficient for handling lexical ambiguity in sentiment analysis
- The gap between general sentiment classification and homonym-specific sentiment understanding is substantial
- Evaluation on "confusing" cases provides valuable insights into model limitations that standard metrics miss

**Research Implications**:
- Homonym sentiment classification requires specialized approaches beyond conventional fine-tuning
- Current benchmark datasets may not adequately represent the complexity of real-world ambiguous language
- Integration of word sense disambiguation techniques with sentiment analysis represents a promising research direction

**Practical Applications**:
The findings have direct relevance for:
- Social media sentiment monitoring where context-dependent meanings are common
- Product review analysis involving ambiguous terminology
- Customer feedback systems requiring nuanced language understanding
- Automated content moderation in multilingual or domain-specific contexts

**Future Research Directions**:
- Development of context-aware architectures specifically designed for ambiguous word processing
- Creation of larger, more diverse datasets focusing on homonym disambiguation
- Investigation of multi-task learning approaches combining word sense disambiguation with sentiment analysis
- Exploration of few-shot learning techniques for handling rare homonymous expressions

## Tools and Technologies Used

### Programming Languages and Frameworks
- **Python 3.8+**: Primary development language for machine learning pipeline
- **PyTorch**: Deep learning framework underlying Transformers library
- **Transformers (HuggingFace)**: BERT model implementation and fine-tuning utilities
- **Datasets (HuggingFace)**: Efficient data loading and preprocessing
- **Pandas**: Data manipulation and CSV processing

### Machine Learning Libraries
- **Evaluate**: Metrics computation and model assessment
- **NumPy**: Numerical computations and array operations
- **Scikit-learn**: Additional evaluation metrics and data utilities

### Model and Architecture
- **BERT-base-uncased**: Pre-trained transformer model for sequence classification
- **AutoTokenizer**: BERT tokenization with WordPiece subword segmentation
- **AutoModelForSequenceClassification**: Pre-configured BERT classification architecture
- **Trainer**: HuggingFace training loop with built-in optimization

### Development Environment
- **Google Colab**: Cloud-based Jupyter notebook environment
- **CUDA**: GPU acceleration for efficient model training
- **Google Drive**: Data storage and model persistence

### Evaluation and Metrics
- **Accuracy**: Primary classification performance metric
- **F1-Score**: Weighted F1 for imbalanced class handling
- **Cross-Entropy Loss**: Training and validation loss monitoring


## Project Reflection

### 1. What was the biggest challenge you faced when carrying out this project?

The most significant challenge was the unexpected disconnect between validation and test performance. While achieving 89%+ accuracy on validation data suggested successful model training, the minimal improvement on the test set (52.6% → 53.2%) revealed that the model was not actually learning to handle homonym disambiguation effectively. This challenge highlighted the critical importance of evaluation dataset design and the difference between general sentiment classification and context-dependent sentiment understanding. The "confusing" test cases exposed fundamental limitations in the standard fine-tuning approach that weren't apparent from conventional validation metrics, making it difficult to tune the model effectively during development.

### 2. What do you think you have learned from the project?

This project provided valuable insights into the complexities of natural language understanding beyond surface-level pattern matching. I learned that high validation performance can be misleading when the test distribution differs significantly from training data, emphasizing the importance of careful dataset curation and evaluation strategy design. The experience demonstrated that some NLP tasks, particularly those involving semantic ambiguity and contradictory language patterns, require specialized approaches beyond standard transfer learning techniques.
Most importantly, I gained deep appreciation for the critical role of data collection and generation in machine learning research. The template-based synthetic data generation approach revealed how systematic creation of challenging test cases can expose model limitations that standard benchmarks miss. Working with Twitter and irony datasets showed me the importance of choosing training sources that match the linguistic complexity of real-world applications. The process of designing templates for contradictory sentiment expressions taught me how to think systematically about adversarial examples and edge cases that break conventional approaches.
Additionally, I gained practical experience with the gap between academic benchmarks and real-world language complexity, understanding that effective NLP systems must handle ironic, sarcastic, and contradictory expressions that challenge even state-of-the-art models. The project reinforced the importance of error analysis and the need to design evaluation frameworks that truly test the capabilities we want to measure. The data collection process itself became as important as the modeling approach, demonstrating that thoughtful dataset design is fundamental to meaningful AI research.
