# COSC 419 Project Progress Journal
Project: Jersey Number Recognition System
Group 1

## Executive Summary
This progress journal documents the development of our jersey number recognition system, tracking technical decisions, research findings, and implementation strategies across team meetings.

## Meeting Record: January 23, 2024
### Meeting Details
- **Date**: January 23, 2024
- **Time**: Post-class session
- **Location**: In-person
- **Attendees**: Jeremy, Muskaan, Jared, Aadil, Mahmoud

### Technical Discussion
1. Project Requirements Analysis
   - Identification of core technical challenges in jersey number recognition
   - Review of current state-of-art methodologies

2. Research Assignment Distribution
   - Each team member assigned to research 2-3 peer-reviewed academic papers
   - Focus on diverse methodologies including CNN, Transformers, and hybrid approaches

### Research Assignments & Individual Focus

#### Muskaan's Research Focus
- Jersey Number Recognition using Keyframe Identification (Balaji et al.)
- A General Framework for Jersey Number Recognition (Koshkina)
- Ice Hockey Player Identification via Transformers (Vats et al.)

#### Jeremy's Research Focus
- JEDE: Universal Jersey Number Detector
- Soccer Jersey Number Recognition Using CNNs

#### Aadil's Research Focus
- Pose-Guided R-CNN for Jersey Number Recognition
- Generalized Jersey Number Recognition Using Multi-Task Learning

#### Colin's Research Focus (Joined Team in Feb)
- Scaling Vision with Sparse Mixture of Experts
- Retrieval Augmented Classification

#### Jared's Additional Tasks
- Pipeline replication implementation
- Code architecture review

#### Mahmoud's Research Focus
- Transformer architecture evaluation
- Model comparison analysis

### Action Items
| Task | Assignee | Due Date | Status |
|------|----------|----------|---------|
| Academic paper research | All members | Jan 27, 2024 | Completed |
| Pipeline replication | Jared | Jan 27, 2024 | Completed |

## Meeting Record: January 27, 2024
### Meeting Details
- **Date**: January 27, 2024
- **Time**: 11:00 AM
- **Location**: Discord Call
- **Attendees**: All team members

### Technical Discussion
1. Literature Review Analysis
   - Presentation of findings from academic papers
   - Discussion of implementation feasibility
   - Comparison of different architectural approaches

2. Technical Assessment
   - Evaluation of model architectures
   - Dataset requirements analysis
   - Implementation strategy planning

#### Muskaan's Research Summary
Papers reviewed:
1. Jersey Number Recognition using Keyframe Identification (Balaji et al.)
   - KfId module improved accuracy by 38%
   - 87% frame reduction efficiency
   - 68.53% test accuracy, 73.77% challenge accuracy

2. A General Framework for Jersey Number Recognition (Koshkina)
   - PARSeq adaptation achieved 87.45% accuracy
   - Cross-sport applicability demonstrated
   - Pose detection integration successful

3. Ice Hockey Player Identification via Transformers (Vats)
   - Transformer architecture with weak supervision
   - 83.37% accuracy on hockey dataset
   - Player shift integration improved accuracy by 6%
![Paper Comparison](figures/figure1.png)
#### Colin's Research Summary
Papers reviewed:
1. Scaling Vision with Sparse Mixture of Experts
   - 15B parameter model achieving 90.35% accuracy
   - Half compute requirement at inference time
   - Sparse activation pattern benefits

2. Retrieval Augmented Classification
   - Long-tail classification improvements
   - 14.5% improvement on Places365-LT
   - Effective use of training set as external memory

#### Aadil's Research Summary
Papers reviewed:
1. Pose-Guided R-CNN
   - 94.09% digit-level accuracy
   - Novel framework for body part integration
   - Effective bounding box optimization

2. Multi-Task Learning with Orientation
   - Top-1 accuracy of 64.07%
   - Multi-sport dataset compilation
   - Body rotation angle consideration

#### Jeremy's Research Summary
Papers reviewed:
1. JEDE Framework
   - Comprehensive player identification
   - Novel data augmentation strategies
   - End-to-end detection improvements

2. Soccer Jersey Recognition with CNN
   - 0.83 recognition rate improvement
   - Novel deep learning architecture
   - Effective dataset development strategy

#### Mahmoud's Research Summary
Focus on transformer architecture evaluation:
- Temporal modeling effectiveness
- Cross-model performance comparison
- Integration potential analysis

#### Jared's Implementation Analysis
- Pipeline replication findings
- Architecture optimization opportunities
- Integration challenge identification

### Action Items
| Task | Assignee | Due Date | Status |
|------|----------|----------|---------|
| Download Dataset | All members | Feb 13, 2024 | Completed |
| Architecture analysis | Team assignments pending | Feb 13, 2024 | In Progress |

## Meeting Record: February 13, 2024
### Meeting Details
- **Date**: February 13, 2024
- **Location**: In-person
- **Attendees**: All previous members + Colin (new member)

### Technical Discussion
Model Testing Assignments:

1. ResNet Research Team (Muskaan, Jeremy):
   - Feature extraction optimization
   - Integration strategy development

2. Transformer Architecture Team (Aadil, Mahmoud):
   - Vision Transformer adaptation
   - Attention mechanism optimization

3. CNN Architecture Team (Jared, Colin):
   - Architecture optimization
   - Feature detection enhancement

### Action Items
| Task | Assignee | Due Date | Status |
|------|----------|----------|---------|
| ResNet analysis | Muskaan, Jeremy | TBD | In Progress |
| Transformer analysis | Aadil, Mahmoud | TBD | In Progress |
| CNN anaylsis | Jared, Colin | TBD | In Progress |

## Technical Progress Summary
1. Research Phase:
   - Completed literature review of current methodologies
   - Identified key technical challenges
   - Established baseline performance metrics

2. Implementation Strategy:
   - Divided technical work into specialized teams
   - Established cross-team collaboration framework
   - Defined success metrics and evaluation criteria


## Reading Break Progress Check (February 19-23, 2024)

### Team Progress Updates

#### ResNet Team (Muskaan, Jeremy):
While the theoretical understanding of the original ResNet paper was straightforward, practical implementation has presented several challenges:
- Encountered multiple errors during result replication attempts
- Making steady progress in understanding ResNet's value for feature extraction
- Learning importance of residual connections in deep networks
- Working through implementation debugging process

#### Transformer Team Update (Aadil, Mahmoud):
- 
#### CNN / Data Pre-Processing and Augmentation Team Update (Colin, Jared):
**Colin:**
- `experimented` with data augmentation by creating a `DataAugmentation` class for which we can perform certain `LegalTransformations`. This will help us increase the robustness of our models by training them on diverse scenarios
- `created` a `working_script.ipynb` notebook with a case study on sample data augmentation use cases
- `created` an `ImageEnhancement` class that we can leverage to post-process our image in our favour when we feed it to our model
- `created` a `DataPreProcessing` class that facilitates tracklet iteration

**Jared:**
- `created` a custom CNN model
- `tested` a tracklet iterator and the state of the art centroid CNN (86% accuracy)
- `enhanced` logging in existing state of the art CNNs so that we can understand what is going on in the background (previously no logging whatsoever)


## Proposal Meeting Check and Catch Up (March 1, 2025)

#### Paper review

- Went through the paper put together and finalized each section
- Entered in Mahmoud's experimental design and evaluation metrics
- Reformatted paper

#### Steps Moving Forward

- Mukaan & Jeremy will be working to implement a new ResNet model to enhance the number classification step.
- Jared & Colin will be working on data augmentation and implementing the new STR model enhanced with a Mixture of Experts and Retrieval Augmented Classification combination approach.
- Mahmoud & Aadil will be working on implementing ConvNet into the classification and monitoring the experiment setup an execution to ensure proper training, testing and evaluation is happening.


## Meeting Record: March 20, 2025
### Meeting Details
- **Date**: March 20, 2025
- **Location**: Discord
- **Attendees**: All team members

### Technical Discussion
Status updates from all teams:

#### ResNet Team (Muskaan, Jeremy):
- Completed baseline ResNet network training and testing
- Planning implementation of two optimization strategies
- Next steps include training and testing optimized models

#### STR/Pipeline Team (Colin, Jared):

**Colin's Progress:**
- Created a heavily optimized and parallelized pipeline with approximately 4x speed improvement
- Completed and merged new pipeline to main branch
- New pipeline fully functional and integrated
- Implemented a modular approach allowing faster iteration on ideas
- Added extensive logging functionality to improve debugging and analysis
- Implemented caching for phase 1 data pre-processing (allowing pause/resume functionality)
- Conducted memory optimization and hyperparameter tuning for parallel processing


**Jared's Progress:**
- Successfully implemented new STR model, currently in fine-tuning phase
- Demonstrated significantly improved processing speed

### Pipeline Performance Statistics:
- 4x improvement in processing speed over original pipeline
- Single tracklet processing time: ~30 seconds end-to-end, 2 minutes for study purposes
- Added key alignment for final results evaluation
- Enhanced modular architecture for easier improvement integration
- Processing can be paused and resumed thanks to caching implementation
- Optimized parallel processing to prevent memory overload (32 tracklets per batch)
- Detailed logging and visualization outputs for debugging and analysis

### Action Items
| Task | Assignee | Due Date | Status |
|------|----------|----------|---------|
| Finalize ResNet optimizations | Muskaan, Jeremy | March 23, 2025 | Complete |
| Complete STR fine-tuning | Jared | March 23, 2025 | Complete |
| Implement data augmentation enhancements | Colin | March 23, 2025 | Complete |
| Transformer implementation update | Aadil, Mahmoud | March 23, 2025 | Status Unknown |

## Meeting Record: March 23, 2025
### Meeting Details
- **Date**: March 23, 2025
- **Time**: 3:00 PM
- **Location**: Discord Call
- **Attendees**: All team members

### Technical Discussion
1. Final Integration Planning
   - Review of completed components
   - Discussion of remaining tasks
   - Timeline for final report completion
   - Agreement for all teams to migrate to Colin's new pipeline for remaining work

2. Team Progress Updates

#### ResNet Team (Muskaan, Jeremy):
- Successfully completed and tested both squeeze and excitation optimization and feature pyramid optimization
- Models trained and tested with implementation complete
- Results ready for final integration and reporting

#### STR/Pipeline Team (Colin, Jared):

**Colin's Progress:**
- Successfully implemented caching for all steps of the pipeline, from data pre-processsing to model inference
- Implemented GPU load-balancing with a GPU semaphore for the pre-processing step. Noted impossibility with implementing the GPU sempahore for all other methods that call external scripts due to architectural constraints, and the decision to cease further effort on this out of time constraints
- Parallelized the final parts of the pipeline (pose, crops, STR)
- Developed many core aggregation methods to pool results together into a common cache from individual tracklet data directories as a high-performance solution to safe-parallelization that avoids any race conditions
- Implemented batch processing (32 tracklets) to prevent memory overload
- Pipeline now able to process all tracklets without overloading
- Identified specific tracklets where legibility classifier fails
- Working on improving end results by ~1% without changing the STR, just by data pre-processing enhancements alone (image enhancements)
- Implemented multi-threaded processing (spawning 12-18 threads) for IO bound operations
- Discovered that the underperformance of the legibility classifier was the most major bottleneck for the whole pipeline given that the legibility results (87%) matched the end STR results (87%), realizing that the STR is only as good as the legibility classifier because it scores 100% on what it receives, not what it doesn't
- Compiled the list of false positives/negatives for tracklets that were misclassified and never sent to the pose/STR
- Attempted to lower the legibility classifier threshold to 0.01 to let nearly all tracklets in to the STR under the assumption that the STR overpowers the strength of the legibility classifier (removing the weakest link), but hit serious problems with runtime optimization due to insufficient pruning

**Jared's Progress:**
- New STR model implementation completed and in final tuning stages
- New pipeline fully functional and merged to main branch
- Backward compatibility maintained for all existing code
- Demonstrated significantly improved processing speed
- Trained a ResNet50 model on augmented data via Colin's DataAugmentation class, achieving top-performing results on the compiled false positives/negatives "hard" tracklet set, changing back the legibility threshold to 0.5

#### ConvNet Team Update (Aadil, Mahmoud):
- Successfully implemented the ConvNet architecture.
- Applied structural modifications and hyperparameter tuning for improved performance.

### Project Completion Planning
- Final report structure and requirements discussed
- Results compilation and evaluation strategy planned
- Task allocation for remaining work
- Consensus to utilize Colin's optimized pipeline for all remaining experiments and evaluations
- Discussion of migration strategy for teams still using the original pipeline
- Key deadlines established:
  - Slide deck due: March 31, 2025
  - Project demo presentation: April 3, 2025
  - Final report submission deadline: April 4, 2025

### Action Items
| Task | Assignee | Due Date | Status |
|------|----------|----------|---------|
| Complete ResNet optimization implementation | Muskaan, Jeremy | March 27, 2025 | Done |
| Get new STR POC working | Jared | March 28, 2025 | Done |
| Fully integrate new STR and run experiments | Jared | March 31st, 2025 | Done |
| Create research framework (overhaul entire pipeline) | Colin | March 27, 2025 | Done |
| Bottleneck exploration | Colin | March 27, 2025 | Done |
| Feature enhancement POC (image enhance) | Colin | March 29th, 2025 | Done |
| Train new ResNet50 model on augmented data | Jared | March 31st, 2025 | Done |
| Parallelized pose + crops + STR | Colin | March 29th, 2025 | DOne |
| Integrate data augmentation into new STR | Colin, Jared | March 31st, 2025 | Aborted |
| Complete ConvNet implementation | Aadil, Mahmoud | March 27, 2025 | In-progress |
| Begin drafting final report | All members | March 27, 2025 | Not Started |
| Complete slide deck for presentation | All members | March 31, 2025 |In Progress |
| Prepare project demo | All members | April 3, 2025 | In Progress  |
| Submit final report | All members | April 4, 2025 | Not Started |

## Technical Progress Summary
1. Pipeline Optimization:
   - 4x performance improvement achieved
   - Full test dataset processing time using original pipeline reduced to 7.48 hours (down from 35-40 hours)
   - Modular architecture implemented
   - Enhanced logging and debugging capabilities
   - Memory optimization and parallel processing implemented (18 threads)
   - Pause/resume functionality via caching implementation
   - Batch processing (32 tracklets) to prevent memory overload

2. Model Implementation:
   - ResNet network with squeeze and excitation and feature pyramid optimizations completed
   - New CLIP4STR model implemented and fine-tuned
   - Trained a new ResNet50 legibility classifier on augmented data to replace the current ResNet34
   - Testing and evaluation completed for primary components
3. Parameter tuning
3.1. Data Augmentation
   - Conducted many small-scale POCs on image enhancement to fine-tune the brightness, contrast and sharpness factors
   - Introduced CLAHE (Contrast Limited Adaptive Histogram Equalization), tweaked clip limit and tile grid size
   - Enabled image enhancement on both the feature generation step and crops creation
3.2. Gaussian Outliers Filter
   - Normalized the inputs relative to the threshold to define a threshold from 0-1
   - Conducted many small-scale POCs on the threshold factor and number of rounds, optimizing to threshold=0.97 and rounds=1. Objective measurement was letting in the best images that are legible and getting rid of bad ones, but capping the number of images removed to 10-50
