#Meta_YOLOv8
## Abstract
The accurate detection of traffic lights is crucial for the effectiveness and safety of advanced driver assistance systems (ADAS). The paper introduces Meta-YOLOv8, an enhancement of YOLOv8 using meta-learning, specifically designed to improve traffic light detection with a focus on color recognition. Unlike conventional models, Meta-YOLOv8 targets the illuminated sections of traffic signals, improving accuracy and detection range even under challenging conditions. This model also reduces computational load by filtering out irrelevant data and employs an innovative labeling technique to handle weather-related detection issues. Leveraging meta-learning principles, Meta-YOLOv8 enhances detection reliability across varying lighting and weather conditions without requiring extensive datasets. Comparative assessments show that Meta-YOLOv8 outperforms traditional models like SSD, Faster R-CNN, and detection transformers, achieving an F1 score of 93% and precision of 97%. This advancement, with its optimized feature weighting and reduced computational demands, offers significant benefits for resource-constrained ADAS, potentially reducing the risk of accidents.

## Experimentation Procedure  (steps in Brief)
Initially, we trained a pre-trained YOLOv8 model on a relevant dataset, utilizing it as the outer loop. The trained weights from this model were then transferred to a simpler YOLOv8 model (the inner loop), which was subsequently fine-tuned on task-specific data, leveraging the insights gained from the outer loop. This approach, grounded in meta-learning principles that optimize the learning process itself, resulted in superior performance in extended-range detection compared to SSD, DeteRand, and Faster RCNN. By incorporating these principles, we enhanced the model's adaptability to various tasks and environments, significantly improving its overall robustness.

The meta-learner, or inner loop, plays a crucial role in this process by refining previously generalized weights to better align them with specific task requirements. As the model processes task-specific data, it updates its weights to achieve the specificity and accuracy needed for successful object detection. This refinement process involves second-order computations to effectively learn across tasks from the same distribution. The system employs a two-stage optimization strategy: the first stage focuses on learning task similarities (outer loop), while the second stage emphasizes task-specific learning, thereby enhancing overall proficiency. Detailed updates on the weight and loss functions in the inner loop are provided in the following section.

To train a foundational model, we employed meta-learning strategies that leverage task similarity. Task similarity describes the extent to which different tasks share common characteristics or patterns. For instance, when preparing a model for traffic light detection, we pre-trained it using images of car turn signals and brake lights. These images share common color features with traffic lights and are more readily available. Meta-learning, often referred to as "learning to learn," trains the base model on a series of tasks, enabling rapid adaptation to new tasks that are similar in nature with minimal additional data or training \cite{dong2023boosting}. This approach is beneficial in practical settings where compiling and annotating extensive datasets for every object of interest is impractical.

##Meta Learner

![Metalearner](https://github.com/user-attachments/assets/e7a31999-e7b8-4e08-bdd0-c78f5287269a)

Fig 1. The base model for traffic light color is initialized with random weights θ and trained on
similar tasks to prime it for final task performance, with its learning trajectory guided by a predefined
loss function and iterative weight updates to θ′
. A meta-learner further refines these weights to
value Θ′, aligning them with the specific task’s requirements, until the model is fine-tuned with
task-specific data, resulting in a tailored set of weights (Θ′i ) optimized for each class detection.

## Data Preparation
In the absence of specific public datasets tailored to the advanced requirements of our object detection model, which include high-quality labeled images covering various lighting conditions, angles, and weather scenarios, we construct a bespoke fusion traffic dataset using different public datasets like Kitty: {https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d},  (accessed on 10-07-2023), 
Kaggle: {https://www.kaggle.com/datasets/wjybuqi/traffic-light-detection-dataset?resource=download},  (accessed on 10-07-2023), 
CARLA: {https://www.kaggle.com/datasets/sachsene/carla-traffic-lights-images},  (accessed on 10-07-2023), 
LISA: {https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset/code}, (accessed on 10-07-2023), 
CityScapes: {https://www.cityscapes-dataset.com/login/}}, (accessed on 1-12-2023), 
Eurocity :{https://eurocity-dataset.tudelft.nl/eval/user/login?_next=/eval/downloads/detection}
```
pip install ultralytics
```
