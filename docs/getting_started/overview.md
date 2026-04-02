# MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics

```{image} /images/MESA.png
:alt: MESA benchmark overview
:width: 650px
```

Welcome to MESA, a dynamic evaluation framework designed to probe language steering and precisely measure language-conditioned policy
generalization. Our framework includes **MESA-Gen**, a pipeline for scalable task and demonstration generation built upon substantial improvements to MimicGen. We also provide **MESA-Bench**, a benchmark designed to probe generalization to unseen spatial configurations, object instances, object categories, and subtask compositions for tabletop manipulation tasks.

This repository enables you to:
1. Design your own tabletop manipulation task suites, either using one of our task skeletons or one that you've defined.
2. Collect a small number of demonstrations for your tasks as needed. 
3. Expand and diversify your dataset using MESA-Gen.
4. Easily run evaluations using your dataset, including new metrics to better measure language-following beyond success rates.

## MESA-Bench

Our benchmark consists of a canonical training set (**MESA-70**) and the 5 following evaluation suites:
- A suite evaluating in-distribution performance
- **MESA-Spatial** - evaluates seen tasks with unseen spatial configurations
- **MESA-Category** - evaluates tasks with unseen object categories
- **MESA-Instance** - evaluates seen tasks with unseen object instances
- **MESA-Compositional** - evaluates unseen composite tasks comprised of in-distribution subtasks

You can download our training set along with our evaluation sets from [Hugging Face](https://huggingface.co/collections/albertwilcox/mesa).

## Citation
This is the official repository for the paper ''MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics'' which was submitted to RSS 2026.

We would also like to thank the authors of MimicLabs, RoboCasa, MimicGen, LIBERO, RoboMimic, and Robosuite.

If you have found MESA useful, please cite in your work:
```bibtex
@article{mesa2026,
  author    = {Albert Wilcox and Frank Chang and Aishani Chakraborty and Nhi Nguyen and Jeremy A. Collins and Vaibhav Saxena and Benjamin Joffe and Siddhath Karamcheti and Animesh Garg},
  title     = {MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics},
  journal   = {arXiv preprint},
  year      = {2026},
}
```
