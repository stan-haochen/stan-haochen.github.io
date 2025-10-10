---
layout: distill
title: A Foundation Model Believer's Take on Embodied AI Research
description: Embodied AI research questions which I considered important, with introduction to some of our recent work.
tags: embodied_ai
giscus_comments: false
date: 2025-10-10
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Hao Chen
    url: "https://stan-haochen.github.io"
    affiliations:
      name: CAG&CG, Zhejiang University

# bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: 1. The Division of Labor: A Tale of Two Systems
  - name: 2. The Quest for a Generalizable Action Expert
  - name: 3. The Hunger for Data and a Thirst for Complexity
  - name: 4. So, Are Reasoning Models a Solved Problem?
  
# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

> "I have been impressed with the urgency of doing. Knowing is not enough; we must apply. Being willing is not enough; we must do."
> 
> —— Leonardo da Vinci

There's no doubt we're in the early innings of embodied intelligence. And I'm an optimist: I believe we might see general-purpose agents leap from the virtual world into our physical one within the next two years. As a researcher whose intuition for data, models, and optimization strategies was largely forged in the COCO-scale era of computer vision, I'd like to share a few personal thoughts on the path forward.

## 1. The Division of Labor: A Tale of Two Systems

The first lesson my experience in computer vision taught me is this: for the foreseeable future, a Vision-Language-Action (VLA) model—one that takes language commands and observations to output actions—should be a **dual-system architecture**. This means a "Reasoning" system (the Vision-Language part) working in tandem with an "Acting" system.

Here’s why this division makes perfect sense:

1.  **Running Efficiency:** Real-time, fine-grained motor control for complex tasks in dynamic environments is non-negotiable. A massive Vision-Language Model (VLM) can't, and frankly, shouldn't, run at a 100Hz frequency. A dual-system setup allows the reasoning and acting modules to operate at their own optimal cadences.
2.  **Generalization and Transfer Learning:** Let's be real, the diversity of embodied manipulation data is, to put it mildly, paltry compared to the vast datasets that forged our VLMs. Training a monolithic model directly on this sparse data would inevitably lead to a catastrophic degradation of the VLM's incredible pre-trained abilities.
3.  **Training Efficiency:** The data efficiency of Reinforcement Learning (RL) on complex manipulation tasks can be notoriously low, often requiring frequent policy adjustments. Hitching the massive VL wagon to the RL horse for end-to-end training would make the entire process unacceptably sluggish.

So, how should these two systems collaborate on embodied tasks? This, I believe, is one of the central questions for the field. To put it another way: how do we give our digital minds the right eyes and hands?

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/foundation_embodied/knowing_doing.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

For simpler tasks, a powerful VLM might suffice on its own. But as complexity ramps up, the division of labor becomes crucial. The boundary between these two systems will be dynamic, shifting as the capabilities of each module evolve. On the most challenging tasks, the Reasoning model may fail to provide perfectly reliable guidance, and the Acting model may struggle to compensate for these upstream errors.

This chasm between what the reasoning model *knows* and what the acting model can *do* is what I call the **knowing-doing gap**. I believe the onus is on the Acting module to bridge this gap, primarily because the reasoning capabilities of foundation models are advancing at a breathtaking pace that shows no signs of slowing down.

This paradigm raises a clear and meaningful question: What information *should* the Reasoning module provide to the Acting module? In our recent work, we proposed **notVLA (Narrative Outline of Trajectories)** to explore this. We argue that the guidance from the VLM should be:

1.  **Text-based:** Because text is the *lingua franca* of zero-shot generalization, enabling adaptive training.
2.  **Sparse:** For the sake of efficiency. Five well-chosen keypoints are plenty to define a smooth trajectory.
3.  **3D:** Because, well, today's VLMs are just that good.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/foundation_embodied/notvla.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

During training, we employ a kinematics-based keyframe selection method to provide sparse supervision for our VLA model, using only the most critical end-effector poses. At inference time, the procedure unfolds in two stages. First, the model generates a planned trajectory via anchor-based depth inference—a two-step, text-guided prediction. Subsequently, this trajectory is processed by a spline-based action detokenizer to produce a sequence of smooth, executable actions. Our experimental results show that this approach yields excellent performance in both general-purpose accuracy and generalization.

## 2. The Quest for a Generalizable Action Model

Another beauty of the dual-system architecture is that it lets us have our cake and eat it too. We get the phenomenal expressive power of VLMs *and* the training efficiency of a specialized action model. With generalized trajectory guidance from the VLM, we can tackle mixed-task training at a scale that would be impossible otherwise, as the VLM elegantly resolves semantic ambiguities. This clears the path for a more focused investigation into scaling up the action models themselves. Following this line of thought, we successfully trained a **generalizable action expert** that achieves zero-shot generalization across different datasets and tasks.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/foundation_embodied/gae.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Yet, there's a vast, exciting frontier to explore here, from the scaling efficiency of different architectures to hybrid imitation-and-reinforcement learning and generalization to more complex tasks and robot bodies. We hope this framework can serve as a catalyst for expanding the data and capabilities in the manipulation domain.

## 3. The Hunger for Data and a Thirst for Complexity

> "Data! Data! Data! I can't make bricks without clay."
> 
> —— Arthur Conan Doyle, The Adventure of the Copper Beeches - a Sherlock Holmes Short Story

If we take a step back, the current bottleneck isn't in perception or planning—it's in *action*. And if the history of language and vision models has taught us anything, it's that learning-based approaches are our best bet for cracking tough problems.

To put the task complexity in perspective: if Gym's [MuJoCo Tasks](https://gymnasium.farama.org/environments/mujoco/) are the MNIST of motion control, then [LIBERO](https://libero-project.github.io/datasets) is perhaps the CIFAR-10. Simply adding more pick-and-place tasks just gets us to CIFAR-100. What we truly need are massive rollouts of complex tasks—like dexterous hand manipulation, precise assembly, and deformable object handling—to fuel the next stage of progress.

So, how much data is enough? The key isn't raw trajectory length, but *diversity*. But instead of asking how much we need, let's consider what we already have. The internet is a treasure trove of human videos demonstrating intricate manipulations, tool use, and assembly. Given the blistering pace of 3D vision and generative modeling this year, creating digital twins from these videos is becoming increasingly feasible. *Before we debate the irreplaceable quality of real-world data collection, we should at least leverage this massive, free dataset.* Once we do, the "sim2real gap" might just become a myth of a bygone era. This, in my view, is the strongest argument for humanoid robots: the data is overwhelmingly human.

Of course, another indispensable route is autonomous exploration in simulation. This is the logical path for tasks where real-world data is scarce and for learning skills that surpass human ability. While we're still in the early stages, its value is already apparent in long-horizon tasks where navigation, manipulation, and whole-body control must work in concert. Imagine needing to find the right vantage point to grasp a complex part or using the environment for leverage to open a heavy door. For scenarios like these, end-to-end RL holds the promise of unlocking emergent capabilities from our foundation models.

To this end, we built **[Odyssey](https://kaijwang.github.io/odyssey.github.io/)**, Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks. It presents the first comprehensive benchmark for long-horizon mobile manipulation, evaluating diverse indoor and outdoor scenarios. We currently provide a hierarchical control architecture guided by an LLM and vision models, and we are excited to expand it with richer tasks, offering the community a robust RL playground to ignite compositional generalization.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/foundation_embodied/odyssey.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

> Our pipeline spans the entire process of a long-horizon task, including multi-modal semantic perception, map-aware global planning, geometry-constrained action grounding, and step-wise execution by a reinforced-learned low-level whole-body policy.

## 4. So, Are Reasoning Models a Solved Problem?

It's a familiar story in AI: as scale increases, handcrafted design gracefully bows out to raw expressive power. The relentless simplification of network architectures is undeniably the right path. Still, it leaves someone like me, who spent the better part of a decade pondering clever network designs (with the recent [DUSt3R](https://arxiv.org/abs/2312.14132) being a rare, delightful exception), a little wistful.

So, can existing large-model architectures handle the ultimate reason problems for embodied agents? The unfortunate, or perhaps fortunate, answer is **yes**—or at least, it's a matter of when, not if.

However, for tasks grounded in the physical world, I argue there's still ample room for specialized optimization. A reasoning model must orchestrate a symphony of different sensors, 3D reconstruction models, object- or scene-level generative models, action policy models, and physics simulators. Simply outputting a string of coordinates is an insufficient medium for this complex communication. We need an **latent representation** that serves as a bridge between reasoning and downstream modules, one that can be refined through end-to-end training.

This brings us to the concept of **World Models**. A clarification I find myself making constantly this year is that a video generation model controlled by camera movements **≠** a world model. A true world model encodes state transitions, not just pixel changes:
$$
s_{t+1} = \mathcal M(s_t, a_t)
$$
Here, the action $a_t$ can be far more abstract than a camera pan (e.g., a thought flashing through your mind), and the state $s_t$ can be far more compact than a 30fps HD video (e.g., a 1024-dimensional vector every 5 seconds). This compactness offers two huge advantages. First, it allows the reasoning model to "think with images" without getting bogged down in the costly business of video generation. Second, for embodied tasks, the future state provides a natural and powerful bridge for end-to-end training.

We recently took a stab at this with our work on [StaMo](https://aim-uofa.github.io/StaMo/). We propose an unsupervised approach that learns a highly compressed two-token state representation for general embodied tasks. Our representation is efficient, interpretable, and integrates seamlessly into existing VLA-based models. More importantly, we find that the difference between these tokens, obtained via latent interpolation, naturally serves as a highly effective latent action, which can be further decoded into executable robot actions. This emergent capability reveals that our representation captures structured dynamics without explicit supervision. We named our method **StaMo** for its ability to learn generalizable robotic **Mo**tion from a compact **Sta**te representation, which is encoded from static images, challenging the prevalent dependence on complex architectures and video data for learning latent actions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/foundation_embodied/odyssey.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

> Our method efficiently compresses and encodes robotic visual representations, enabling the learning of a compact state representation. Motion naturally emerges as the difference between these states in the highly compressed token space. This approach facilitates efficient world modeling and demonstrates strong generalization, with the potential to scale up with more data. Please see [our paper](https://arxiv.org/abs/2510.05057) for more details.