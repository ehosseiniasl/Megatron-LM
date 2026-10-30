# GPQA

### Paper

Title: SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines

Abstract: https://arxiv.org/abs/2502.14739

SuperGPQA is a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.

Homepage: `https://github.com/SuperGPQA/SuperGPQA/`

### Citation

```
@misc{pteam2025supergpqascalingllmevaluation,
      title={SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines}, 
      author={M-A-P Team and Xinrun Du and Yifan Yao and Kaijing Ma and Bingli Wang and Tianyu Zheng and Kang Zhu and Minghao Liu and Yiming Liang and Xiaolong Jin and Zhenlin Wei and Chujie Zheng and Kaixing Deng and Shuyue Guo and Shian Jia and Sichao Jiang and Yiyan Liao and Rui Li and Qinrui Li and Sirun Li and Yizhi Li and Yunwen Li and Dehua Ma and Yuansheng Ni and Haoran Que and Qiyao Wang and Zhoufutu Wen and Siwei Wu and Tianshun Xing and Ming Xu and Zhenzhu Yang and Zekun Moore Wang and Junting Zhou and Yuelin Bai and Xingyuan Bu and Chenglin Cai and Liang Chen and Yifan Chen and Chengtuo Cheng and Tianhao Cheng and Keyi Ding and Siming Huang and Yun Huang and Yaoru Li and Yizhe Li and Zhaoqun Li and Tianhao Liang and Chengdong Lin and Hongquan Lin and Yinghao Ma and Zhongyuan Peng and Zifan Peng and Qige Qi and Shi Qiu and Xingwei Qu and Yizhou Tan and Zili Wang and Chenqing Wang and Hao Wang and Yiya Wang and Yubo Wang and Jiajun Xu and Kexin Yang and Ruibin Yuan and Yuanhao Yue and Tianyang Zhan and Chun Zhang and Jingyang Zhang and Xiyue Zhang and Xingjian Zhang and Yue Zhang and Yongchi Zhao and Xiangyu Zheng and Chenghua Zhong and Yang Gao and Zhoujun Li and Dayiheng Liu and Qian Liu and Tianyu Liu and Shiwen Ni and Junran Peng and Yujia Qin and Wenbo Su and Guoyin Wang and Shi Wang and Jian Yang and Min Yang and Meng Cao and Xiang Yue and Zhaoxiang Zhang and Wangchunshu Zhou and Jiaheng Liu and Qunshu Lin and Wenhao Huang and Ge Zhang},
      year={2025},
      eprint={2502.14739},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14739}, 
}
```

This dataset is gated, so you will have to accept the terms of use at https://huggingface.co/datasets/Idavidrein/gpqa and login via `huggingface-cli login` using your HF Hub token before running this task.

### Groups, Tags, and Tasks

#### Groups

None

#### Tags

* `supergpqa`

#### Tasks

* `super_{main}_zeroshot`
* `supergpqa_{main}_n_shot`
* `supergpqa_{main}_generative_n_shot`
* `supergpqa_{main}_cot_zeroshot`
* `supergpqa_{main}_cot_n_shot`

