# SAIPP-Net: A Sampling-Assisted Indoor Pathloss Prediction Method for Wireless Communication Systems

This repository contains the implementation of **SAIPP-Net**, a novel sampling-assisted indoor pathloss prediction method designed for the **MLSP 2025 The Sampling-Assisted Pathloss Radio Map Prediction Data Competition**. This competition was launched to promote deep learning methods for indoor pathloss radio map prediction with ground truth pathloss samples, with a particular focus on the role of sampling strategies. 

## Competition Tasks

The competition includes two supervised tasks. Task 1 is set to evaluate prediction performance for a fixed set of random samples at two sparsity levels (0.02\% and 0.5\%), while Task 2 allows participants to jointly optimize sampling locations and pathloss prediction subject to the same sampling constraints. The overarching goal of this competition is to assess how effectively different methods can exploit sparse measurements and choose sampling locations to improve prediction accuracy, while ensuring computational efficiency across diverse indoor environments. 

For competition details, visit the [official competition page](https://sapradiomapchallenge.github.io/index.html).

## Installation

To install the necessary dependencies, run the following command in your virtual environment:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset can be downloaded from the following link: [Dataset URL](https://ieee-dataport.org/documents/indoor-radio-map-dataset)

### Training
Training code in the 'training.py'.

### Test code for sampling rate 0.5%
Test code for sampling rate 0.5% in the 'test_for_rate0.5.py'.

### Fine-tune and test code for sampling rate 0.02%
Fine-tune and test code for sampling rate 0.02% in the 'finetune_and_test_for_rate0.02.py'.

### Trained Models
Download the pretrained model from [this link](https://huggingface.com).

## Results

**SAIPP-Net** achieved the best RMSE esult among all participating teams: 
- **Task 1 (0.02%)**: 5.99
- **Task 1 (0.5%)**: 3.32
- **Task 2 (0.02%)**: 6.08
- **Task 2 (0.5%)**: 3.28
- **Weighted Score**: 4.67

Check the competition [leaderboard](https://sapradiomapchallenge.github.io/results.html) for detailed results. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{Feng2025SAIPPNet,
  title={SAIPP-Net: A Sampling-Assisted Indoor Pathloss Prediction Method for Wireless Communication Systems},
  author={Feng, Bin and Zheng, Meng and Liang, Wei and Zhang, Lei},
  booktitle={Proc. IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2025},
  month={August}
}
```

## Contact

For any questions, please contact [fengbin3@sia.cn](mailto:fengbin3@sia.cn).
