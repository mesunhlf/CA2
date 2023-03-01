### Overview

The code is repository for ["Cyclical Adversarial Attack Pierces Black-box Deep Neural Networks"](https://www.sciencedirect.com/science/article/pii/S0031320322003120) (Pattern Recognition).

### Prerequisites

python **3.6**  
tensorflow **1.14**  

### Pipeline 
<img src="/figure/overview.png" width = "700" height = "350" align=center/>

### Dataset
We select 1000 images from ImageNet validation dataset. 
All tested images can be correctly classified by vanilla models, and thereby treated as the standard benchmark to be collected in the SACP2019 adversarial competition (Tianchi Security AI Challenger Program Competition). 

The download link is [here](https://drive.google.com/file/d/1oC1ITY8SnQeeC4JxAnGh5HNItasdTQnx/view?usp=sharing). 

### Run the Code  
The standalone CA<sup>2</sup>: `CA2.py`.  
The strongest combination CA<sup>2</sup>-SIM*: `CA2-SIM.py`.

### Experimental Results
We attack four normally trained models to generate adversarial examples, and test the transferability against ten defense models.

<b>Standalone Experiment</b>
<img src="/figure/exp1.png" width = "700" height = "500" align=center/>

<b>Ensemble Experiment</b>
<img src="/figure/exp3.png" width = "700" height = "300" align=center/>

### Citation
If you find this project is useful for your research, please consider citing:

	@article{huang2022cyclical,
	  title={Cyclical adversarial attack pierces black-box deep neural networks},
	  author={Huang, Lifeng and Wei, Shuxin and Gao, Chengying and Liu, Ning},
	  journal={Pattern Recognition},
	  volume={131},
	  pages={108831},
	  year={2022},
	  publisher={Elsevier}
	}
