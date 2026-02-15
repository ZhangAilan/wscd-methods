# 弱监督变化检测方法复现

## 环境配置
```
conda create --name wscd python=3.8 -y
conda activate wscd
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130   ##根据自己的cuda版本选择
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## 数据集
- [WHU-CD-256](https://aistudio.baidu.com/datasetdetail/251669)
- [LEVIR-CD-256](https://www.kaggle.com/datasets/keykeylv/levir-cd-256)


## [TransWCD](https://github.com/zhenghuizhao/TransWCD)
预训练权重```TransWCD/transwcd/pretrained```  [mit_b1.pth](https://drive.google.com/file/d/11Tf0NCm1ry_vmqqoCcKTdtmDEAA_PhTQ/view?usp=sharing)

修改```TransWCD/transwcd/configs/WHU.yaml```中```root_dir```为数据集路径

结果保存在```TransWCD/transwcd/work_dir_WHU/```


## [ACWCD](https://github.com/WenhaoLiu03/ACWCD)

预训练权重```ACWCD/pretrained```  [mit_b1.pth](https://drive.google.com/file/d/11Tf0NCm1ry_vmqqoCcKTdtmDEAA_PhTQ/view?usp=sharing)

WHU-CD数据：```ACWCD\configs\BCD.yaml```，修改```root_dir```为数据集路径