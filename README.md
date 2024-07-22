# attribution_analysis
## 数据集说明
ESAD数据集整体分布如下。
|             | other       | outside     | self        | All         |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Anxiety     | 73          | 275         | 50          | 398         |
| Anger       | 98          | 78          | 38          | 214         |
| Weariness   | 125         | 173         | 20          | 318         |
| All         | 296         | 526         | 108         | 930         |

根据五折交叉验证原理得到的测试集分布如下，其中前三折为or前的，后两折为or后的。
|             | other       | outside     | self        | All         |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Anxiety     | 15or14      | 55          | 10          | 80or79      |
| Anger       | 20or19      | 16or15      | 8or7        | 44or41      |
| Weariness   | 25          | 35or34      | 4           | 64or63      |
| All         | 60or58      | 106or104    | 22or21      | 188or183    |



## test.py
```
conda create -n ESAD python=3.9.16
conda activate ESAD
pip install tqdm tensorboardX numpy==1.24 scipy scikit-learn pandas
conda install opencv 
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

修改test.py中的root_dir、model_folder、fold_folder

cd ../medium/attribution_analysis
python test.py
```
需要显存9G左右，numpy版本高了(2.x.x)会报错  
修改test.py中的root_dir、model_folder、fold_folder
