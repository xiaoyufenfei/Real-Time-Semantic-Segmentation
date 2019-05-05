# RealtimeSS
Collection and testing of Real time semantic segmentation models

- **从某个检查点恢复训练：**

```
python main.py --batch-size=6 --dataset cityscapes --dataset-dir /data/wangyu/Datasets/ --epochs 300 --height 512 --width 1024 --mode train --name ERFnet  --save-dir save/ERFnet --resume
```

- **从头训练：**

```
python main.py --batch-size=6 --dataset cityscapes --dataset-dir /data/wangyu/Datasets/ --epochs 300 --height 512 --width 1024 --mode train --name ERFnet  --weight-decay 1e-4 --save-dir save/ERFnet
```

- **测试IoU:**

```
python main.py --dataset cityscapes --dataset-dir /data/wangyu/Datasets/ --height 512 --width 1024 --mode test --name ERFnet  --save-dir save/ERFnet
```

- **测试单张图片：**

```
python main.py --dataset cityscapes --dataset-dir /data/wangyu/Datasets/ --height 512 --width 1024 --mode single --name ERFnet  --save-dir save/ERFnet
```



### Tips

```
This project results is not good，need more improve ！！
```

