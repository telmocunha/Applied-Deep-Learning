dataset= open("data/KITTI/training.txt","w")

nimages= 500

for i in range(int(nimages*0.8)):
    num=str(i)
    dataset.write("KITTI_dataset/Imgs/"+num.zfill(6)+".png\n")

dataset.close()

dataset_test= open("data/KITTI/testing.txt","w")

for i in range(int(nimages*0.2)):
    num= str(i+(int(nimages*0.8)))
    dataset_test.write("KITTI_dataset/Imgs/"+ num.zfill(6) +".png\n")

dataset_test.close()