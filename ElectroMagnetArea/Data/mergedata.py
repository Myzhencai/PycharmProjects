import numpy as np

def merge3():
    area0 = np.loadtxt("area0.txt")
    area1 = np.loadtxt("area1.txt")
    area2 = np.loadtxt("area2.txt")

    mergeddata = np.r_[area0,area1]
    mergeddata = np.r_[mergeddata,area2]
    print(mergeddata)

    np.savetxt("megedData.txt",mergeddata)


def merge4(filePath):
    area0 = np.loadtxt(filePath+"area0.txt")
    area1 = np.loadtxt(filePath+"area1.txt")
    area2 = np.loadtxt(filePath+"area2.txt")
    area3 = np.loadtxt(filePath+"area3.txt")

    mergeddata = np.r_[area0, area1]
    mergeddata = np.r_[mergeddata, area2]
    mergeddata = np.r_[mergeddata, area3]
    print(mergeddata)

    np.savetxt(filePath+"megedData.txt", mergeddata)


if __name__ =="__main__":
    # filePath ="/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/"
    filePath ="/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata7new/"
    merge4(filePath)