import glob
import cv2
fileNum = 66

print("starting")
pathname = "/home/kirsty/Desktop/fyp/fyp/Test Cases 2018/test"+str(fileNum)+"/*.png" # refine path name!!!
filenames = sorted(glob.glob(pathname))
for i in range (0, len(filenames)):
    current = filenames[i]
    image = cv2.imread(current, cv2.IMREAD_GRAYSCALE)
    if(i < 10):
        imagename = str(fileNum)+"_0"+str(i)+".png"
    else:
	    imagename = str(fileNum)+"_"+str(i)+".png"
    newPathname = "/home/kirsty/Desktop/fyp/fyp/RENAMED/test"+str(fileNum)+"/"+imagename
    print(newPathname)
    cv2.imwrite(newPathname, image)

    newPathname2 = "/home/kirsty/Desktop/fyp/fyp/RENAMED_MERGED/"+imagename
    cv2.imwrite(newPathname2, image)

print("done")
