import numpy as np



diff1 = open("diff1.txt", "w+")
diff2 = open("diff2.txt", "w+")
epoch1 = open("epoch1.txt", "w+")
epoch2 = open("epoch2.txt", "w+")
minmse1 = open("minmse1.txt", "w+")
minmse2 = open("minmse2.txt", "w+")
maxmse1 = open("maxmse1.txt", "w+")
maxmse2 = open("maxmse2.txt", "w+")

for i in range(1, 221):
    fname = str(i)+"/1_Difference.txt"
    diff_1hl = np.loadtxt(fname, ndmin=2)

    
    diff1.write(str(diff_1hl[-1]).replace('[','').replace(']','')+"\n")
    epoch1.write(str(len(diff_1hl))+"\n")

    fname2 = str(i)+"/2_Difference.txt"
    diff_2hl = np.loadtxt(fname2, ndmin=2)

    diff2.write(str(diff_2hl[-1]).replace('[','').replace(']','')+"\n")
    epoch2.write(str(len(diff_2hl))+"\n")

    fname3 = str(i)+"/1_MSE.txt"
    mse1hl = np.loadtxt(fname3, ndmin=2)
    fname4 = str(i)+"/2_MSE.txt"
    mse2hl = np.loadtxt(fname4, ndmin=2)

    minmse1.write(str(mse1hl[-1]).replace('[','').replace(']','')+"\n")
    maxmse1.write(str(mse1hl[0]).replace('[','').replace(']','')+"\n")

    minmse2.write(str(mse2hl[-1]).replace('[','').replace(']','')+"\n")
    maxmse2.write(str(mse2hl[0]).replace('[','').replace(']','')+"\n")

    print(str(i)+":\t"+str(len(diff_2hl))+" | "+str(diff_2hl[-1]).replace('[','').replace(']',''))
    

