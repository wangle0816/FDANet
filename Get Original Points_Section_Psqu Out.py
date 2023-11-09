import re
import csv
import numpy as np
from numpy import arctan, arcsin
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from ThreePoint2Circle import points2circle
from PointRotateXOY import RotXOY
from TheoPointCal import initial_Theoretical_Point


def z_value(w,d1,t_Stable,t_Step2,t_Total,R,dy,rr=72.5):
    z_Point=[]
    for i in range(9):
        z=rr+w*rr*((t_Step2+1.0)+i*(t_Stable+1))-R[i]*(arctan((R[i]+rr-dy[i])/d1)-arcsin((R[i]-170.0)/R[i]))#计算公式
        z_Point.append(z)
    z_Last=rr+w*rr*t_Total-R[9]*(arctan((R[9]+rr-dy[9])/d1)-arcsin((R[9]-170.0)/R[9]))
    z_Point.append(z_Last)
    return(z_Point)


def process_Para_Read(filePath,FileNameinx):
    i=FileNameinx
    fr = open(filePath, 'r')
    reader = csv.reader(fr)
    paralist = list(reader)
    r_out = float(paralist[i][1])/2.
    w=float(paralist[i][3])
    d1=float(paralist[i][5])
    dy =[float(c) for c in paralist[i][6:16]]
    R= [float(c) for c in paralist[i][36:46]]
    TA=[float(c) for c in paralist[i][46:56]]
    return r_out,w,d1,dy,R,TA

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,m
'''
num = 2048:[FileNameinx,length] [1,118.9]   [2,151.3]    [3,83.7]    [4,90]      [5,104.1 ]  [7,208 ]    [8,180]     [10,196]
                     [11,283]    [12,301.5]   [16,355.5]  [19,430]    [21,237.6]  [23,438]    [24,395]    [26,253]
                     [27 429]    [29 169.5]   [30 452]    [35 359.5]  [36 271.3]  [38 514.5]  [39 478.5]  [40 541]
                     [41 451.5]  [42 200]     [43 164.7]  [44 478]    [45 513]    [46 287.5]   [47 537]   [48 404] [49 294]   
                       
                     
num=8192:[FileNameinx,length] [index,length*4]]           
'''




filePath=r"F:\0_Paper\Point cloud complement\Sample2.csv"
FileNameinx=11
length=283*4
r_out,w,d1,dy,R,TA=process_Para_Read(filePath,FileNameinx)
print('rD',r_out,'\n','R',R)
#r_out=27.493 /2.
#w=2.050
#d1=160.528
#dy=[18.285 ,	19.501 ,	20.735 ,	21.405,	22.906 ,	25.032, 	29.106,	30.999 ,	35.856, 	49.148 ]     #每个样本的LR提升距离
#R= [567.528 ,	528.774 ,	494.197 ,	477.116 ,	442.550, 	400.847 ,	338.381 ,	315.128 ,	267.140 ,	186.784 ]     #样本弯曲半径
Part_Num=8

t_Stable=500/75/w
t_Step2=1.5*t_Stable
t_Total=10*(1+t_Stable)+0.5*t_Stable

z_End=z_value(w,d1,t_Stable,t_Step2,t_Total,R,dy,rr=72.5)
z_End=np.array(z_End)
print('z_End',z_End)
z_Begin=z_End-length
print('z_Begin',z_Begin)


FileName=['inpJob-irr'+str(FileNameinx)+'-0.txt','inpJob-irr'+str(FileNameinx)+'-1.txt']
print(FileName)
#直管节点提取
inFile_txt = open(FileName[0], 'r')
inFile_read=inFile_txt.read()

begin=inFile_read.find('*Node')
end=inFile_read.rfind('*Element')
inFile_read=inFile_read[begin+6:end-1]
#print((inFile_read))
#File_num=re.findall(r'-?\d+\.?\d*e?-?\d*?', inFile_read)
File_num=re.split('[,\n]',inFile_read)
File_num= [x.strip() for x in File_num if x.strip() != '']
#print(File_num[:100])
line_index=int(len(File_num)/4)#计算行数
data_list=[]
for i in range(line_index):
    for j in range(3):
        File_num[4*i+j+1]=float(File_num[4*i+j+1])#字符转换
    File_num[4 * i ]=int(File_num[4*i])
    data_list.append(File_num[4*i:4*i+4])#每4个一行
data_0=np.array(data_list)#转化为数组
#print('data_0','\n',data_0)

#弯管节点提取
inFile_txt = open(FileName[1], 'r')
inFile_read=inFile_txt.read()
begin=inFile_read.find('*Node')
end=inFile_read.rfind('*Element')
inFile_read=inFile_read[begin+6:end-1]
#File_num=re.findall(r'-?\d+\.?\d*e?-?\d*?', inFile_read)
File_num=re.split('[,\n]',inFile_read)
File_num= [x.strip() for x in File_num if x.strip() != '']
line_index=int(len(File_num)/4)
data_list=[]
for i in range(line_index):
    for j in range(3):
        File_num[4*i+j+1]=float(File_num[4*i+j+1])
    File_num[4 * i] = int(File_num[4 * i])
    data_list.append(File_num[4*i:4*i+4])
data_1=np.array(data_list)

#print('data_1',data_1)


z_data0=data_0[:,3]

#提取稳定成形管段范围内所对应的直管段行号
row_all=[]

for i in range(Part_Num):
    row=[r for r in range(len(data_0[:,0])) if (z_data0[r]>=z_Begin[i] and z_data0[r]<=z_End[i])]
    row_all.append(list(row))
p_all=[]
row_out=[]

for i in range(Part_Num):
    row_index=[]
    for j in range(len(row_all[i])):
        r_2=np.sqrt(np.square(data_0[row_all[i][j],1])+np.square(data_0[row_all[i][j],2]))
        if r_2<r_out-0.1:
            row_index.append(j)
            #print(j)
    row_out.append(np.delete(row_all[i],row_index))
print('row_out',row_out)
for i in range(Part_Num):
    p_all.append(data_1[row_out[i], :])
    straP=data_0[row_out[i], :]
    print('len(p_all[i])',len(p_all[i]))

#找出每段管子的首圈和末圈
zi_Max_row=[]
zi_Min_row=[]
for i in range(Part_Num):
    fnMax=data_0[row_out[i], :]
    zi_Max_inx=np.where(fnMax[:,3]==max(fnMax[:,3]))
    zi_Max_row.append(fnMax[zi_Max_inx[0],0]-1)
    fnMin=data_0[row_out[i], :]
    zi_Min_inx=np.where(fnMin[:,3]==min(fnMin[:,3]))
    zi_Min_row.append(fnMin[zi_Min_inx[0],0])
zi_Max_row=np.array(zi_Max_row)
zi_Min_row=np.array(zi_Min_row)

p_all_XOY=[]
p_all_XOYi_proce=[]
CrossPoint=[]
for i in range(Part_Num):
    Endthr_Row=(np.random.choice(zi_Max_row[i,:], 3, replace=False)).astype(int)
    Endthr_P=data_1[Endthr_Row,1:]
    Sta_Row=zi_Min_row[i,:].astype(int)
    Sta_P=data_1[Sta_Row,1:]
    cen_StaPi=Sta_P.mean(0)
    p_all_XOYi=RotXOY(cen_StaPi,Endthr_P,p_all[i][:,1:])
    p_all_XOYi,Psqu_max=pc_normalize(p_all_XOYi)
    #print('Psqu_max',Psqu_max)
    p_all_XOYi_index=np.append(np.expand_dims(p_all[i][:, 0], axis=-1),p_all_XOYi,axis=1)
    p_all_XOYi_procei=[]
    procei = [R[i], TA[i], r_out, Psqu_max]
    CrossPointi =initial_Theoretical_Point(R[i], TA[i], r_out, Psqu_max)
    CrossPoint.append(CrossPointi)
    for j in range(len(p_all_XOYi_index)):
        p_all_XOYi_procei.append(list(np.append(p_all_XOYi_index[j],procei)))
    p_all_XOYi_proce.append(p_all_XOYi_procei)
    print('p_all_XOYi_procei',len(p_all_XOYi_procei))
CrossPoint=np.array(CrossPoint)
print('CrossPoint',CrossPoint.shape)
print('p_all_XOYi_proce',type(p_all_XOYi_proce))
p_all_XOY=p_all_XOYi_proce
print('p_all_XOY',type(p_all_XOY))


'''
center_P=[]
for i in range(Part_Num):
    center_P.append(np.mean(p_all[i], axis=0))
center_P=np.array(center_P)
#center_P[:,0]=0

p_all_moved=[]
for i in range(Part_Num):
    p_all_moved.append(p_all[i]-center_P[i])
'''
#p_all_moved=np.array(p_all_moved)
#print((p_all_moved[0]))

#print(p_all_moved)

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6), facecolor='w')
cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
for i in range(Part_Num):
    #x=np.delete(p_all_moved[i], 0, axis=1)
    x=np.array(p_all_XOY[i])
    s=CrossPoint[i]
    #print(x)
    if i ==0:
        print('x',x[0])
    #print('x',x)
    ax1 = fig.add_subplot(241+i, projection='3d')
    ax1.scatter(x[:, 1], x[:, 2],x[:, 3],alpha=0.3,c="#FF0000",s=10)
    plt.title('管段'+str(i+1))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.view_init( azim=0,elev=90)
    plt.grid(True)
    np.savetxt('D:\Dataset\singletube\8192\irr'+str(FileNameinx)+'-Original_Out' + str(i) + '.pts', x, fmt="%f %f %f %f %f %f %f %f")
    np.savetxt('D:\Dataset\singletube\8192\irr' + str(FileNameinx) + '-Original_Out' + str(i) +'_Section'+'.pts', s, fmt="%f %f %f")

plt.show()

