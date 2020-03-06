import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
from copy import deepcopy
#####---------------------------------------------------------------
K = np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02],
              [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02],
              [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])
Kt = np.transpose(K)
Kinv=np.linalg.inv(K)
N=np.array([[2/375,0,-1],[0,2/1242,-1],[0,0,1]])
# N=np.array([[0.012,0,-9],[0,0.012,-2.5],[0,0,1]])
C=np.eye(4)
#####----------------------------------------------------------
def F_esti(pt1,pt2):
  x=[]
  for i in range(8):
    x=np.append(x,np.kron(pt1[i,:],pt2[i,:]))
#   print(x.reshape(8,9))
  x=x.reshape(8,9)
  u,d,vt=np.linalg.svd(x)
  F=vt[8,:]
  F=F.reshape(3,3)
#   print("F_old:")
#   print(F)
  u2,d2,vt2=np.linalg.svd(F)
  diag=np.array([d2[0],d2[1],0])
  d3=np.diag(diag)
  F=u2@d3@vt2
  return F
#   print("F_new:")
#   print(F)

def ransac(pt1,pt2): 
#   pt1=np.hstack((pts1,np.ones((pts1.shape[0],1))))
#   pt2=np.hstack((pts2,np.ones((pts2.shape[0],1))))
#   print(pt1)
#   print(pt2)

  thresh=0.05
  max_inliers=0
  iterations=1500
  for i in range(iterations):
    ind=np.random.randint(low=0,high=pt1.shape[0],size=8)
    ept1=pt1[ind]
    ept2=pt2[ind]
    F_est=F_esti(ept1,ept2)
    error=np.diag(pt1@F_est@(pt2.T))
#     print(error)
    inliers=(error>thresh).sum()
#     print(inliers)
    if inliers>max_inliers:
      max_inliers=inliers
      F=F_est
#       print(max_inliers)
  return (N.T)@F@N

###--------------------------------------------

L=np.array([0,0,0,1])
file=open('my-result.txt','w')
file.write("1.000000e+00 -1.822835e-10 5.241111e-10 -5.551115e-17 -1.822835e-10 9.999999e-01 -5.072855e-10 -3.330669e-16 5.241111e-10 -5.072855e-10 9.999999e-01 2.220446e-16")
file.write("\n")

def readfile(fil):
    truc=np.array(list(map(float,fil.readline().split()))).reshape(3,4)
    truc=np.vstack((truc,L))
    return truc[0:3,3]

g_truth=open("ground-truth.txt","r")
truth_old=readfile(g_truth)
img1 = cv2.imread('images/000000.png')
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
pts1 = np.array([x.pt for x in kp1],dtype=np.float32)

for i in range(1,801):
  img2=cv2.imread('images/' '000' + str(i).zfill(3)+'.png')
  kp2,des2 = orb.detectAndCompute(img2,None)
  pts2 = np.array([x.pt for x in kp1],dtype=np.float32)
  
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  dmatches = sorted(matches, key = lambda x:x.distance)
  pts1  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
  pts2  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
  pts1=pts1.reshape(pts1.shape[0],2)
  pts2=pts2.reshape(pts2.shape[0],2)
#   print(pts1[0:5,:])
#   print(pts2[0:5,:])
  norm_pts1=np.hstack((pts1,np.ones((pts1.shape[0],1))))
  norm_pts2=np.hstack((pts2,np.ones((pts2.shape[0],1))))
  norm_pts1=N@(norm_pts1.T)
  norm_pts2=N@(norm_pts2.T)
  F=ransac(norm_pts1.T,norm_pts2.T)
  E=Kt@F@K
#   E=cv2.findEssentialMat(pts1,pts2,K,method=cv2.RANSAC,prob=0.999)
#   print(E)
  _,R,t,_=cv2.recoverPose(E,pts2,pts1,K)
  truth_new=readfile(g_truth)
#   print(truth_new)
  t=t*np.linalg.norm(truth_new-truth_old)
  truth_old=truth_new
  T=np.hstack((R,t))
  T=np.vstack((T,L))
  C=C@T  
#   C=C/C[3,3]
  S=C[0:3,:]
#   S=S/S[3,3]
  S=S.flatten()
  print(S)
  
  for i in range(12):
    if i<11:
      file.write(str(S[i]))
      file.write(" ")
    else:
      file.write(str(S[i]))
      file.write("\n")  
  ####swapping of key-pts,descriptors and images####
  img1=img2  
  kp1=kp2
  des1=des2
  pts1=pts2
