import os
import cv2
f = open('lefts.txt', 'r')

for i, l in enumerate(f):
    l = l.replace('\n', '').split(' ')
    try : 
       os.mkdir('./new/' + str(2*i + 1))
    except:
       pass
    for j, n in enumerate(l):
       k = int(n) - i*10
       img = cv2.imread('./'+ '{0:0=3d}'.format(i+1) + '/' + '{0:0=2d}'.format(k) + '.png')
       cv2.imwrite('./new/' + str(2*i + 1) + '/' + str(j+1) + '.png', img)
    

f1 = open('rights.txt', 'r')

for i, l in enumerate(f1):
    l = l.replace('\n', '').split(' ')
    try : 
       os.mkdir('./new/' + str(2*i + 2))
    except:
       pass
    for j, n in enumerate(l):
       k = int(n) - i*10
       img = cv2.imread('./'+ '{0:0=3d}'.format(i+1) + '/' + '{0:0=2d}'.format(k) + '.png')
       cv2.imwrite('./new/' + str(2*i + 2) + '/' + str(j+1) + '.png', img)
       #print('./'+ '{0:0=3d}'.format(i+1)+ '/' + '{0:0=2d}'.format(k) + '.png')
       #print('./new/' + str(2*i + 2) + '/' + str(j+1) + '.png')
