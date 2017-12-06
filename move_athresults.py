#!/Users/dora/Library/Enthought/Canopy_64bit/User/bin/python

import time
import datetime
import os
import re
import shutil

locdir=os.getcwd()

try:
    for dataLine in open( locdir+'/tst/cylindrical/athinput.torus9_hydro_2D', 'r').read().split('\n'):
        
        if 'Nx1' in dataLine:
            nx1 = re.findall('Nx1\s*=\s*(\d*)', dataLine)[0]            
#             print nx1

        if 'Nx2' in dataLine:
            nx2 = re.findall('Nx2\s*=\s*(\d*)', dataLine)[0]            
#             print nx2

        if 'Nx3' in dataLine:
            nx3 = re.findall('Nx3\s*=\s*(\d*)', dataLine)[0]            
#             print nx3
        
        if 'nc0' in dataLine:             
            nc0 = re.sub(r'\.' , "", re.findall('nc0\s*=\s*(\d*\.\d*e\d*)', dataLine)[0])
                        
        if 'F2Fedd' in dataLine:         
            F2Fedd = re.findall('F2Fedd\s*=\s*(\d*\.\d*)', dataLine)[0]
            F2Fedd_str=re.sub(r'\.', "", F2Fedd)
                                                
        if 'beta' in dataLine:         
            beta = re.findall('beta\s*=\s*(\d*.\d*)', dataLine)[0]
            beta_str=re.sub(r'\.', "", beta)
            
            print re.findall('beta\s*=\s*(\d*.\d*)', dataLine)
            
            
except Exception, e:
    print(str(e))
    
# try:
#     for dataLine in open( locdir+'/torus1/zmp_inp', 'r').read().split('\n'):
#         if '&ggen1' in dataLine:
#             nbl1 = re.findall('nbl=(\d*),', dataLine)[0]            
#         if '&ggen2' in dataLine:
#             nbl2 = re.findall('nbl=(\d*),', dataLine)[0]            
# except Exception, e:
#     print(str(e))

        



date = time.strftime("%d/%m/%Y").split("/")
curY = date[2]
curD = date[0]
curM = date[1]

curM = datetime.date.today().strftime('%b')

dirToSave = os.getcwd()+ '/Soloviev'+curM + curY+curD+'_'+nx1+'x'+nx2+'x'+nx3+'_L'+F2Fedd+'n'+nc0+'/'

try: 
    os.stat(dirToSave)       
except:
    print "making dir:  ", dirToSave 
    os.mkdir(dirToSave)

binDir = locdir+'/bin' 
print binDir

for file in os.listdir(binDir):    
    
    if file.startswith("mhdXwind"):                                
        print("file = ", file)        
        try:
            print("dirToSave", dirToSave+file, os.path.isdir(dirToSave) )
            shutil.move(binDir+'/'+file, dirToSave+file)
#             shutil.copyfile(binDir+'/'+file, dirToSave+file)
        
        except IOError as e:
             print "I/O error({0}): {1}".format(e.errno, e.strerror)
             exit()
             
try:
    fileToCopy = locdir+'/tst/cylindrical/athinput.torus9_hydro_2D'
    fileToSave= dirToSave+'athinput.torus9_hydro_2D'
    shutil.copyfile(fileToCopy, fileToSave)
#     print(locdir+'/torus1/zmp_inp', dirToSave+'athinput.torus9_hydro_2D')
except:    
    print("Error: can't copy torus9_hydro_2D")
    print(fileToCopy, fileToSave)
    

print "done"
