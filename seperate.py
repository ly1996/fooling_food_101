import os, shutil

TESTSET_PATH = '../test-set/'
TRAINSET_PATH = '../images/'
TESTSET_FILE_PATH = '../meta/test.txt'

f = open(TESTSET_FILE_PATH)
dirs = f.readlines()
for d in dirs:
  orgDir = TRAINSET_PATH + d
  orgDir = orgDir[0:-1] + '.jpg'
  if not os.path.isfile(orgDir):
    print('%s does not exist'%(orgDir))
  else:
    fpath, fname = os.path.split(d)
    newDir = TESTSET_PATH + d
    newDir = newDir[0:-1] + '.jpg'
    newPath = TESTSET_PATH + fpath
    if not os.path.exists(newPath):
      os.makedirs(newPath)
    shutil.move(orgDir, newDir)
    print('move %s -> %s'%(orgDir, newDir))
