from pathlib import Path
import sys

partL=['funcPart','ModelPart']

def addPart(partList)->None:
    tempDir=Path.cwd()
    while 1:
        if tempDir.name=='imgClassification':
            break
        if tempDir==tempDir.parent:
            raise Exception("root_path: Lose!")
        tempDir=tempDir.parent
    # print("root_path:",tempDir)
    [sys.path.append(str(tempDir/part)) for part in partList]

addPart(partL)

import getData
import ResNet
import trainF

if __name__=="__main__":
    # addPart(partL)
    print(sys.path)


