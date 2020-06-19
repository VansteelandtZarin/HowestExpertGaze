# Script dat de naam van alle bestancode to rename multiple
import os

dirname = "Code/newChoco/"  # map waar je alle bestanden van een bepaald type wilt aanpassen
filetype = ".jpg"  # extensie van je type bestand
for count, filename in enumerate(os.listdir(dirname)):
    dst = "Choco" + str(count) + filetype
    src = dirname + '/' + filename
    dst = dirname + dst
    os.rename(src, dst)
