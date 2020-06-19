# als je een fout gemaakt hebt tijdens het labelen, kan je het met dit script de foutief gelabelde aanpassen naar het gewenste label
import glob
import re

misslabeled_dir = "dataset-handsv1"  # aanpassen naar map met foutief gelabelde
wrong_labelnr = 4  # labelnr dat verkeer is
right_labelnr = 2  # labelnr waarmee we het verkeere labelnr gaan vervangen

txt_file_paths = glob.glob(r"%s/*.txt" % misslabeled_dir)
for i, file_path in enumerate(txt_file_paths):
    # get image size
    with open(file_path, "r") as f_o:
        lines = f_o.readlines()
        text_converted = []
        for line in lines:
            numbers = re.findall("[0-9.]+", line)
            if numbers:
                if numbers[0] == wrong_labelnr:
                    text = "{} {} {} {} {}".format(right_labelnr, numbers[1], numbers[2], numbers[3], numbers[4])
                text_converted.append(text)
        # Write file
        with open(file_path, 'w') as fp:
            for item in text_converted:
                fp.writelines("%s\n" % item)
