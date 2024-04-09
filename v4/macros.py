with open("/mnt/data/smart_health_02/dailinrui/code/cmu/aug/load/totalseg.txt", 'r') as fp:
    seg_id_and_names = fp.readlines()
LABEL_MAPPING = {line.split('|')[1].strip(): int(line.split('|')[0].strip()) for line in seg_id_and_names}