import os
import glob
import xml.etree.ElementTree as ET

from config import *


def gt_parser(gt_xml_path, output_path, labels):
    with open(labels[label]['path_test']) as fr:
        file_prefixes = [os.path.splitext(os.path.basename(file_path.strip()))[0] for file_path in fr.readlines()]
        
    tree = ET.parse(gt_xml_path) 
    root = tree.getroot() 

    for item in root.findall('image'):
        image_prefix = item.attrib['id']
        if image_prefix in file_prefixes:
            width = int(item.attrib['width'])
            height = int(item.attrib['height'])
            for box in item.findall('box'):
                if box.attrib['label'] == 'head':
                    xmin = max(0, eval(box.attrib['xtl']))
                    ymin = max(0, eval(box.attrib['ytl']))
                    xmax = min(width, eval(box.attrib['xbr']))
                    ymax = min(height, eval(box.attrib['ybr']))

                    for attribute in box.findall('attribute'):
                        if attribute.attrib['name'] == 'has_safety_helmet':
                            # has_safety_helmet = attribute.text
                            label_index = labels['helmet']['index'][f'{attribute.text}']
                            with open(os.path.join(output_path['helmet'], f'{image_prefix}.txt'), 'a+') as fa:
                                fa.write(f'{attribute.text} {xmin} {ymin} {xmax} {ymax}\n')
                        if attribute.attrib['name'] == 'mask':
                            # mask = attribute.text
                            label_index = labels['mask']['index'][f'{attribute.text}']
                            with open(os.path.join(output_path['mask'], f'{image_prefix}.txt'), 'a+') as fa:
                                fa.write(f'{attribute.text} {xmin} {ymin} {xmax} {ymax}\n')


if __name__ == '__main__':
    for label, path in gt_save_dir.items():    
        if os.path.exists(path):
            os.system(f"rm {path}/*")
        os.makedirs(path, exist_ok=True)
    os.system(f"rm {img_dir}/*.txt")
    gt_parser(xml_annotation_path, gt_save_dir, labels)