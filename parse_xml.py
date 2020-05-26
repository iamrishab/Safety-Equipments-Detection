import os
import glob
import xml.etree.ElementTree as ET

from config import *


def xml_parser(gt_xml_path, img_dir, output_path, labels):
    """
    Description: XML parser for groud truths
    """
    tree = ET.parse(gt_xml_path) 
    root = tree.getroot() 
    total_imgs = []

    for item in root.findall('image'):
        image_prefix = item.attrib['id']
        width = int(item.attrib['width'])
        height = int(item.attrib['height'])
        img_path = glob.glob(f"{img_dir}/{image_prefix}.*")
        if len(img_path)!=1 and not os.path.exists(img_path[0]) and \
                        os.path.splitext(img_path)[-1].lower() in SUPPORTED_IMG_FORMATS:
            continue
        for box in item.findall('box'):
            if box.attrib['label'] == 'head':
                xmin_norm = max(0, eval(box.attrib['xtl'])) / width
                ymin_norm = max(0, eval(box.attrib['ytl'])) / height
                xmax_norm = min(width, eval(box.attrib['xbr'])) / width
                ymax_norm = min(height, eval(box.attrib['ybr'])) / height
                
                if not xmin_norm < xmax_norm <= 1. and not ymin_norm < ymax_norm <= 1.:
                    continue
                    
                if img_path[0] not in total_imgs:
                    total_imgs.append(img_path[0])
                
                width_norm = xmax_norm - xmin_norm
                height_norm = ymax_norm - ymin_norm
                xcenter_norm = xmin_norm + width_norm / 2
                ycenter_norm = ymin_norm + height_norm / 2
                
                for attribute in box.findall('attribute'):
                    if attribute.attrib['name'] == 'has_safety_helmet':
                        # has_safety_helmet = attribute.text
                        label_index = labels['helmet']['index'][f'{attribute.text}']
                        with open(os.path.join(output_path['helmet'], f'{image_prefix}.txt'), 'a+') as fa:
                            fa.write(f'{label_index} {xcenter_norm} {ycenter_norm} {width_norm} {height_norm}\n')
                    if attribute.attrib['name'] == 'mask':
                        # mask = attribute.text
                        label_index = labels['mask']['index'][f'{attribute.text}']
                        with open(os.path.join(output_path['mask'], f'{image_prefix}.txt'), 'a+') as fa:
                            fa.write(f'{label_index} {xcenter_norm} {ycenter_norm} {width_norm} {height_norm}\n')
                            
    total_num_imgs = len(total_imgs)

    for label in labels:
        num_img_train = int(total_num_imgs * labels[label]['train_split'])
        num_img_val = int(total_num_imgs * labels[label]['validation_split'])
        num_img_test = int(total_num_imgs * labels[label]['test_split'])
        
        img_path_train = total_imgs[:num_img_train]
        img_path_val = total_imgs[num_img_train:num_img_train+num_img_val]
        img_path_test = total_imgs[num_img_train+num_img_val:]
        
        with open(labels[label]['path_train'], 'w') as fw:
            for img_path in img_path_train:
                fw.write(os.path.abspath(img_path)+'\n')
                
        with open(labels[label]['path_val'], 'w') as fw:
            for img_path in img_path_val:
                fw.write(os.path.abspath(img_path)+'\n')
                
        with open(labels[label]['path_test'], 'w') as fw:
            for img_path in img_path_test:
                fw.write(os.path.abspath(img_path)+'\n')


if __name__ == '__main__':
    for label, path in labels_save_dir.items():    
        if os.path.exists(path):
            os.system(f"rm {path}/*")
        os.makedirs(path, exist_ok=True)
    
    xml_parser(xml_annotation_path, img_dir, labels_save_dir, labels)
