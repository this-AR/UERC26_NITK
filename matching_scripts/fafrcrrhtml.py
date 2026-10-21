# import tensorflow as tf
import os
import sys
import argparse
# import facenet
# import numpy as np
# import cv2
# from datetime import datetime
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

script_folder_path = "./scripts/"
image_folder_path = "/run/user/1000/gvfs/sftp:host=40.87.43.97,user=/media/user/DATA/matching/IIT_matching/IITD_Test1/"

def html(args, name, ext):

    # folder_path = os.path.join(script_folder_path, args.folder)
    script_path = os.path.join(args.folder, args.folder.split('/')[2] + ext)
    script_file = open(script_path, 'r')

    htmlfile = open(os.path.join(args.folder, name+".html"), "w")
    htmlfile.write("<html>\n<body>\n")

    i=0
    for l in script_file:
        if i==0:
            i=1
            continue
        nums = l.split('\t')
        if len(nums)<=1:
            break
        # print(nums)
        image_path_1 = os.path.join(image_folder_path+str(nums[0]), str(nums[1])+".bmp")
        image_path_2 = os.path.join(image_folder_path+str(nums[2]), str(nums[3])+".bmp")

        htmlfile.write('<div style="width:60%;height:300px;">\n')
        htmlfile.write('<img src = "' + image_path_1 + '" alt ="image_1" style="float:left;width:50%;height:100%;object-fit:contain;">\n')
        htmlfile.write('<img src = "' + image_path_2 + '" alt ="image_2" style="float:left;width:50%;height:100%;object-fit:contain;">\n')
        htmlfile.write('</div>\n')
        htmlfile.write('<p>'+str(nums[0])+'.'+str(nums[1])+' ; '+str(nums[2])+'.'+str(nums[3])+'</p>')
    htmlfile.write("</body>\n</html>\n")
    htmlfile.close()

def htmlcrr(args, name, ext):

    # folder_path = os.path.join(script_folder_path, args.folder)
    script_path = os.path.join(args.folder, args.folder.split('/')[2] + ext)
    script_file = open(script_path, 'r')

    htmlfile = open(os.path.join(args.folder, name+".html"), "w")
    htmlfile.write("<html>\n<body>\n")

    for l in script_file:
        xx = l.split("\t")
        # print(xx)
        anchor = xx[0].split(' ')
        anchor_img = str(anchor[0])
        anchor_img_num = str(anchor[1])

        positive = xx[1].split(',')
        positive_img = str(positive[-1][0])
        positive_img_num = str(positive[0][-1])

        negative = xx[2].split(',')
        negative_img = str(negative[-1][0])
        negative_img_num = str(negative[0][-1])

        image_path_1 = os.path.join(image_folder_path+anchor_img, anchor_img_num+".bmp")
        image_path_2 = os.path.join(image_folder_path+positive_img, positive_img_num+".bmp")
        image_path_3 = os.path.join(image_folder_path+negative_img, negative_img_num+".bmp")

        htmlfile.write('<div style="width:60%;height:300px;">\n')
        htmlfile.write('<img src = "' + image_path_1 + '" alt ="image_1" style="float:left;width:33%;height:100%;object-fit:contain;">\n')
        htmlfile.write('<img src = "' + image_path_2 + '" alt ="image_2" style="float:left;width:33%;height:100%;object-fit:contain;">\n')
        htmlfile.write('<img src = "' + image_path_3 + '" alt ="image_3" style="float:left;width:33%;height:100%;object-fit:contain;">\n')
        htmlfile.write('</div>\n')
        htmlfile.write('<p>'+anchor_img+'.'+anchor_img_num+' ; '+positive_img+'.'+positive_img_num+' ; '+negative_img+'.'+negative_img_num+'</p>')


    htmlfile.write("</body>\n</html>\n")
    htmlfile.close()

        

def main(args):
    html(args, "FA", ".FA")
    html(args, "FR", ".FR")
    htmlcrr(args, "CRR", "_CRR.dat")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model', type=str,
    #     help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='./models/siamese/Ear_new_rot_3layer_gabor_initialisation_fixed_and_trainable_gabor_withNo1x1_iitd')
    # parser.add_argument('--data_dir', type=str,
    #     help='Path to the data directory',default='/home/parinaya/IITD/IITD_Test1')
    # parser.add_argument('--number_of_gallery_images', type=int,
    #     help='Number of images used for training for each subject',default=2)
    # parser.add_argument('--embedding_size', type=int,
    #     help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--folder', type=str, required=True,
        help='Script folder name')    
    parser.add_argument('--fa', type=str,
        help='Path to FA file')
    parser.add_argument('--fr', type=str,
        help='Path to FR file')
    parser.add_argument('--crr', type=str,
        help='Path to CRR file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
