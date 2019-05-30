import os
import glob


def get_cityscapes_list_augmented(root, image_path, label_path, lst_path, is_fine=True, sample_rate=1, is_train = True):
    index = 0
    train_lst = []


    # images
    all_images = glob.glob(os.path.join(image_path, '*.png'))
    all_images.sort()
    for p in all_images:
        l = p.replace('test', 'testannot')
        if os.path.isfile(l):
            index += 1
            if index % 100 == 0:
                print "%d out of %d done." % (index, len(all_images))
            if index % sample_rate != 0:
                continue
            if is_train:
                for i in range(1, 8):
                    train_lst.append([str(index), p, l, "512", str(256 * i)])
            else:
                train_lst.append(p+','+l)
        else:
            print "dismiss %s" % (p)

    train_out = open(lst_path, "w")
    for line in train_lst:
        train_out.write(line+'\n')
    train_out.close()
if __name__ == '__main__':
    train_val = 'test'
    root = '/home/zhengxiawu/data/cityscapes/gtFine_trainvaltest'
    image_path = '/home/zhengxiawu/data/CamVid/test/'
    label_path = '/home/zhengxiawu/data/CamVid/testannot/'
    lst_path = train_val+'.txt'
    get_cityscapes_list_augmented(root, image_path, label_path, lst_path, is_fine=True, sample_rate=1,is_train=False)
