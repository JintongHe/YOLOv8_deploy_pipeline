# coding = utf8

import os.path


# 获取数据
import shutil


def get_data():
    cate_name_dicts = {'17': 'Shipping-Container', 	'27': 'Storage-Tank'}
    path = r'E:\work\data\storageTank_shippingContainer'
    # 创建文件路径
    image_path = path + '/images'
    label_path = path + '/labels'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    for data_type in ['train', 'val', 'test']:
        image_data_type_path = image_path + '/' + data_type
        label_data_type_path = label_path + '/' + data_type
        if not os.path.exists(image_data_type_path):
            os.makedirs(image_data_type_path)
        if not os.path.exists(label_data_type_path):
            os.makedirs(label_data_type_path)

    # 复制所需数据
    source_path = r'E:\work\data\xview_all\train_val_test'
    source_image_path = source_path + '/images'
    source_label_path = source_path + '/labels'
    for data_type in ['train', 'val', 'test']:
        image_names = os.listdir(source_image_path + '/' + data_type)
        print(len(image_names))
        index = 0
        for image_name in image_names:
            index += 1
            if index % 1000 == 0:
                print(index)
            source_image = source_image_path + '/' + data_type + '/' + image_name
            # 获取所需label
            label_list = []
            f = open(source_label_path + '/' + data_type + '/' + image_name.split('.')[0] + '.txt', encoding='utf8')
            for line in f:
                cate_id = line.split(' ')[0]
                if cate_id in cate_name_dicts.keys():
                    label_list.append(line)
            f.close()

            if len(label_list) > 0:
                target_image = image_path + '/' + data_type + '/' + image_name
                # shutil.copy(source_image, target_image)
                out = open(label_path + '/' + data_type + '/' + image_name.split('.')[0] + '.txt', 'w', encoding='utf8')
                for tmp_label in label_list:
                    tmp_arr = tmp_label.split(' ')
                    if tmp_arr[0] == '17':
                        tmp_arr[0] = '0'
                    if tmp_arr[0] == '27':
                        tmp_arr[0] = '1'
                    tmp_label = ' '.join(tmp_arr)
                    out.write(tmp_label)
                out.close()


if __name__ == '__main__':
    get_data()
