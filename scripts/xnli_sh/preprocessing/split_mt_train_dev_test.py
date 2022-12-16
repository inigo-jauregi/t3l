import argparse
import random


def main(args):

    src_list = []
    train_file = open(args.train_path + '.' + args.src)
    for line in train_file:
        if len(line.strip()) > 0:
            src_list.append(line)
    train_file.close()

    tgt_list = []
    train_file = open(args.train_path + '.' + args.tgt)
    for line in train_file:
        if len(line.strip()) > 0:
            tgt_list.append(line)
    train_file.close()

    print(len(src_list))
    print(len(tgt_list))

    list_zipped = list(zip(src_list, tgt_list))
    random.shuffle(src_list)
    src_list, tgt_list = zip(*list_zipped)

    dev_src_list = src_list[:args.num_for_dev]
    dev_tgt_list = tgt_list[:args.num_for_dev]
    limit_test = args.num_for_dev + args.num_for_test
    test_src_list = src_list[args.num_for_dev:limit_test]
    test_tgt_list = tgt_list[args.num_for_dev:limit_test]
    train_src_list = src_list[limit_test:]
    train_tgt_list = tgt_list[limit_test:]

    print('Mini train: ', len(train_src_list), ' | ', len(train_tgt_list))
    print('Mini dev: ', len(dev_src_list), ' | ', len(dev_tgt_list))
    print('Mini test: ', len(test_src_list), ' | ', len(test_tgt_list))

    the_path = args.train_path

    file_name = the_path.split('/')[-1].split('.')[0]
    the_path_to_folder = "/".join(the_path.split('/')[:-1])

    # Training file
    train_src_write = open(the_path_to_folder + '/train.' + args.src, 'w')
    train_tgt_write = open(the_path_to_folder + '/train.' + args.tgt, 'w')
    for i in range(len(train_src_list)):
        train_src_write.write(train_src_list[i])
        train_tgt_write.write(train_tgt_list[i])
    train_src_write.close()
    train_tgt_write.close()

    # Dev file
    dev_src_write = open(the_path_to_folder + '/dev.' + args.src, 'w')
    dev_tgt_write = open(the_path_to_folder + '/dev.' + args.tgt, 'w')
    for i in range(len(dev_src_list)):
        dev_src_write.write(dev_src_list[i])
        dev_tgt_write.write(dev_tgt_list[i])
    dev_src_write.close()
    dev_tgt_write.close()

    # Testng file
    test_src_write = open(the_path_to_folder + '/test.' + args.src, 'w')
    test_tgt_write = open(the_path_to_folder + '/test.' + args.tgt, 'w')
    for i in range(len(test_src_list)):
        test_src_write.write(test_src_list[i])
        test_tgt_write.write(test_tgt_list[i])
    test_src_write.close()
    test_tgt_write.close()


my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('--train_path', type=str, required=True,
                       help="Dataset path.")
my_parser.add_argument('--src', type=str, required=True,
                       help="Source language.")
my_parser.add_argument('--tgt', type=str, required=True,
                       help="Target language.")
my_parser.add_argument('--num_for_dev', type=int, required=True,
                       help="Number of validation examples.")
my_parser.add_argument('--num_for_test', type=int, required=True,
                       help="Number of test examples.")


if __name__ == "__main__":
    my_args = my_parser.parse_args()
    main(my_args)