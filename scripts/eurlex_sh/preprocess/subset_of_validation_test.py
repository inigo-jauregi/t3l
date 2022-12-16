import argparse
import random

def main(args):

    sample_list = []
    dev_file = open(args.dev_path)
    for line in dev_file:
        sample_list.append(line)
    dev_file.close()

    random.shuffle(sample_list)

    print(len(sample_list))

    mini_dev_list = sample_list[:args.num_for_dev]

    print('Mini dev: ', len(mini_dev_list))

    the_path = args.dev_path

    file_name = the_path.split('/')[-1].split('.')[0]
    the_path_to_folder = "/".join(the_path.split('/')[:-1])

    mini_dev_write = open(the_path_to_folder + '/' + file_name + '-' + f'_{args.num_for_dev}.json', 'w')

    for i in range(len(mini_dev_list)):
        mini_dev_write.write(mini_dev_list[i])

    mini_dev_write.close()


my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('--dev_path', type=str, required=True,
                       help="Dataset path.")
my_parser.add_argument('--num_for_dev', type=int, required=True,
                       help="Output joined dataset name.")

if __name__ == "__main__":
    my_args = my_parser.parse_args()
    main(my_args)